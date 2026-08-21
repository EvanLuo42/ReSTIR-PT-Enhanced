#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.Scene;
import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.LobeType;
__exported import Rendering.Lights.EnvMapSampler;
__exported import Rendering.Lights.EmissiveLightSampler;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.Surface;
import RenderPasses.ReSTIRPTPass.Common.GRIS;
import RenderPasses.ReSTIRPTPass.Common.Lights;

ShiftResult makeFailedShift()
{
    ShiftResult result = makeInvalidShiftResult();
    result.flags = kRecordReplayFailed;
    return result;
}

float3 shiftedContribution(PathSample src, float3 prefixThp, float3 fNew, float3 fAtNew, bool visible)
{
    if (!visible || !any(src.suffixThroughput > 0.f))
        return 0.f;
    const float3 F = prefixThp * fNew * fAtNew * src.suffixThroughput;
    return isFiniteFloat3(F) ? F : 0.f;
}

float hybridJacobian(float newDensity, float oldDensity)
{
    const float j = (oldDensity > 0.f) ? (newDensity / oldDensity) : 0.f;
    return (j > 0.f && isfinite(j) && isfinite(newDensity)) ? j : 0.f;
}

bool replayScatterStep(
    PathSample src,
    uint scatterVertex,
    RestirParams params,
    uint2 pixel,
    inout SampleGenerator sg,
    inout Surface current,
    inout float3 thp,
    inout InteriorList interiors,
    inout uint rejectedHits,
    out Surface next,
    out bool missed,
    out BSDFSample bs
)
{
    next = {};
    missed = false;
    bs = {};
    let mi = getSurfaceMI(current, params, pixel);
    if (!mi.sample(current.sd, sg, bs, params.hasFeatureFlag(kEnableBSDFSampling)) ||
        !isValidScatterSample(bs))
        return false;

    const uint expected = unpackLobe(src.sampledLobes, scatterVertex);
    if (expected != 0u && (bs.lobeType & 0xffu) != expected)
        return false;

    updateInteriorAfterScatter(interiors, current, bs);
    thp *= bs.weight;
    const float3 origin = offsetScatterOrigin(current, bs);
    return traceMediumHit(origin, bs.wo, interiors, rejectedHits, thp, params, pixel, false, next, missed, false);
}

bool replayPrefix(
    PathSample src,
    Surface destPrimary,
    uint k,
    RestirParams params,
    uint2 pixel,
    out Surface prefix,
    out float3 prefixThp,
    out InteriorList interiors
)
{
    prefix = destPrimary;
    prefixThp = 1.f;
    interiors = makeEmptyInteriorList();
    if (k <= 2u)
        return destPrimary.valid;

    SampleGenerator sg = SampleGenerator(src.randomSeed, 0u);
    Surface current = destPrimary;
    float3 thp = 1.f;
    uint rejectedHits = 0u;

    const uint lastReplay = k - 2u;
    for (uint v = 1u; v <= lastReplay; ++v)
    {
        Surface next;
        bool missed;
        BSDFSample bs;
        if (!replayScatterStep(
            src, v, params, pixel, sg, current, thp, interiors, rejectedHits, next, missed, bs
        ) || missed)
            return false;
        current = next;
    }

    prefix = current;
    prefixThp = thp;
    return prefix.valid && isFiniteFloat3(prefixThp);
}

bool evalNeeEndpointReconnect(
    PathSample src,
    Surface prefix,
    float3 prefixThp,
    RestirParams params,
    uint2 pixel,
    GeometryTime srcTime,
    GeometryTime destTime,
    out float3 contribution,
    out float jacobian,
    out PathSample shifted
)
{
    contribution = 0.f;
    jacobian = 0.f;
    shifted = src;

    if ((src.flags & kPathEndsOnAnalytic) != 0u)
    {
        const AnalyticNeeEval ev = evalAnalyticNee(src, prefix.position);
        if (!ev.valid || !any(ev.Li > 0.f))
            return false;
        const float3 dir = normalize(ev.dir);
        SampleGenerator evalSg = SampleGenerator(src.randomSeed, 1u);
        let prefixMI = getSurfaceMI(prefix, params, pixel);
        const float3 fNew = prefixMI.eval(prefix.sd, dir, evalSg);
        if (!any(fNew > 0.f))
            return false;
        const bool visible = ev.infinite
            ? traceDirectionalVisibility(offsetRayOrigin(prefix, dir), dir, FLT_MAX)
            : traceReconnectVisibility(prefix, ev.dest, ev.destNormal);
        if (!visible)
            return false;

        contribution = prefixThp * fNew * ev.Li;
        jacobian = 1.f;
        if (!any(contribution > 0.f) || !isFiniteFloat3(contribution))
            return false;

        shifted.contribution = contribution;
        shifted.target = targetPdf(contribution);
        shifted.proposalPdf = 1.f;
        shifted.incidentDirection = dir;
        shifted.reconnectionPosition = ev.dest;
        shifted.reconnectionNormal = ev.destNormal;
        return shifted.target > 0.f;
    }

    const bool envEndpoint = (src.flags & kPathEndsOnEnvironment) != 0u;
    float3 dest = src.reconnectionPosition;
    float3 destNormal = src.reconnectionNormal;
    if (!envEndpoint)
    {
        if (!resolveReconnectEndpoint(
            src.reconnectionHit,
            src.lightTriangleIndex,
            src.reconnectionPosition,
            src.reconnectionNormal,
            srcTime,
            destTime,
            dest,
            destNormal
        ))
            return false;
    }
    float3 dir = src.incidentDirection;
    if (!envEndpoint)
    {
        const float3 toDest = dest - prefix.position;
        const float dist = length(toDest);
        if (dist < 1e-5f)
            return false;
        dir = toDest / dist;
    }
    else if (dot(dir, dir) < 1e-12f)
        return false;
    dir = normalize(dir);

    SampleGenerator evalSg = SampleGenerator(src.randomSeed, 1u);
    let prefixMI = getSurfaceMI(prefix, params, pixel);
    const float3 fNew = prefixMI.eval(prefix.sd, dir, evalSg);
    const float pBsdf = prefixMI.evalPdf(prefix.sd, dir, true);
    if (!any(fNew > 0.f) || !any(src.suffixThroughput > 0.f))
        return false;
    const bool visible = envEndpoint
        ? traceDirectionalVisibility(offsetRayOrigin(prefix, dir), dir, FLT_MAX)
        : traceReconnectVisibility(prefix, dest, destNormal);
    if (!visible)
        return false;

    const float pNew = envEndpoint
        ? src.reconnectDensity
        : lightSolidAnglePdfFromArea(
            src.reconnectDensity,
            prefix.position,
            dest,
            destNormal
        );
    if (!(pNew > 0.f) || !isfinite(pNew))
        return false;

    float mNew = 1.f;
    if (envEndpoint || src.lightTriangleIndex != kInvalidIndex)
        mNew = evalBalanceMIS(float(getPathRisM(src)), pNew, 1.f, pBsdf);
    if (!(mNew > 0.f))
        return false;

    contribution = prefixThp * fNew * mNew * src.suffixThroughput / pNew;
    jacobian = 1.f;
    if (!any(contribution > 0.f) || !isFiniteFloat3(contribution))
        return false;

    shifted.contribution = contribution;
    shifted.target = targetPdf(contribution);
    shifted.proposalPdf = mNew;
    shifted.reconnectionPosition = dest;
    shifted.reconnectionNormal = destNormal;
    return shifted.target > 0.f;
}

bool evalReconnect(
    PathSample src,
    Surface prefix,
    float3 prefixThp,
    InteriorList interiors,
    RestirParams params,
    uint2 pixel,
    EmissiveLightSampler emissiveSampler,
    GeometryTime srcTime,
    GeometryTime destTime,
    out float3 contribution,
    out float jacobian,
    out PathSample shifted
)
{
    contribution = 0.f;
    jacobian = 0.f;
    shifted = src;

    if (isLightEndpointReconnect(src))
        return evalNeeEndpointReconnect(
            src, prefix, prefixThp, params, pixel, srcTime, destTime, contribution, jacobian, shifted
        );

    float3 dest = src.reconnectionPosition;
    float3 destNormal = src.reconnectionNormal;
    if (!resolveReconnectEndpoint(
        src.reconnectionHit,
        src.lightTriangleIndex,
        src.reconnectionPosition,
        src.reconnectionNormal,
        srcTime,
        destTime,
        dest,
        destNormal
    ))
        return false;

    const float3 toDest = dest - prefix.position;
    const float dist = length(toDest);
    if (dist < 1e-5f)
        return false;
    const float3 dir = toDest / dist;

    SampleGenerator evalSg = SampleGenerator(src.randomSeed, 1u);
    let prefixMI = getSurfaceMI(prefix, params, pixel);
    const uint prefixLobe = unpackLobe(src.sampledLobes, src.reconnectionVertex - 1u);
    if (!hasExactSingleLobe(prefixMI.getLobeTypes(prefix.sd), prefixLobe))
        return false;
    const float3 fNew = prefixMI.eval(prefix.sd, dir, evalSg);
    const float pFromNew = prefixMI.evalPdf(prefix.sd, dir, true);
    const float gNew = geometryTerm(prefix.position, dest, destNormal);
    if (!any(fNew > 0.f) || !(gNew > 0.f) || !(pFromNew > 0.f))
        return false;
    if (!traceReconnectVisibility(prefix, dest, destNormal))
        return false;

    if (isBsdfHitEmissiveEndpointReconnect(src))
    {
        const float pLight = evalStoredEmissivePdf(
            prefix.position,
            prefix.sd.getOrientedFaceNormal(),
            src.lightTriangleIndex,
            dest,
            destNormal,
            params,
            emissiveSampler
        );
        const float misNew = evalBalanceMIS(1.f, pFromNew, float(getPathRisM(src)), pLight);
        if (!(misNew > 0.f) || !any(src.suffixThroughput > 0.f))
            return false;

        contribution = prefixThp * fNew * misNew * src.suffixThroughput / pFromNew;
        jacobian = hybridJacobian(pFromNew * gNew, src.reconnectDensity);
        if (!any(contribution > 0.f) || !isFiniteFloat3(contribution) || !(jacobian > 0.f))
            return false;

        shifted.contribution = contribution;
        shifted.target = targetPdf(contribution);
        shifted.proposalPdf = misNew;
        shifted.reconnectDensity = pFromNew * gNew;
        shifted.reconnectionPosition = dest;
        shifted.reconnectionNormal = destNormal;
        return shifted.target > 0.f;
    }

    float pAtNew = 1.f;
    float3 fAtNew = 1.f;
    float3 woAt = src.incidentDirection;
    const bool reconnectToSurface = packedHitIsValid(src.reconnectionHit);
    if (reconnectToSurface && (src.flags & kPathHasReconnectBSDF) != 0u)
    {
        const HitInfo hit = unpackStoredHit(src.reconnectionHit);
        woAt = mapWorldDirBetweenFrames(hit.getInstanceID(), src.incidentDirection, srcTime, destTime);
        Surface atK;
        const float3 origin = offsetRayOrigin(prefix, dir);
        InteriorList destInteriors = interiors;
        if (!prefix.sd.mtl.isThinSurface() && dot(prefix.faceNormal, dir) < 0.f)
            destInteriors.handleIntersection(prefix.sd.materialID, prefix.sd.mtl.getNestedPriority(), prefix.sd.frontFacing);
        bool loaded = false;
        if (destTime == GeometryTime::Previous)
        {
            atK = {};
            loaded = loadPreviousShading(hit, dir, atK.sd);
            if (loaded)
            {
                atK.hit = hit;
                atK.position = dest;
                atK.faceNormal = atK.sd.faceN;
                atK.viewDir = -dir;
                atK.isPrimary = false;
                atK.sd.IoR = computeOutsideIoR(destInteriors, atK.sd.materialID, atK.sd.frontFacing);
                atK.valid = true;
            }
        }
        else
            loaded = loadSurface(hit, origin, dir, false, params, pixel, destInteriors, atK, false);
        if (loaded && any(woAt != 0.f))
        {
            let atMI = getSurfaceMI(atK, params, pixel);
            const uint atLobe = unpackLobe(src.sampledLobes, src.reconnectionVertex);
            if (atLobe != 0u && !hasExactSingleLobe(atMI.getLobeTypes(atK.sd), atLobe))
                return false;
            pAtNew = atMI.evalPdf(atK.sd, woAt, true);
            fAtNew = atMI.eval(atK.sd, woAt, evalSg);
        }
        else
            return false;
        if (!(pAtNew > 0.f) || !any(fAtNew > 0.f))
            return false;
    }

    const float newDensity = pFromNew * gNew * pAtNew;
    jacobian = hybridJacobian(newDensity, src.reconnectDensity);
    contribution = shiftedContribution(src, prefixThp, fNew, fAtNew, true);
    if (!any(contribution > 0.f) || !(jacobian > 0.f))
        return false;

    shifted.contribution = contribution;
    shifted.target = targetPdf(contribution);
    shifted.reconnectDensity = newDensity;
    shifted.reconnectionPosition = dest;
    shifted.reconnectionNormal = destNormal;
    shifted.incidentDirection = woAt;
    return shifted.target > 0.f;
}

bool replayFullPath(
    PathSample src,
    Surface destPrimary,
    RestirParams params,
    uint2 pixel,
    EnvMapSampler envMapSampler,
    EmissiveLightSampler emissiveSampler,
    out float3 contribution,
    out PathSample shifted
)
{
    contribution = 0.f;
    shifted = src;
    if ((src.flags & kPathEndsWithNEE) != 0u)
        return false;
    if (!destPrimary.valid)
        return false;

    const bool envEnd = (src.flags & kPathEndsOnEnvironment) != 0u;
    const uint scatterCount = max(getPathLength(src), 1u) - 1u;
    if (scatterCount == 0u)
        return false;

    SampleGenerator sg = SampleGenerator(src.randomSeed, 0u);
    Surface current = destPrimary;
    float3 thp = 1.f;
    float shiftedMis = 1.f;
    InteriorList interiors = makeEmptyInteriorList();
    uint rejectedHits = 0u;

    for (uint v = 1u; v <= scatterCount; ++v)
    {
        Surface next;
        bool missed;
        BSDFSample bs;
        const bool hitSomething = replayScatterStep(
            src, v, params, pixel, sg, current, thp, interiors, rejectedHits, next, missed, bs
        );

        if (v < scatterCount)
        {
            if (!hitSomething)
                return false;
            current = next;
            continue;
        }

        if (!hitSomething && !missed)
            return false;

        const uint risM = getPathRisM(src);
        float mis = 1.f;
        if (envEnd)
        {
            if (!missed)
                return false;
#if USE_ENV_LIGHT
            const float3 Le = gScene.envMap.eval(bs.wo, 0.f);
            mis = bsdfLightMIS(params, bs.isLobe(LobeType::Delta), bs.pdf, evalEnvTypePdf(bs.wo, envMapSampler), float(risM));
            contribution = thp * Le * mis;
#else
            return false;
#endif
        }
        else
        {
            if (!hitSomething)
                return false;
            const float lightPdf = evalHitEmissivePdf(
                current.position,
                current.sd.getOrientedFaceNormal(),
                next.hit,
                next.position,
                next.sd.getOrientedFaceNormal(),
                params,
                emissiveSampler
            );
            mis = bsdfLightMIS(params, bs.isLobe(LobeType::Delta), bs.pdf, lightPdf, float(risM));
            applySurfaceMaterial(next, params, pixel);
            contribution = thp * next.properties.emission * mis;
        }
        shiftedMis = mis;
    }

    if (!any(contribution > 0.f) || !isFiniteFloat3(contribution))
        return false;

    shifted.contribution = contribution;
    shifted.target = targetPdf(contribution);
    shifted.proposalPdf = shiftedMis;
    shifted.reconnectionVertex = kInvalidIndex;
    return shifted.target > 0.f;
}

bool needsRandomReplay(PathSample sample)
{
    if (!isValidPath(sample))
        return false;
    if (sample.reconnectionVertex == kInvalidIndex)
        return true;
    return sample.reconnectionVertex > 2u;
}

ShiftResult identityShift(PathSample sample)
{
    return encodeShiftResult(sample, 1.f);
}

ShiftResult evaluateHybridShift(
    PathSample src,
    Surface destPrimary,
    RestirParams params,
    uint2 destPixel,
    EnvMapSampler envMapSampler,
    EmissiveLightSampler emissiveSampler,
    GeometryTime srcTime,
    GeometryTime destTime
)
{
    if (!isValidPath(src) || !destPrimary.valid || !(src.target > 0.f))
        return makeFailedShift();

    const uint k = src.reconnectionVertex;

    if (k == kInvalidIndex)
    {
        float3 contribution;
        PathSample shifted;
        if (!replayFullPath(src, destPrimary, params, destPixel, envMapSampler, emissiveSampler, contribution, shifted))
            return makeFailedShift();

        return encodeShiftResult(shifted, 1.f);
    }

    Surface prefix;
    float3 prefixThp;
    InteriorList interiors;
    if (!replayPrefix(src, destPrimary, max(k, 2u), params, destPixel, prefix, prefixThp, interiors))
        return makeFailedShift();

    float3 contribution;
    float jacobian;
    PathSample shifted;
    if (!evalReconnect(
        src, prefix, prefixThp, interiors, params, destPixel, emissiveSampler, srcTime, destTime, contribution, jacobian, shifted
    ))
        return makeFailedShift();

    return encodeShiftResult(shifted, jacobian);
}
