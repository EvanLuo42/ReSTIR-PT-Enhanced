#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Scene.HitInfo;
import Utils.Sampling.SampleGenerator;
import Utils.Math.HashUtils;
import Utils.Math.BitTricks;
import Utils.Color.ColorHelpers;
import Utils.Geometry.GeometryHelpers;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.LobeType;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.Surface;
import RenderPasses.ReSTIRPTPass.Common.Lights;
import RenderPasses.ReSTIRPTPass.Common.GRIS;

struct WalkVertex
{
    float3 position;
    float3 normal;
    float3 throughput;
    float3 wo;
    float3 fToNext;
    uint4 packedHit;
    float bsdfPdf;
    float roughnessAlpha;
    uint scatterFlags;
};

static const uint kWalkLobeMask = 0xffu;
static const uint kWalkDelta = 1u << 8u;
static const uint kWalkExactLobePdf = 1u << 9u;

struct ReconnectCache
{
    float3 reconnectionPosition;
    float reconnectDensity;
    uint reconnectionVertex;
    float3 reconnectionNormal;
    uint flags;
    float3 incidentDirection;
    float3 denominator;
    uint4 reconnectionHit;
};

struct PathTree
{
    uint2 randomSeed;
    uint2 packedLobes;
    uint nextCandidateID;
    float primaryFootprint;
    float pathUcw;
    ReconnectCache reconnect;
    GRISState gris;
};

SampleGenerator makePathGenerator(uint2 seed)
{
    return SampleGenerator(seed, 0u);
}

SampleGenerator makeAuxGenerator(uint2 pixel, uint frame, uint salt)
{
    return SampleGenerator(pixel, frame * 747796405u + salt);
}

PathSample makeEmptyPath()
{
    PathSample s = {};
    s.reconnectionVertex = kInvalidIndex;
    s.lightTriangleIndex = kInvalidIndex;
    return s;
}

PathReservoir makeEmptySourceReservoir()
{
    PathReservoir r = makeEmptyPathReservoir();
    r.confidence = 1.f;
    return r;
}

void setReconnectSuffix(
    inout PathSample sample,
    float3 prefixThroughput,
    float3 connectionWeight,
    float3 reconnectionWeight,
    bool reconnectToNeeEndpoint,
    float neeSolidAnglePdf
)
{
    if (reconnectToNeeEndpoint)
    {

        sample.suffixThroughput = neeSolidAnglePdf > 0.f
            ? safeComponentRatio(
                sample.contribution * neeSolidAnglePdf,
                prefixThroughput * connectionWeight * sample.proposalPdf
            )
            : float3(1.f, 1.f, 1.f);
    }
    else
        sample.suffixThroughput = safeComponentRatio(
            sample.contribution,
            prefixThroughput * connectionWeight * reconnectionWeight
        );
}

void cacheInternalReconnection(
    inout ReconnectCache cache,
    float primaryFootprint,
    WalkVertex prefix,
    WalkVertex at,
    uint k,
    RestirParams params
)
{
    if (cache.reconnectionVertex != kInvalidIndex || k < 2u)
        return;
    if ((prefix.scatterFlags & kWalkExactLobePdf) == 0u ||
        (prefix.scatterFlags & kWalkDelta) != 0u ||
        (at.scatterFlags & kWalkExactLobePdf) == 0u ||
        (at.scatterFlags & kWalkDelta) != 0u ||
        !(prefix.bsdfPdf > 0.f) || !(at.bsdfPdf > 0.f) ||
        !any(prefix.fToNext > 0.f) || !any(at.fToNext > 0.f))
        return;

    const bool skipInverse = isDiffuseLobe(at.scatterFlags & kWalkLobeMask);
    const float roughness = sampledLobeAlpha(
        prefix.scatterFlags & kWalkLobeMask,
        prefix.roughnessAlpha
    );
    float gFwd;
    if (!passesEnhancedReconnectionGate(
        prefix.position,
        prefix.normal,
        at.position,
        at.normal,
        prefix.bsdfPdf,
        at.bsdfPdf,
        skipInverse,
        primaryFootprint,
        roughness,
        params,
        gFwd
    ))
        return;

    cache.reconnectionPosition = at.position;
    cache.reconnectionNormal = at.normal;
    cache.reconnectionHit = at.packedHit;
    cache.incidentDirection = at.wo;
    cache.reconnectDensity = prefix.bsdfPdf * gFwd * at.bsdfPdf;
    cache.reconnectionVertex = k;
    cache.flags = kPathHasReconnectBSDF;
    cache.denominator = prefix.throughput * prefix.fToNext * at.fToNext;
}

bool applyCachedReconnection(inout PathSample sample, ReconnectCache cache)
{
    if (cache.reconnectionVertex == kInvalidIndex)
        return false;

    sample.reconnectionVertex = cache.reconnectionVertex;
    sample.reconnectionPosition = cache.reconnectionPosition;
    sample.reconnectionNormal = cache.reconnectionNormal;
    sample.reconnectionHit = cache.reconnectionHit;
    sample.incidentDirection = cache.incidentDirection;
    sample.reconnectDensity = cache.reconnectDensity;
    sample.flags = (sample.flags & ~kPathHasReconnectBSDF) | cache.flags;
    sample.suffixThroughput = safeComponentRatio(sample.contribution, cache.denominator);
    return true;
}

bool chooseReconnection(
    inout PathSample sample,
    ReconnectCache cache,
    uint2 packedLobes,
    float primaryFootprint,
    WalkVertex endpointPrefix,
    uint lastSurface,
    bool extraLightVertex,
    float pConnectFrom,
    float3 fLightEdge,
    RestirParams params
)
{
    sample.sampledLobes = packedLobes;
    const uint d = extraLightVertex ? lastSurface + 1u : lastSurface;

    if (cache.reconnectionVertex <= d &&
        applyCachedReconnection(sample, cache))
        return true;

    const bool neeEndpoint = extraLightVertex &&
        (sample.flags & kPathEndsWithNEE) != 0u;
    if (neeEndpoint)
    {
        PathSample trial = sample;
        trial.reconnectionVertex = d;
        trial.flags = trial.flags & ~kPathHasReconnectBSDF;

        const bool analyticNee = (trial.flags & kPathEndsOnAnalytic) != 0u;
        const bool analyticArea = analyticNee && isAnalyticAreaLight(trial.lightTriangleIndex);
        const bool analyticIdentity = analyticNee && !analyticArea;
        const bool infiniteNee = (trial.flags & kPathEndsOnEnvironment) != 0u || analyticIdentity;

        bool accepted = analyticIdentity;
        bool forced = false;
        float gFwd = 0.f;
        if (!analyticIdentity)
        {
            accepted = passesEnhancedReconnectionGate(
                endpointPrefix.position,
                endpointPrefix.normal,
                trial.reconnectionPosition,
                trial.reconnectionNormal,
                pConnectFrom,
                1.f,
                true,
                primaryFootprint,
                endpointPrefix.roughnessAlpha,
                params,
                gFwd
            );
            forced = !accepted && params.hasFeatureFlag(kEnableForcedNEEReconnect);
        }
        if (accepted || forced)
        {
            trial.reconnectDensity = infiniteNee
                ? pConnectFrom
                : (pConnectFrom > 0.f && gFwd > 0.f ? pConnectFrom * gFwd : 0.f);
            setReconnectSuffix(
                trial,
                endpointPrefix.throughput,
                fLightEdge,
                1.f,
                true,
                pConnectFrom
            );
            sample = trial;
            return true;
        }
    }
    else if (!extraLightVertex &&
        (endpointPrefix.scatterFlags & kWalkExactLobePdf) != 0u &&
        (endpointPrefix.scatterFlags & kWalkDelta) == 0u &&
        endpointPrefix.bsdfPdf > 0.f && any(endpointPrefix.fToNext > 0.f))
    {

        PathSample trial = sample;
        trial.reconnectionVertex = d;
        trial.incidentDirection = 0.f;
        trial.flags = trial.flags & ~kPathHasReconnectBSDF;
        const float roughness = sampledLobeAlpha(
            endpointPrefix.scatterFlags & kWalkLobeMask,
            endpointPrefix.roughnessAlpha
        );
        float gFwd;
        if (passesEnhancedReconnectionGate(
            endpointPrefix.position,
            endpointPrefix.normal,
            trial.reconnectionPosition,
            trial.reconnectionNormal,
            endpointPrefix.bsdfPdf,
            1.f,
            true,
            primaryFootprint,
            roughness,
            params,
            gFwd
        ))
        {
            trial.reconnectDensity = endpointPrefix.bsdfPdf * gFwd;

            trial.suffixThroughput = safeComponentRatio(
                sample.contribution * endpointPrefix.bsdfPdf,
                endpointPrefix.throughput * endpointPrefix.fToNext * sample.proposalPdf
            );
            sample = trial;
            return true;
        }
    }

    sample.reconnectionVertex = kInvalidIndex;
    return false;
}

void considerCandidate(inout PathTree tree, PathSample sample, float sourceUcw, inout SampleGenerator grisSg)
{
    sample.randomSeed = tree.randomSeed;
    sample.candidateID = tree.nextCandidateID++;
    sample.target = targetPdf(sample.contribution);
    if (!(sample.target > 0.f) || !isFiniteFloat3(sample.contribution) || !isValidPath(sample))
        return;

    const float ucw = isfinite(sourceUcw) && sourceUcw > 0.f ? sourceUcw : 0.f;
    const float weight = sample.target * ucw;
    grisUpdate(tree.gris, sample, weight, 1.f, grisSg);
}

void addNEECandidate(
    inout PathTree tree,
    RestirParams params,
    uint surfaceIndex,
    WalkVertex surfaceVertex,
    LightSample light,
    float3 f,
    float mis,
    inout SampleGenerator grisSg
)
{
    if (!light.valid || !any(f > 0.f))
        return;

    const float3 contrib = surfaceVertex.throughput * f * light.Li * mis;
    if (!any(contrib > 0.f) || !isFiniteFloat3(contrib))
        return;

    PathSample sample = makeEmptyPath();
    sample.flags = neePathFlags(light.lightType);
    sample.techniqueKey = packTechniqueKey(
        TechniqueKind::NEE,
        1u,
        surfaceIndex + 1u,
        surfaceIndex + 1u
    );
    sample.contribution = contrib;
    sample.proposalPdf = mis;
    sample.incidentDirection = light.dir;
    sample.reconnectionPosition = light.dest;
    sample.reconnectionNormal = light.normal;
    sample.reconnectionHit = 0u;
    sample.lightTriangleIndex = light.triangleIndex;

    chooseReconnection(
        sample,
        tree.reconnect,
        tree.packedLobes,
        tree.primaryFootprint,
        surfaceVertex,
        surfaceIndex,
        true,
        light.pdf,
        f,
        params
    );
    sample.flags = packPathRisM(sample.flags, light.risM);
    considerCandidate(tree, sample, tree.pathUcw * light.risUcw, grisSg);
}

bool traceSelectedLight(LightSample light)
{
    if (light.lightType == 0u || light.distance >= 0.5f * FLT_MAX)
        return traceDirectionalVisibility(light.origin, light.dir, FLT_MAX);
    float3 dest = light.dest;
    if (light.lightType == 1u && dot(light.normal, light.normal) > 1e-12f)
    {
        float3 n = dot(light.normal, -light.dir) >= 0.f ? light.normal : -light.normal;
        dest = computeRayOrigin(dest, n);
    }
    return traceVisibility(light.origin, dest);
}

void addHitLightCandidate(
    inout PathTree tree,
    RestirParams params,
    uint hitIndex,
    WalkVertex endpointPrefix,
    WalkVertex endpoint,
    float3 endpointThroughput,
    float3 Le,
    float mis,
    bool envMiss,
    uint lightTriangleIndex,
    uint risM,
    inout SampleGenerator grisSg
)
{
    const float3 contrib = endpointThroughput * Le * mis;
    if (!any(contrib > 0.f) || !isFiniteFloat3(contrib))
        return;

    PathSample sample = makeEmptyPath();
    sample.flags = packPathRisM(cameraHitFlags(envMiss), risM);
    sample.techniqueKey = packTechniqueKey(
        envMiss ? TechniqueKind::NEE : TechniqueKind::CameraHit,
        envMiss ? 1u : 0u,
        hitIndex,
        envMiss ? hitIndex + 1u : hitIndex
    );
    sample.contribution = contrib;
    sample.proposalPdf = mis;
    sample.incidentDirection = envMiss ? endpointPrefix.wo : 0.f;
    sample.lightTriangleIndex = lightTriangleIndex;
    if (envMiss)
    {
        sample.reconnectionPosition = endpointPrefix.position + endpointPrefix.wo;
        sample.reconnectionNormal = 0.f;
        sample.reconnectionHit = 0u;
    }
    else
    {
        sample.reconnectionPosition = endpoint.position;
        sample.reconnectionNormal = endpoint.normal;
        sample.reconnectionHit = endpoint.packedHit;
#if USE_EMISSIVE_LIGHTS
        if (lightTriangleIndex != kInvalidIndex)
            sample.reconnectionNormal = gScene.lightCollection.getTriangle(lightTriangleIndex).normal;
#endif
    }

    const float pFrom = endpointPrefix.bsdfPdf;
    const float3 fEdge = any(endpointPrefix.fToNext != 0.f) ? endpointPrefix.fToNext : 1.f;
    chooseReconnection(
        sample,
        tree.reconnect,
        tree.packedLobes,
        tree.primaryFootprint,
        endpointPrefix,
        hitIndex,
        envMiss,
        pFrom,
        fEdge,
        params
    );
    considerCandidate(tree, sample, tree.pathUcw, grisSg);
}

PathReservoir samplePathTree(
    RestirParams params,
    EnvMapSampler envMapSampler,
    EmissiveLightSampler emissiveSampler,
    StructuredBuffer<LightTileEntry> lightTiles,
    Texture2D<PackedHitInfo> vbuffer,
    uint2 pixel,
    out float3 primaryEmission
)
{
    primaryEmission = 0.f;

    PathTree tree = {};
    tree.reconnect.reconnectionVertex = kInvalidIndex;
    tree.gris = makeGRISState();
    tree.pathUcw = 1.f;

    const uint seed = blockCipherTEA(interleave_32bit(pixel), params.frameIndex).x;
    tree.randomSeed = uint2(seed, params.frameIndex);
    SampleGenerator pathSg = makePathGenerator(tree.randomSeed);
    SampleGenerator neeSg = makeAuxGenerator(pixel, params.frameIndex, 0x9e3779b9u);
    SampleGenerator rrSg = makeAuxGenerator(pixel, params.frameIndex, 0x85ebca6bu);
    SampleGenerator grisSg = makeAuxGenerator(pixel, params.frameIndex, 0xc2b2ae35u);

    const Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
    const HitInfo primaryHit = unpackHitInfo(vbuffer[pixel]);
    if (!primaryHit.isValid())
    {
#if USE_ENV_LIGHT
        primaryEmission = gScene.envMap.eval(cameraRay.dir, 0.f);
#endif
        return makeEmptySourceReservoir();
    }

    InteriorList interiors = makeEmptyInteriorList();
    uint rejectedHits = 0u;
    Surface surface;
    if (!loadSurfaceShading(primaryHit, cameraRay.origin, cameraRay.dir, true, surface))
        return makeEmptySourceReservoir();
    surface.sd.IoR = computeOutsideIoR(interiors, surface.sd.materialID, surface.sd.frontFacing);
    IMaterialInstance mi = getSurfaceMI(surface, params, pixel);
    applySurfaceMaterial(surface, mi);

    primaryEmission = surface.properties.emission;
    tree.primaryFootprint = primaryRayFootprint(cameraRay.origin, surface.position, surface.faceNormal);

    WalkVertex previous = {};
    WalkVertex current = {};
    current.position = surface.position;
    current.normal = surface.sd.getOrientedFaceNormal();
    current.throughput = 1.f;
    current.packedHit = packStoredHit(primaryHit);
    current.roughnessAlpha = surface.roughnessAlpha;

    const uint maxD = min(params.maxPathLength, kMaxPathVertices - 1u);
    uint diffuseBounces = 0u;
    uint specularBounces = 0u;
    uint transmissionBounces = 0u;

    for (uint bounce = 1u; bounce < maxD; ++bounce)
    {
        if (params.hasFeatureFlag(kDisableCaustics) && diffuseBounces > 0u &&
            surface.properties.roughness <= params.specularRoughnessThreshold)
        {
            surface.sd.mtl.setActiveLobes((uint)LobeType::Diffuse);
            mi = getSurfaceMI(surface, params, pixel);
        }

        const uint lobeTypes = mi.getLobeTypes(surface.sd);
        const bool hasNonDelta = (lobeTypes & (uint)LobeType::NonDelta) != 0;
        const uint risM = neeRisM(params, bounce);
        const bool insideVolume = !interiors.isEmpty();
        const bool allowVolumeLights =
            !insideVolume || params.hasFeatureFlag(kUseLightsInDielectricVolumes);
        const bool allowPrimaryNEE =
            bounce > 1u || !params.hasFeatureFlag(kDisableDirectIllumination);
        if (hasNonDelta && params.hasFeatureFlag(kEnableNEE) && allowVolumeLights && allowPrimaryNEE)
        {
            LightSample light;
            if (sampleUnifiedLight(surface, mi, envMapSampler, emissiveSampler, lightTiles, pixel, bounce, neeSg, params, light))
            {
                const float3 f = light.hasBsdfF ? light.bsdfF : mi.eval(surface.sd, light.dir, neeSg);
                float mis = 1.f;
                if (params.hasFeatureFlag(kEnableMIS) && light.lightType != 2u)
                {
                    const float scatterPdf = mi.evalPdf(surface.sd, light.dir, true);
                    const float m = float(max(light.risM, 1u));
                    mis = evalBalanceMIS(m, light.pdf, 1.f, scatterPdf);
                }
                if (any(f * light.Li * mis > 0.f) && traceSelectedLight(light))
                    addNEECandidate(tree, params, bounce, current, light, f, mis, grisSg);
            }
        }

        if (params.hasFeatureFlag(kEnableInitialRussianRoulette) && bounce >= params.initialRussianRouletteStart)
        {
            const float survival = max(params.initialRussianRouletteMinSurvival, saturate(luminance(current.throughput)));
            if (sampleNext1D(rrSg) > survival)
                break;

            tree.pathUcw /= max(survival, 1e-6f);
        }

        BSDFSample bs = {};
        if (!mi.sample(surface.sd, pathSg, bs, params.hasFeatureFlag(kEnableBSDFSampling)) ||
            !isValidScatterSample(bs))
            break;

        if (bs.isLobe(LobeType::Transmission))
        {
            if (transmissionBounces >= params.maxTransmissionBounces)
                break;
            transmissionBounces++;
        }
        else if (bs.isLobe(LobeType::Diffuse) ||
            surface.properties.roughness > params.specularRoughnessThreshold)
        {
            if (diffuseBounces >= params.maxDiffuseBounces)
                break;
            diffuseBounces++;
        }
        else
        {
            if (specularBounces >= params.maxSpecularBounces)
                break;
            specularBounces++;
        }

        current.wo = bs.wo;
        current.bsdfPdf = bs.pdf;
        current.fToNext = bs.weight * bs.pdf;
        current.scatterFlags = (bs.lobeType & kWalkLobeMask) |
            (bs.isLobe(LobeType::Delta) ? kWalkDelta : 0u) |
            (hasExactSingleLobe(lobeTypes, bs.lobeType) ? kWalkExactLobePdf : 0u);
        tree.packedLobes = packLobe(tree.packedLobes, bounce, bs.lobeType);

        if (bounce >= 2u)
            cacheInternalReconnection(
                tree.reconnect,
                tree.primaryFootprint,
                previous,
                current,
                bounce,
                params
            );

        updateInteriorAfterScatter(interiors, surface, bs);
        const float3 origin = offsetScatterOrigin(surface, bs);
        float3 nextThp = current.throughput * bs.weight;
        Surface nextSurface;
        bool missed = false;
        if (!traceMediumHit(origin, bs.wo, interiors, rejectedHits, nextThp, params, pixel, false, nextSurface, missed, false))
        {
            if (missed)
            {
#if USE_ENV_LIGHT
                float3 Le = gScene.envMap.eval(bs.wo, 0.f);
                const float mis = bsdfLightMIS(
                    params,
                    (current.scatterFlags & kWalkDelta) != 0u,
                    bs.pdf,
                    evalEnvTypePdf(bs.wo, envMapSampler),
                    1.f
                );
                WalkVertex noEndpoint = {};
                addHitLightCandidate(
                    tree, params, bounce, current, noEndpoint, nextThp, Le, mis, true, kInvalidIndex, 1u, grisSg
                );
#endif
            }
            break;
        }

        const uint nextIndex = bounce + 1u;
        if (nextIndex >= kMaxPathVertices)
            break;

        WalkVertex next = {};
        next.position = nextSurface.position;
        next.normal = nextSurface.sd.getOrientedFaceNormal();
        next.throughput = nextThp;
        next.packedHit = packStoredHit(nextSurface.hit);

        mi = getSurfaceMI(nextSurface, params, pixel);
        applySurfaceMaterial(nextSurface, mi);
        next.roughnessAlpha = nextSurface.roughnessAlpha;

        if (any(nextSurface.properties.emission > 0.f) &&
            (interiors.isEmpty() || params.hasFeatureFlag(kUseLightsInDielectricVolumes)))
        {
            uint lightTriangleIndex = kInvalidIndex;
            float mis = 1.f;
            if (params.hasFeatureFlag(kEnableMIS) && (current.scatterFlags & kWalkDelta) == 0u)
            {
                const float lightPdf = evalHitEmissivePdf(
                    origin,
                    current.normal,
                    nextSurface.hit,
                    nextSurface.position,
                    nextSurface.sd.getOrientedFaceNormal(),
                    params,
                    emissiveSampler,
                    lightTriangleIndex
                );
                mis = evalBalanceMIS(1.f, bs.pdf, float(risM), lightPdf);
            }
            addHitLightCandidate(
                tree,
                params,
                nextIndex,
                current,
                next,
                next.throughput,
                nextSurface.properties.emission,
                mis,
                false,
                lightTriangleIndex,
                risM,
                grisSg
            );
        }

        previous = current;
        current = next;
        surface = nextSurface;
    }

    PathReservoir reservoir = finalizeGRIS(tree.gris);
    reservoir.confidence = 1.f;
    return reservoir;
}
