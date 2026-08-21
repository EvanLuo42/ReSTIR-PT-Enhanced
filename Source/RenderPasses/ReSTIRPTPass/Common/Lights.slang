#include "Scene/SceneDefines.slangh"

import Scene.Scene;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Rendering.Lights.EmissiveLightSamplerHelpers;
import Rendering.Lights.LightHelpers;
import Rendering.Materials.IMaterialInstance;
import Utils.Sampling.SampleGenerator;
import Utils.Geometry.GeometryHelpers;
import Utils.Math.MathHelpers;
import Utils.Color.ColorHelpers;
import Scene.HitInfo;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.Surface;

struct LightSample
{
    float3 Li;
    float3 Le;
    float3 normal;
    float pdf;
    float3 origin;
    float3 dest;
    float3 dir;
    float distance;
    uint lightType;
    uint triangleIndex;
    uint risM;
    float risUcw;
    bool valid;
    bool hasBsdfF;
    float3 bsdfF;
};

float evalBalanceMIS(float n0, float p0, float n1, float p1)
{
    const float q0 = max(n0, 0.f) * max(p0, 0.f);
    const float q1 = max(n1, 0.f) * max(p1, 0.f);
    const float sum = q0 + q1;
    return sum > 0.f ? q0 / sum : 0.f;
}

float evalBalanceMIS(float p0, float p1)
{
    return evalBalanceMIS(1.f, p0, 1.f, p1);
}

float evalEmissivePdf(
    float3 posW,
    float3 normalW,
    TriangleLightHit hit,
    RestirParams params,
    EmissiveLightSampler emissiveSampler
)
{
#if !USE_EMISSIVE_LIGHTS
    return 0.f;
#else
    if (hit.triangleIndex == kInvalidIndex)
        return 0.f;
    if (params.hasFeatureFlag(kEnablePresampledLights) && params.lightPowerInvWeightsSum > 0.f)
    {
        const float sel = gScene.lightCollection.fluxData[hit.triangleIndex].flux * params.lightPowerInvWeightsSum;
        return sel * evalTrianglePdf(posW, hit);
    }
    return emissiveSampler.evalPdf(posW, normalW, true, hit);
#endif
}

void getLightTypeProbabilities(out float pEnv, out float pEmissive, out float pAnalytic)
{
#if USE_ENV_LIGHT
    pEnv = 1.f;
#else
    pEnv = 0.f;
#endif
#if USE_EMISSIVE_LIGHTS
    pEmissive = 1.f;
#else
    pEmissive = 0.f;
#endif
#if USE_ANALYTIC_LIGHTS
    pAnalytic = 1.f;
#else
    pAnalytic = 0.f;
#endif
    const float sum = pEnv + pEmissive + pAnalytic;
    if (sum > 0.f)
    {
        const float inv = 1.f / sum;
        pEnv *= inv;
        pEmissive *= inv;
        pAnalytic *= inv;
    }
}

float evalEmissiveTypePdf(
    float3 posW,
    float3 normalW,
    TriangleLightHit hit,
    RestirParams params,
    EmissiveLightSampler emissiveSampler
)
{
    float pEnv, pEmissive, pAnalytic;
    getLightTypeProbabilities(pEnv, pEmissive, pAnalytic);
    return pEmissive * evalEmissivePdf(posW, normalW, hit, params, emissiveSampler);
}

float evalEnvTypePdf(float3 dir, EnvMapSampler envMapSampler)
{
#if !USE_ENV_LIGHT
    return 0.f;
#else
    float pEnv, pEmissive, pAnalytic;
    getLightTypeProbabilities(pEnv, pEmissive, pAnalytic);
    return pEnv > 0.f ? pEnv * envMapSampler.evalPdf(dir) : 0.f;
#endif
}

float bsdfLightMIS(RestirParams params, bool deltaLobe, float bsdfPdf, float lightPdf, float lightCount)
{
    if (!params.hasFeatureFlag(kEnableMIS) || deltaLobe)
        return 1.f;
    return evalBalanceMIS(1.f, bsdfPdf, lightCount, lightPdf);
}

bool makeEmissiveTriangleHit(HitInfo hit, float3 posW, float3 fallbackNormal, out TriangleLightHit lightHit)
{
    lightHit = {};
#if !USE_EMISSIVE_LIGHTS
    return false;
#else
    if (hit.getType() != HitType::Triangle)
        return false;
    const TriangleHit triHit = hit.getTriangleHit();
    lightHit.triangleIndex = gScene.lightCollection.getTriangleIndex(triHit.instanceID, triHit.primitiveIndex);
    lightHit.posW = posW;
    lightHit.normalW = fallbackNormal;
    if (lightHit.triangleIndex != kInvalidIndex)
        lightHit.normalW = gScene.lightCollection.getTriangle(lightHit.triangleIndex).normal;
    return lightHit.triangleIndex != kInvalidIndex;
#endif
}

float evalHitEmissivePdf(
    float3 shadingPos,
    float3 shadingNormal,
    HitInfo hit,
    float3 hitPos,
    float3 hitFallbackNormal,
    RestirParams params,
    EmissiveLightSampler emissiveSampler,
    out uint triangleIndex
)
{
    triangleIndex = kInvalidIndex;
    TriangleLightHit lightHit;
    if (!makeEmissiveTriangleHit(hit, hitPos, hitFallbackNormal, lightHit))
        return 0.f;
    triangleIndex = lightHit.triangleIndex;
    return evalEmissiveTypePdf(shadingPos, shadingNormal, lightHit, params, emissiveSampler);
}

float evalHitEmissivePdf(
    float3 shadingPos,
    float3 shadingNormal,
    HitInfo hit,
    float3 hitPos,
    float3 hitFallbackNormal,
    RestirParams params,
    EmissiveLightSampler emissiveSampler
)
{
    uint triangleIndex;
    return evalHitEmissivePdf(
        shadingPos, shadingNormal, hit, hitPos, hitFallbackNormal, params, emissiveSampler, triangleIndex
    );
}

float evalStoredEmissivePdf(
    float3 shadingPos,
    float3 shadingNormal,
    uint triangleIndex,
    float3 lightPos,
    float3 lightNormal,
    RestirParams params,
    EmissiveLightSampler emissiveSampler
)
{
    if (triangleIndex == kInvalidIndex)
        return 0.f;
    TriangleLightHit lightHit;
    lightHit.triangleIndex = triangleIndex;
    lightHit.posW = lightPos;
    lightHit.normalW = lightNormal;
    return evalEmissiveTypePdf(shadingPos, shadingNormal, lightHit, params, emissiveSampler);
}

uint screenLightTile(uint2 pixel, RestirParams params)
{
    const uint tilesX = max(1u, (params.frameDim.x + 7u) / 8u);
    const uint tile = (pixel.y / 8u) * tilesX + (pixel.x / 8u);
    return params.lightTileCount > 0u ? tile % params.lightTileCount : 0u;
}

uint neeRisM(RestirParams params, uint bounceIndex)
{
    if (!params.hasFeatureFlag(kEnablePresampledLights) || params.lightTileCount == 0u || params.lightsPerTile == 0u)
        return 1u;
    return params.neeCandidateCount(bounceIndex);
}

bool reconstructTileLight(
    LightTileEntry entry,
    Surface surface,
    out LightSample ls
)
{
    ls = {};
#if !USE_EMISSIVE_LIGHTS
    return false;
#else
    if (entry.lightIndex == kInvalidIndex)
        return false;
    if (gScene.lightCollection.isEmpty() || entry.lightIndex >= gScene.lightCollection.getTriangleCount())
        return false;

    const EmissiveTriangle tri = gScene.lightCollection.getTriangle(entry.lightIndex);

    const float3 bary = sample_triangle(saturate(entry.sample));
    const float3 posW = tri.getPosition(bary);
    const float3 toLight = posW - surface.position;
    const float dist = length(toLight);
    if (dist < 1e-5f)
        return false;

    const float3 dir = toLight / dist;
    const float3 origin = offsetRayOrigin(surface, dir);
    if (dot(tri.normal, -dir) <= kMinCosTheta)
        return false;

    float3 Le = gScene.materials.evalEmissive(tri.materialID, tri.getTexCoord(bary));
    if (gScene.materials.hasLightProfile(tri.materialID))
        Le *= gScene.materials.evalLightProfile(dot(tri.normal, -dir));

    TriangleLightHit hit;
    hit.triangleIndex = entry.lightIndex;
    hit.posW = posW;
    hit.normalW = tri.normal;

    const float selPdf = entry.selectionPdf > 0.f ? entry.selectionPdf : 0.f;
    const float saPdf = selPdf * evalTrianglePdf(surface.position, hit);
    if (!(saPdf > 0.f) || !isfinite(saPdf) || !any(Le > 0.f))
        return false;

    ls.valid = true;
    ls.Le = Le;
    ls.Li = Le / saPdf;
    ls.pdf = saPdf;
    ls.normal = tri.normal;
    ls.origin = origin;

    ls.dest = posW;
    ls.dir = dir;
    ls.distance = dist;
    ls.lightType = 1u;
    ls.triangleIndex = entry.lightIndex;
    ls.risM = 1u;
    ls.risUcw = 1.f;
    return true;
#endif
}

bool sampleEnvLight(Surface surface, EnvMapSampler envMapSampler, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};
#if !USE_ENV_LIGHT
    return false;
#else
    EnvMapSample envSample;
    if (!envMapSampler.sample(sampleNext2D(sg), envSample))
        return false;
    if (!(envSample.pdf > 0.f) || !any(envSample.Le > 0.f))
        return false;

    ls.valid = true;
    ls.Le = envSample.Le;
    ls.Li = envSample.Le / envSample.pdf;
    ls.pdf = envSample.pdf;
    ls.origin = offsetRayOrigin(surface, envSample.dir);
    ls.dest = ls.origin + envSample.dir;
    ls.dir = envSample.dir;
    ls.distance = FLT_MAX;
    ls.lightType = 0u;
    ls.triangleIndex = kInvalidIndex;
    ls.risM = 1u;
    ls.risUcw = 1.f;
    return true;
#endif
}

bool isAnalyticAreaLight(uint lightIndex)
{
#if !USE_ANALYTIC_LIGHTS
    return false;
#else
    if (lightIndex >= gScene.getLightCount())
        return false;
    switch (gScene.getLight(lightIndex).type)
    {
    case LightType::Rect:
    case LightType::Disc:
    case LightType::Sphere:
        return true;
    default:
        return false;
    }
#endif
}

struct AnalyticNeeEval
{
    float3 dir;
    float3 dest;
    float3 destNormal;
    float3 Li;
    bool infinite;
    bool valid;
};

AnalyticNeeEval evalAnalyticNee(PathSample src, float3 prefixPos)
{
    AnalyticNeeEval e = {};
#if !USE_ANALYTIC_LIGHTS
    return e;
#else
    const uint index = src.lightTriangleIndex;
    const uint lightCount = gScene.getLightCount();
    if (index >= lightCount)
        return e;

    float pEnv, pEmissive, pAnalytic;
    getLightTypeProbabilities(pEnv, pEmissive, pAnalytic);
    if (!(pAnalytic > 0.f))
        return e;

    const LightData light = gScene.getLight(index);
    AnalyticLightSample ls = {};
    switch (light.type)
    {
    case LightType::Point:
        if (!samplePointLight(prefixPos, light, ls))
            return e;
        break;
    case LightType::Directional:
        if (!sampleDirectionalLight(prefixPos, light, ls))
            return e;
        e.infinite = true;
        break;
    case LightType::Distant:
        if (dot(src.incidentDirection, src.incidentDirection) < 1e-12f)
            return e;
        ls.dir = normalize(src.incidentDirection);
        ls.distance = kMaxLightDistance;
        ls.Li = light.intensity;
        e.infinite = true;
        break;
    default:
        ls.posW = src.reconnectionPosition;
        ls.normalW = src.reconnectionNormal;
        if (!finalizeAreaLightSample(prefixPos, light, ls))
            return e;
        break;
    }

    if (!any(ls.Li > 0.f) || !all(isfinite(ls.Li)) || dot(ls.dir, ls.dir) < 1e-12f)
        return e;

    e.valid = true;
    e.dir = ls.dir;
    e.dest = ls.posW;
    e.destNormal = isAnalyticAreaLight(index) ? ls.normalW : float3(0.f);
    e.Li = ls.Li * (float(lightCount) / pAnalytic);
    return e;
#endif
}

bool sampleAnalyticLight(Surface surface, inout SampleGenerator sg, out LightSample ls)
{
    ls = {};
#if !USE_ANALYTIC_LIGHTS
    return false;
#else
    const uint lightCount = gScene.getLightCount();
    if (lightCount == 0u)
        return false;
    const uint lightIndex = min(uint(sampleNext1D(sg) * lightCount), lightCount - 1u);
    const LightData light = gScene.getLight(lightIndex);
    AnalyticLightSample analytic;
    if (!sampleLight(surface.position, light, sg, analytic))
        return false;
    ls.valid = true;
    ls.pdf = analytic.pdf / float(lightCount);
    ls.Li = analytic.Li * float(lightCount);
    ls.Le = ls.pdf > 0.f ? ls.Li * ls.pdf : ls.Li;
    ls.origin = offsetRayOrigin(surface, analytic.dir);
    ls.dir = analytic.dir;
    ls.distance = analytic.distance;
    ls.lightType = 2u;
    ls.triangleIndex = lightIndex;
    ls.risM = 1u;
    ls.risUcw = 1.f;
    if (isAnalyticAreaLight(lightIndex))
    {
        ls.normal = analytic.normalW;
        ls.dest = analytic.posW;
    }
    else if (light.type == LightType::Point)
    {
        ls.dest = analytic.posW;
    }
    else
    {
        ls.dest = ls.origin + analytic.dir;
    }
    return any(ls.Li > 0.f) && all(isfinite(ls.Li));
#endif
}

bool sampleEmissiveDirect(
    Surface surface,
    EmissiveLightSampler emissiveSampler,
    inout SampleGenerator sg,
    out LightSample ls
)
{
    ls = {};
#if !USE_EMISSIVE_LIGHTS
    return false;
#else
    TriangleLightSample tls;
    const float3 n = surface.sd.getOrientedFaceNormal();
    if (!emissiveSampler.sampleLight(surface.position, n, true, sg, tls))
        return false;
    if (!(tls.pdf > 0.f) || !any(tls.Le > 0.f))
        return false;
    ls.valid = true;
    ls.Le = tls.Le;
    ls.Li = tls.Le / tls.pdf;
    ls.pdf = tls.pdf;
    ls.normal = tls.normalW;
    ls.origin = offsetRayOrigin(surface, tls.posW - surface.position);
    ls.dest = tls.posW;
    ls.distance = tls.distance;
    ls.dir = tls.dir;
    ls.lightType = 1u;
    ls.triangleIndex = tls.triangleIndex;
    ls.risM = 1u;
    ls.risUcw = 1.f;
    return true;
#endif
}

bool sampleEmissiveFromTiles(
    Surface surface,
    const IMaterialInstance mi,
    EmissiveLightSampler emissiveSampler,
    StructuredBuffer<LightTileEntry> lightTiles,
    uint2 pixel,
    uint bounceIndex,
    inout SampleGenerator sg,
    RestirParams params,
    out LightSample ls
)
{
    ls = {};
#if !USE_EMISSIVE_LIGHTS
    return false;
#else
    if (!params.hasFeatureFlag(kEnablePresampledLights) || params.lightTileCount == 0u || params.lightsPerTile == 0u)
        return sampleEmissiveDirect(surface, emissiveSampler, sg, ls);

    const uint n = params.neeCandidateCount(bounceIndex);
    const uint tile = screenLightTile(pixel, params);
    const uint base = tile * params.lightsPerTile;

    float weightSum = 0.f;
    LightSample selected = {};
    float selectedPhat = 0.f;
    for (uint i = 0u; i < n; ++i)
    {
        const uint slot = min(uint(sampleNext1D(sg) * params.lightsPerTile), params.lightsPerTile - 1u);
        const LightTileEntry entry = lightTiles[base + slot];
        LightSample candidate;
        if (!reconstructTileLight(entry, surface, candidate))
            continue;

        const float3 f = mi.eval(surface.sd, candidate.dir, sg);

        const float w = luminance(f * candidate.Li);
        if (!(w > 0.f) || !isfinite(w))
            continue;

        weightSum += w;
        if (sampleNext1D(sg) * weightSum < w)
        {
            selected = candidate;
            selected.bsdfF = f;
            selected.hasBsdfF = true;
            selectedPhat = w;
        }
    }

    if (!selected.valid || !(weightSum > 0.f) || selectedPhat <= 0.f)
        return false;

    selected.risUcw = weightSum / (float(n) * selectedPhat);
    selected.risM = n;
    ls = selected;
    return true;
#endif
}

bool sampleUnifiedLight(
    Surface surface,
    const IMaterialInstance mi,
    EnvMapSampler envMapSampler,
    EmissiveLightSampler emissiveSampler,
    StructuredBuffer<LightTileEntry> lightTiles,
    uint2 pixel,
    uint bounceIndex,
    inout SampleGenerator sg,
    RestirParams params,
    out LightSample ls
)
{
    ls = {};
    float pEnv, pEmissive, pAnalytic;
    getLightTypeProbabilities(pEnv, pEmissive, pAnalytic);
    const float sum = pEnv + pEmissive + pAnalytic;
    if (sum <= 0.f)
        return false;

    float u = sampleNext1D(sg);
    uint type = 0u;
    float typePdf = 1.f;
    if (u < pEnv)
    {
        type = 0u;
        typePdf = pEnv;
    }
    else
    {
        u -= pEnv;
        if (u < pEmissive)
        {
            type = 1u;
            typePdf = pEmissive;
        }
        else
        {
            type = 2u;
            typePdf = pAnalytic;
        }
    }

    bool valid = false;
    if (type == 0u)
        valid = sampleEnvLight(surface, envMapSampler, sg, ls);
    else if (type == 1u)
        valid = sampleEmissiveFromTiles(surface, mi, emissiveSampler, lightTiles, pixel, bounceIndex, sg, params, ls);
    else
        valid = sampleAnalyticLight(surface, sg, ls);

    if (!valid || typePdf <= 0.f)
        return false;

    ls.pdf *= typePdf;
    ls.Li /= typePdf;

    return any(ls.Li > 0.f) && (ls.pdf > 0.f || ls.lightType == 2u);
}
