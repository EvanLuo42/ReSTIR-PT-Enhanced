#include "Scene/SceneDefines.slangh"
#include "Utils/Math/MathConstants.slangh"

import Scene.RaytracingInline;
import Rendering.Materials.IMaterialInstance;
import Rendering.Materials.LobeType;
import Utils.Geometry.GeometryHelpers;
import RenderPasses.ReSTIRPTPass.Params;
__exported import RenderPasses.ReSTIRPTPass.Common.SurfaceLoad;

static const uint kMaxRejectedHits = 16u;

float3 offsetRayOrigin(Surface surface, float3 dir)
{
    const float3 n = surface.faceNormal;
    if (dot(n, n) < 1e-8f)
        return surface.position;
    return computeRayOrigin(surface.position, dot(n, dir) >= 0.f ? n : -n);
}

float3 offsetScatterOrigin(Surface surface, BSDFSample bs)
{
    if (bs.isLobe(LobeType::Transmission) && !surface.sd.mtl.isThinSurface())
        return surface.sd.computeRayOrigin(false);
    return offsetRayOrigin(surface, bs.wo);
}

void updateInteriorAfterScatter(inout InteriorList interiors, Surface surface, BSDFSample bs)
{
    if (bs.isLobe(LobeType::Transmission) && !surface.sd.mtl.isThinSurface())
        interiors.handleIntersection(surface.sd.materialID, surface.sd.mtl.getNestedPriority(), surface.sd.frontFacing);
}

bool isValidScatterSample(BSDFSample bs)
{
    if (!any(bs.weight > 0.f) || !all(isfinite(bs.weight)) || !all(isfinite(bs.wo)))
        return false;
    if (bs.isLobe(LobeType::Delta))
        return true;
    return bs.pdf > 0.f && isfinite(bs.pdf);
}

bool traceVisibility(float3 origin, float3 dest)
{
    const float3 delta = dest - origin;
    const float dist = length(delta);
    if (dist < 1e-5f)
        return false;
    const float3 dir = delta / dist;
    SceneRayQuery<1> query;
    Ray ray;
    ray.origin = origin;
    ray.dir = dir;
    ray.tMin = 0.f;

    ray.tMax = dist;
    return query.traceVisibilityRay(ray);
}

bool traceDirectionalVisibility(float3 origin, float3 dir, float tMax = FLT_MAX)
{
    if (!all(isfinite(origin)) || !all(isfinite(dir)) || dot(dir, dir) < 1e-12f)
        return false;
    SceneRayQuery<1> query;
    Ray ray;
    ray.origin = origin;
    ray.dir = normalize(dir);
    ray.tMin = 0.f;
    ray.tMax = tMax;
    return query.traceVisibilityRay(ray);
}

bool traceReconnectVisibility(Surface prefix, float3 dest, float3 destNormal)
{
    const float3 toDest = dest - prefix.position;
    const float dist = length(toDest);
    if (dist < 1e-5f)
        return false;
    const float3 dir = toDest / dist;
    float3 originN = prefix.faceNormal;
    if (dot(originN, originN) < 1e-8f)
        originN = prefix.sd.getOrientedFaceNormal();
    if (dot(originN, dir) < 0.f)
        originN = -originN;
    const float3 origin = computeRayOrigin(prefix.position, originN);

    float3 destPos = dest;
    if (any(destNormal != 0.f))
    {
        float3 n = destNormal;
        if (dot(n, -dir) < 0.f)
            n = -n;
        destPos = computeRayOrigin(dest, n);
    }
    return traceVisibility(origin, destPos);
}

bool tracePathRay(float3 origin, float3 dir, float tMin, float tMax, out HitInfo hit, out float hitT)
{
    SceneRayQuery<1> query;
    Ray ray;
    ray.origin = origin;
    ray.dir = dir;
    ray.tMin = tMin;

    ray.tMax = tMax;
    hit = query.traceRay(ray, hitT);
    return hit.isValid();
}

bool traceMediumHit(
    float3 origin,
    float3 dir,
    inout InteriorList interiors,
    inout uint rejectedHits,
    inout float3 throughput,
    RestirParams params,
    uint2 pixel,
    bool isPrimary,
    out Surface surface,
    out bool missed,
    bool applyProperties = true
)
{
    surface = {};
    missed = false;
    for (uint i = 0u; i <= kMaxRejectedHits; ++i)
    {
        HitInfo hit;
        float hitT;
        if (!tracePathRay(origin, dir, 0.f, FLT_MAX, hit, hitT))
        {
            missed = true;
            return false;
        }
        if (!loadSurfaceShading(hit, origin, dir, isPrimary, surface))
            return false;

        throughput *= evalVolumeTransmittance(interiors, surface.hitT);

        const uint nestedPriority = surface.sd.mtl.getNestedPriority();
        if (!interiors.isTrueIntersection(nestedPriority))
        {
            if (rejectedHits >= kMaxRejectedHits)
                return false;
            rejectedHits++;
            interiors.handleIntersection(surface.sd.materialID, nestedPriority, surface.sd.frontFacing);
            origin = surface.sd.computeRayOrigin(false);
            continue;
        }

        surface.sd.IoR = computeOutsideIoR(interiors, surface.sd.materialID, surface.sd.frontFacing);
        finalizeSurface(surface, params, pixel, applyProperties);
        return surface.valid;
    }
    return false;
}
