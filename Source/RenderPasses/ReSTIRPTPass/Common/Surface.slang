#include "Scene/SceneDefines.slangh"

import Scene.Scene;
import RenderPasses.ReSTIRPTPass.Params;
__exported import RenderPasses.ReSTIRPTPass.Common.SurfaceTrace;

bool isIdentitySurfaceMapping(float3 posA, float3 nA, float3 woA, uint matA, float3 posB, float3 nB, float3 woB, uint matB)
{
    if (matA != matB)
        return false;
    const float positionScale = max(1.f, max(length(posA), length(posB)));
    return length(posA - posB) <= 1e-6f * positionScale &&
        dot(normalize(nA), normalize(nB)) >= 1.f - 1e-6f &&
        dot(normalize(woA), normalize(woB)) >= 1.f - 1e-6f;
}

bool isIdentitySurfaceMapping(SurfaceGeometry current, SurfaceGeometry history)
{
    if (!current.valid || !history.valid)
        return false;
    return isIdentitySurfaceMapping(
        current.position, current.faceNormal, current.viewDir, current.materialID,
        history.position, history.faceNormal, history.viewDir, history.materialID
    );
}

bool normalsAndDepthCompatible(
    float3 nA,
    float3 pA,
    float3 cameraA,
    float3 nB,
    float3 pB,
    float3 cameraB,
    float normalThreshold,
    float depthThreshold
)
{
    const float nDot = dot(normalize(nA), normalize(nB));
    if (nDot < normalThreshold)
        return false;
    const float da = length(pA - cameraA);
    const float db = length(pB - cameraB);
    const float denom = max(min(da, db), 1e-3f);
    return abs(da - db) <= depthThreshold * denom;
}

bool surfacesCompatible(SurfaceGeometry a, SurfaceGeometry b, RestirParams params)
{
    if (!a.valid || !b.valid)
        return false;
    const float3 camera = gScene.camera.getPosition();
    return normalsAndDepthCompatible(
        a.faceNormal, a.position, camera, b.faceNormal, b.position, camera,
        params.spatialNormalThreshold, params.spatialDepthThreshold
    );
}

bool temporalSurfacesCompatible(SurfaceGeometry current, SurfaceGeometry history, RestirParams params)
{
    if (!current.valid || !history.valid)
        return false;
    return normalsAndDepthCompatible(
        current.faceNormal, current.position, gScene.camera.getPosition(),
        history.faceNormal, history.position, gScene.camera.data.prevPosW,
        params.spatialNormalThreshold, params.temporalDepthThreshold
    );
}
