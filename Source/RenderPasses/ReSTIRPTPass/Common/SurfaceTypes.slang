#include "Scene/SceneDefines.slangh"

import Scene.Scene;
__exported import Scene.HitInfo;
import Scene.Shading;
import Rendering.Materials.IMaterial;
import Utils.Math.MathHelpers;
import RenderPasses.ReSTIRPTPass.Common.Types;

struct Surface
{
    HitInfo hit;
    ShadingData sd;
    BSDFProperties properties;
    float3 position;
    float3 faceNormal;
    float3 viewDir;
    float hitT;
    float roughnessAlpha;
    bool valid;
    bool isPrimary;
};

struct SurfaceGeometry
{
    float3 position;
    float3 faceNormal;
    float3 viewDir;
    uint materialID;
    bool valid;
};

enum class GeometryTime : uint
{
    Current = 0,
    Previous = 1,
};

HitInfo unpackStoredHit(uint4 packed)
{
#if HIT_INFO_USE_COMPRESSION
    PackedHitInfo hitPacked = packed.xy;
#else
    PackedHitInfo hitPacked = packed;
#endif
    return unpackHitInfo(hitPacked);
}

uint4 packStoredHit(HitInfo hit)
{
    PackedHitInfo packed = hit.pack();
#if HIT_INFO_USE_COMPRESSION
    return uint4(packed, 0u, 0u);
#else
    return packed;
#endif
}

bool packedHitIsValid(uint4 packedHit)
{
    return packedHit.x != 0u;
}

float3 mapWorldPointBetweenFrames(
    GeometryInstanceID instanceID,
    float3 pos,
    GeometryTime fromTime,
    GeometryTime toTime
)
{
    if (fromTime == toTime)
        return pos;
    const float4x4 prevW = gScene.getPrevWorldMatrix(instanceID);
    const float4x4 currW = gScene.getWorldMatrix(instanceID);
    if (fromTime == GeometryTime::Previous)
        return mul(currW, mul(inverse(prevW), float4(pos, 1.f))).xyz;
    return mul(prevW, mul(gScene.getInstanceFromWorld(instanceID), float4(pos, 1.f))).xyz;
}

float3 mapWorldDirBetweenFrames(
    GeometryInstanceID instanceID,
    float3 dir,
    GeometryTime fromTime,
    GeometryTime toTime
)
{
    if (fromTime == toTime || dot(dir, dir) < 1e-12f)
        return dir;
    const float3x3 prevR = (float3x3)gScene.getPrevWorldMatrix(instanceID);
    const float3x3 currR = (float3x3)gScene.getWorldMatrix(instanceID);
    if (fromTime == GeometryTime::Previous)
        return normalize(mul(currR, mul((float3x3)inverse(gScene.getPrevWorldMatrix(instanceID)), dir)));
    return normalize(mul(prevR, mul((float3x3)gScene.getInstanceFromWorld(instanceID), dir)));
}

float3 mapWorldNormalBetweenFrames(
    GeometryInstanceID instanceID,
    float3 normal,
    GeometryTime fromTime,
    GeometryTime toTime
)
{
    if (fromTime == toTime || dot(normal, normal) < 1e-12f)
        return normal;
    if (fromTime == GeometryTime::Previous)
    {
        const float3 objectN = mul(transpose(gScene.getPrevInverseTransposeWorldMatrix(instanceID)), normal);
        return normalize(mul(gScene.getInverseTransposeWorldMatrix(instanceID), objectN));
    }
    const float3 objectN = mul(transpose(gScene.getInverseTransposeWorldMatrix(instanceID)), normal);
    return normalize(mul(gScene.getPrevInverseTransposeWorldMatrix(instanceID), objectN));
}
