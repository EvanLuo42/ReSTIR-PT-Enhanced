#include "Scene/SceneDefines.slangh"

import Scene.Scene;
import Scene.Shading;
import Scene.Material.TextureSampler;
import Rendering.Materials.IMaterial;
import Rendering.Materials.IMaterialInstance;
import Scene.Material.ShadingUtils;
import Rendering.Materials.TexLODHelpers;
import Rendering.Materials.TexLODTypes;
import Rendering.Materials.MaterialInstanceHints;
__exported import Rendering.Materials.InteriorListHelpers;
import Rendering.Volumes.HomogeneousVolumeSampler;
import Utils.Geometry.GeometryHelpers;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
__exported import RenderPasses.ReSTIRPTPass.Common.SurfaceTypes;

uint getMaterialHints(const HitInfo hit, bool isPrimary, RestirParams params)
{
    uint hints = 0u;
    if (!isPrimary &&
        (hit.getType() == HitType::Triangle || hit.getType() == HitType::DisplacedTriangle) &&
        params.hasFeatureFlag(kAdjustShadingNormals))
    {
        hints |= (uint)MaterialInstanceHints::AdjustShadingNormal;
    }
    return hints;
}

void computeDerivativesAtPrimaryTriangleHit(const TriangleHit hit, uint2 pixel, uint2 frameDim, out float2 ddx, out float2 ddy)
{
    VertexData v = gScene.getVertexData(hit);
    float hitT = length(v.posW - gScene.camera.getPosition());
    const float3 nonNormalizedRayDir = gScene.camera.computeNonNormalizedRayDirPinhole(pixel, frameDim);
    float3 rayDir = normalize(nonNormalizedRayDir);

    float3 dDdx, dDdy;
    computeRayDirectionDifferentials(nonNormalizedRayDir, gScene.camera.data.cameraU, gScene.camera.data.cameraV, frameDim, dDdx, dDdy);

    RayDiff rd = RayDiff(float3(0.f), float3(0.f), dDdx, dDdy);
    rd = rd.propagate(v.posW.xyz, rayDir, hitT, v.faceNormalW);

    const float4x4 worldMat = gScene.getWorldMatrix(hit.instanceID);
    const float3x3 worldInvTransposeMat = gScene.getInverseTransposeWorldMatrix(hit.instanceID);
    const uint3 vertexIndices = gScene.getIndices(hit.instanceID, hit.primitiveIndex);
    StaticVertexData vertices[3] = {gScene.getVertex(vertexIndices[0]), gScene.getVertex(vertexIndices[1]), gScene.getVertex(vertexIndices[2])};
    float2 dBarydx, dBarydy;
    float3 unnormalizedN, normals[3];
    prepareRayDiffAtHitPoint(v, vertices, hit.getBarycentricWeights(), rayDir, worldMat, worldInvTransposeMat, rd, unnormalizedN, normals, dBarydx, dBarydy, ddx, ddy);
}

ITextureSampler createTextureSampler(const HitInfo hit, bool isPrimary, RestirParams params, uint2 pixel)
{
    const bool isTriangleHit = hit.getType() == HitType::Triangle;
    if (TexLODMode(params.primaryLodMode) == TexLODMode::RayDiffs && isPrimary && isTriangleHit)
    {
        float2 ddx, ddy;
        computeDerivativesAtPrimaryTriangleHit(hit.getTriangleHit(), pixel, params.frameDim, ddx, ddy);
        return ExplicitGradientTextureSampler(ddx, ddy);
    }
    const float lod = isPrimary ? 0.f : params.lodBias;
    return ExplicitLodTextureSampler(lod);
}

IMaterialInstance getSurfaceMI(Surface surface, RestirParams params, uint2 pixel)
{
    let lod = createTextureSampler(surface.hit, surface.isPrimary, params, pixel);
    return gScene.materials.getMaterialInstance(surface.sd, lod, getMaterialHints(surface.hit, surface.isPrimary, params));
}

InteriorList makeEmptyInteriorList()
{
    InteriorList interiors = {};
#if INTERIOR_LIST_SLOT_COUNT >= 1
    interiors.slots[0] = 0u;
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 2
    interiors.slots[1] = 0u;
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 3
    interiors.slots[2] = 0u;
#endif
#if INTERIOR_LIST_SLOT_COUNT >= 4
    interiors.slots[3] = 0u;
#endif
    return interiors;
}

float3 evalVolumeTransmittance(InteriorList interiors, float distance)
{
    if (interiors.isEmpty() || !(distance > 0.f) || !isfinite(distance))
        return float3(1.f, 1.f, 1.f);
    const uint materialID = interiors.getTopMaterialID();
    let material = gScene.materials.getMaterial(materialID);
    let vp = material.getHomogeneousVolumeProperties(gScene.materials, materialID);
    return HomogeneousVolumeSampler::evalTransmittance(vp, distance);
}

void applySurfaceMaterial(inout Surface surface, IMaterialInstance mi)
{
    surface.properties = mi.getProperties(surface.sd);
    surface.roughnessAlpha = max(surface.properties.roughness * surface.properties.roughness, 0.f);
    surface.valid = true;
}

void applySurfaceMaterial(inout Surface surface, RestirParams params, uint2 pixel)
{
    applySurfaceMaterial(surface, getSurfaceMI(surface, params, pixel));
}

void finalizeSurface(inout Surface surface, RestirParams params, uint2 pixel, bool applyProperties)
{
    if (applyProperties)
        applySurfaceMaterial(surface, params, pixel);
    else
        surface.valid = true;
}

bool loadCurrentVertexData(
    const HitInfo hit,
    const float3 rayOrigin,
    const float3 rayDir,
    out VertexData v,
    out uint materialID
)
{
    materialID = 0u;
    v = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        StaticVertexData vertices[3];
        v = gScene.getVertexData(triangleHit, vertices);
        materialID = gScene.getMaterialID(triangleHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        v = gScene.getVertexDataFromCurve(curveHit);
        materialID = gScene.getMaterialID(curveHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_SDF_GRID)
    if (hit.getType() == HitType::SDFGrid)
    {
        const SDFGridHit sdfGridHit = hit.getSDFGridHit();
        v = gScene.getVertexDataFromSDFGrid(sdfGridHit, rayOrigin, rayDir);
        materialID = gScene.getMaterialID(sdfGridHit.instanceID);
        return true;
    }
#endif
    return false;
}

void transformVertexFrameToPrevious(const GeometryInstanceID instanceID, inout VertexData v)
{
    const float3x3 currentWorld = (float3x3)gScene.getWorldMatrix(instanceID);
    const float3x3 currentWorldInv = (float3x3)gScene.getInstanceFromWorld(instanceID);
    const float3x3 previousWorld = (float3x3)gScene.getPrevWorldMatrix(instanceID);
    const float3x3 previousInvTranspose = gScene.getPrevInverseTransposeWorldMatrix(instanceID);

    const float3 objectNormal = mul(transpose(currentWorld), v.normalW);
    const float3 objectFaceNormal = mul(transpose(currentWorld), v.faceNormalW);
    const float3 objectTangent = mul(currentWorldInv, v.tangentW.xyz);
    v.normalW = normalize(mul(previousInvTranspose, objectNormal));
    v.faceNormalW = normalize(mul(previousInvTranspose, objectFaceNormal));
    v.tangentW.xyz = normalize(mul(previousWorld, objectTangent));
}

bool getPreviousHitPosition(const HitInfo hit, out float3 position)
{
    position = 0.f;
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        position = gScene.getPrevPosW(hit.getTriangleHit());
        return all(isfinite(position));
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        position = gScene.getPrevPosW(hit.getDisplacedTriangleHit());
        return all(isfinite(position));
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        position = gScene.getPrevPosWFromCurve(hit.getCurveHit());
        return all(isfinite(position));
    }
#endif

    return false;
}

bool loadPreviousVertexData(
    const HitInfo hit,
    const float3 rayDir,
    out VertexData v,
    out uint materialID
)
{
    materialID = 0u;
    v = {};

#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_TRIANGLE_MESH)
    if (hit.getType() == HitType::Triangle)
    {
        const TriangleHit triangleHit = hit.getTriangleHit();
        const GeometryInstanceData instance = gScene.getGeometryInstance(triangleHit.instanceID);
        const bool use16Bit = (instance.flags & uint(GeometryInstanceFlags::Use16BitIndices)) != 0u;
        const uint3 localIndices = gScene.getLocalIndices(instance.ibOffset, triangleHit.primitiveIndex, use16Bit);
        const uint3 currentIndices = localIndices + instance.vbOffset;
        StaticVertexData vertices[3] = {
            gScene.getVertex(currentIndices[0]),
            gScene.getVertex(currentIndices[1]),
            gScene.getVertex(currentIndices[2])
        };

        if (instance.isDynamic())
        {
            const uint previousOffset = gScene.meshes[instance.geometryID].prevVbOffset;
            [unroll]
            for (uint i = 0u; i < 3u; ++i)
                vertices[i].position = gScene.prevVertices[localIndices[i] + previousOffset].position;
        }

        v = gScene.computeVertexData(
            vertices,
            gScene.getPrevWorldMatrix(triangleHit.instanceID),
            gScene.getPrevInverseTransposeWorldMatrix(triangleHit.instanceID),
            gScene.isObjectFrontFaceCW(triangleHit.instanceID),
            triangleHit.getBarycentricWeights()
        );
        materialID = gScene.getMaterialID(triangleHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_DISPLACED_TRIANGLE_MESH)
    if (hit.getType() == HitType::DisplacedTriangle)
    {
        const DisplacedTriangleHit displacedTriangleHit = hit.getDisplacedTriangleHit();
        v = gScene.getVertexData(displacedTriangleHit, -rayDir);
        transformVertexFrameToPrevious(displacedTriangleHit.instanceID, v);
        v.posW = gScene.getPrevPosW(displacedTriangleHit);
        materialID = gScene.getMaterialID(displacedTriangleHit.instanceID);
        return true;
    }
#endif
#if SCENE_HAS_GEOMETRY_TYPE(GEOMETRY_TYPE_CURVE)
    if (hit.getType() == HitType::Curve)
    {
        const CurveHit curveHit = hit.getCurveHit();
        const uint v0 = gScene.getFirstCurveVertexIndex(curveHit.instanceID, curveHit.primitiveIndex);
        const StaticCurveVertexData vertices[2] = {gScene.getCurveVertex(v0), gScene.getCurveVertex(v0 + 1u)};
        const float3 positions[2] = {
            gScene.prevCurveVertices[v0].position,
            gScene.prevCurveVertices[v0 + 1u].position
        };
        const float4x4 previousWorld = gScene.getPrevWorldMatrix(curveHit.instanceID);
        const float4 sphereA = transformSphere(float4(positions[0], vertices[0].radius), previousWorld);
        const float4 sphereB = transformSphere(float4(positions[1], vertices[1].radius), previousWorld);
        float3 sphereCenter;
        uvToCurveHitPos(curveHit.barycentrics, sphereA, sphereB, v.posW, sphereCenter);
        v.normalW = v.posW - sphereCenter;
        v.curveRadius = length(v.normalW);
        if (!(v.curveRadius > 0.f))
            return false;
        v.normalW /= v.curveRadius;
        v.faceNormalW = v.normalW;
        v.tangentW = float4(normalize(sphereB.xyz - sphereA.xyz), 1.f);
        v.texC = lerp(vertices[0].texCrd, vertices[1].texCrd, curveHit.barycentrics.x);
        materialID = gScene.getMaterialID(curveHit.instanceID);
        return true;
    }
#endif
    return false;
}

bool loadHitShading(
    const HitInfo hit,
    const float3 rayOrigin,
    const float3 rayDir,
    out ShadingData sd
)
{
    sd = {};
    VertexData v;
    uint materialID;
    if (!loadCurrentVertexData(hit, rayOrigin, rayDir, v, materialID))
        return false;
    sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    return true;
}

bool loadPreviousShading(const HitInfo hit, const float3 rayDir, out ShadingData sd)
{
    sd = {};
    VertexData v;
    uint materialID;
    if (!loadPreviousVertexData(hit, rayDir, v, materialID))
        return false;
    sd = gScene.materials.prepareShadingData(v, materialID, -rayDir);
    return true;
}

bool loadSurfaceShading(const HitInfo hit, const float3 rayOrigin, const float3 rayDir, bool isPrimary, out Surface surface)
{
    surface = {};
    surface.hit = hit;
    if (!hit.isValid())
        return false;

    if (!loadHitShading(hit, rayOrigin, rayDir, surface.sd))
        return false;
    surface.position = surface.sd.posW;
    surface.faceNormal = surface.sd.faceN;
    surface.viewDir = -rayDir;
    surface.hitT = length(surface.position - rayOrigin);
    surface.isPrimary = isPrimary;
    return true;
}

bool loadSurface(
    const HitInfo hit,
    const float3 rayOrigin,
    const float3 rayDir,
    bool isPrimary,
    RestirParams params,
    uint2 pixel,
    InteriorList interiors,
    out Surface surface,
    bool applyProperties = true
)
{
    if (!loadSurfaceShading(hit, rayOrigin, rayDir, isPrimary, surface))
        return false;
    surface.sd.IoR = computeOutsideIoR(interiors, surface.sd.materialID, surface.sd.frontFacing);
    finalizeSurface(surface, params, pixel, applyProperties);
    return true;
}

bool loadPreviousSurface(
    const HitInfo hit,
    bool isPrimary,
    RestirParams params,
    uint2 pixel,
    out Surface surface,
    bool applyProperties = true
)
{
    surface = {};
    surface.hit = hit;
    if (!hit.isValid())
        return false;

    float3 previousPosition;
    if (!getPreviousHitPosition(hit, previousPosition))
        return false;
    const float3 rayOrigin = gScene.camera.data.prevPosW;
    const float3 toSurface = previousPosition - rayOrigin;
    const float hitT = length(toSurface);
    if (!(hitT > 0.f) || !isfinite(hitT))
        return false;
    const float3 rayDir = toSurface / hitT;

    if (!loadPreviousShading(hit, rayDir, surface.sd))
        return false;
    surface.position = surface.sd.posW;
    surface.faceNormal = surface.sd.faceN;
    surface.viewDir = -rayDir;
    surface.hitT = hitT;
    surface.isPrimary = isPrimary;
    surface.sd.IoR = 1.f;
    finalizeSurface(surface, params, pixel, applyProperties);
    return true;
}

bool loadSurfaceFromVBuffer(
    Texture2D<PackedHitInfo> vbuffer,
    uint2 pixel,
    RestirParams params,
    out Surface surface,
    bool applyProperties = true
)
{
    const HitInfo hit = unpackHitInfo(vbuffer[pixel]);
    const Ray cameraRay = gScene.camera.computeRayPinhole(pixel, params.frameDim);
    return loadSurface(hit, cameraRay.origin, cameraRay.dir, true, params, pixel, makeEmptyInteriorList(), surface, applyProperties);
}

bool loadSurfaceGeometryFromVBuffer(
    Texture2D<PackedHitInfo> vbuffer,
    uint2 pixel,
    uint2 frameDim,
    out SurfaceGeometry geometry
)
{
    geometry = {};
    const HitInfo hit = unpackHitInfo(vbuffer[pixel]);
    if (!hit.isValid())
        return false;
    const Ray cameraRay = gScene.camera.computeRayPinhole(pixel, frameDim);
    VertexData vertex;
    uint materialID;
    if (!loadCurrentVertexData(hit, cameraRay.origin, cameraRay.dir, vertex, materialID))
        return false;
    geometry.position = vertex.posW;
    geometry.faceNormal = vertex.faceNormalW;
    geometry.viewDir = -cameraRay.dir;
    geometry.materialID = materialID;
    geometry.valid = all(isfinite(geometry.position)) && all(isfinite(geometry.faceNormal));
    return geometry.valid;
}

bool loadPreviousSurfaceGeometryFromVBuffer(
    Texture2D<PackedHitInfo> previousVBuffer,
    uint2 pixel,
    out SurfaceGeometry geometry
)
{
    geometry = {};
    const HitInfo hit = unpackHitInfo(previousVBuffer[pixel]);
    if (!hit.isValid())
        return false;

    float3 previousPosition;
    if (!getPreviousHitPosition(hit, previousPosition))
        return false;
    const float3 rayOrigin = gScene.camera.data.prevPosW;
    const float3 toSurface = previousPosition - rayOrigin;
    const float hitT = length(toSurface);
    if (!(hitT > 0.f) || !isfinite(hitT))
        return false;
    const float3 rayDir = toSurface / hitT;

    VertexData vertex;
    uint materialID;
    if (!loadPreviousVertexData(hit, rayDir, vertex, materialID))
        return false;
    geometry.position = vertex.posW;
    geometry.faceNormal = vertex.faceNormalW;
    geometry.viewDir = -rayDir;
    geometry.materialID = materialID;
    geometry.valid = all(isfinite(geometry.position)) && all(isfinite(geometry.faceNormal));
    return geometry.valid;
}

bool loadPreviousSurfaceFromVBuffer(
    Texture2D<PackedHitInfo> previousVBuffer,
    uint2 pixel,
    RestirParams params,
    out Surface surface,
    bool applyProperties = true
)
{
    const HitInfo hit = unpackHitInfo(previousVBuffer[pixel]);
    return loadPreviousSurface(hit, true, params, pixel, surface, applyProperties);
}

bool resolveReconnectEndpoint(
    uint4 packedHit,
    uint lightTriangleIndex,
    float3 storedPos,
    float3 storedNormal,
    GeometryTime srcTime,
    GeometryTime destTime,
    out float3 pos,
    out float3 normal
)
{
    pos = storedPos;
    normal = storedNormal;
    if (srcTime == destTime)
        return all(isfinite(pos));
    if (packedHitIsValid(packedHit))
    {
        const HitInfo hit = unpackStoredHit(packedHit);
        VertexData v;
        uint materialID;
        bool ok = false;
        if (destTime == GeometryTime::Previous)
            ok = loadPreviousVertexData(hit, storedPos - storedNormal, v, materialID);
        else
            ok = loadCurrentVertexData(hit, storedPos - storedNormal, storedNormal, v, materialID);
        if (!ok || !all(isfinite(v.posW)))
            return false;
        pos = v.posW;
        normal = v.normalW;
        return true;
    }

#if USE_EMISSIVE_LIGHTS
    if (lightTriangleIndex != kInvalidIndex && lightTriangleIndex < gScene.lightCollection.getTriangleCount())
    {
        const EmissiveTriangle tri = gScene.lightCollection.getTriangle(lightTriangleIndex);
        const MeshLightData mesh = gScene.lightCollection.meshData[tri.lightIdx];
        GeometryInstanceID instanceID;
        instanceID.index = mesh.instanceID;
        pos = mapWorldPointBetweenFrames(instanceID, storedPos, srcTime, destTime);
        if (destTime == GeometryTime::Current)
            normal = tri.normal;
        else
            normal = mapWorldNormalBetweenFrames(instanceID, storedNormal, srcTime, destTime);
        return all(isfinite(pos));
    }
#endif
    if (srcTime != destTime)
        return false;
    return all(isfinite(pos));
}

