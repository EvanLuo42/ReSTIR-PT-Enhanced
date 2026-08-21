#include "Scene/SceneDefines.slangh"

import Scene.Scene;
import Rendering.Lights.EmissivePowerSampler;
import Utils.Sampling.SampleGenerator;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;

bool samplePowerTriangle(EmissivePowerSampler powerSampler, inout SampleGenerator sg, out uint triangleIndex, out float selectionPdf, out float2 uv)
{
    triangleIndex = kInvalidIndex;
    selectionPdf = 0.f;
    uv = 0.f;
    if (gScene.lightCollection.isEmpty())
        return false;

    const uint triangleCount = gScene.lightCollection.getTriangleCount();
    if (triangleCount == 0u)
        return false;

    const float uLight = sampleNext1D(sg);
    uint idx = min(uint(uLight * triangleCount), triangleCount - 1u);
    const uint2 packed = powerSampler._emissivePower.triangleAliasTable[idx];
    const float threshold = f16tof32(packed.x >> 16u);
    const uint selectAbove = ((packed.x & 0xFFFFu) << 8u) | ((packed.y >> 24u) & 0xFFu);
    const uint selectBelow = packed.y & 0xFFFFFFu;
    triangleIndex = (sampleNext1D(sg) >= threshold) ? selectAbove : selectBelow;
    if (triangleIndex >= triangleCount)
        return false;

    selectionPdf = gScene.lightCollection.fluxData[triangleIndex].flux * powerSampler._emissivePower.invWeightsSum;
    uv = sampleNext2D(sg);
    return selectionPdf > 0.f;
}

struct PresampleLightsPass
{
    RestirParams params;
#if USE_EMISSIVE_LIGHTS
    EmissivePowerSampler powerSampler;
#endif
    RWStructuredBuffer<LightTileEntry> lightTiles;

    void execute(uint index)
    {
        const uint entryCount = params.lightTileCount * params.lightsPerTile;
        if (index >= entryCount)
            return;

        LightTileEntry entry = {};
        entry.lightIndex = kInvalidIndex;

#if USE_EMISSIVE_LIGHTS
        if (params.hasFeatureFlag(kEnablePresampledLights) && !gScene.lightCollection.isEmpty())
        {
            SampleGenerator sg = SampleGenerator(uint2(index, params.frameIndex), params.frameIndex);
            uint triangleIndex;
            float selectionPdf;
            float2 uv;
            if (samplePowerTriangle(powerSampler, sg, triangleIndex, selectionPdf, uv))
            {
                entry.lightIndex = triangleIndex;
                entry.selectionPdf = selectionPdf;
                entry.sample = uv;
            }
        }
#endif
        lightTiles[index] = entry;
    }
};

ParameterBlock<PresampleLightsPass> gPass;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.x);
}
