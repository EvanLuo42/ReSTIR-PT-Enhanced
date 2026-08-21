#include "Scene/SceneDefines.slangh"

import Scene.RaytracingInline;
import Rendering.Lights.EnvMapSampler;
import Rendering.Lights.EmissiveLightSampler;
import Utils.Sampling.SampleGenerator;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Path.PathSampling;

struct InitialSamplePass
{
    RestirParams params;

    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;

    Texture2D<PackedHitInfo> vbuffer;
    StructuredBuffer<LightTileEntry> lightTiles;

    RWStructuredBuffer<PathReservoir> currentReservoirs;
    RWTexture2D<float4> outputColor;

    void execute(uint2 pixel)
    {
        if (!params.isPixelInBounds(pixel))
            return;

        const uint index = params.pixelIndex(pixel);
        float3 emission;
        const PathReservoir reservoir = samplePathTree(
            params,
            envMapSampler,
            emissiveSampler,
            lightTiles,
            vbuffer,
            pixel,
            emission
        );

        currentReservoirs[index] = reservoir;
        outputColor[pixel] = float4(emission, 0.f);
    }
};

ParameterBlock<InitialSamplePass> gPass;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.xy);
}
