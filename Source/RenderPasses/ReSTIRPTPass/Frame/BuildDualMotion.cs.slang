#include "Scene/SceneDefines.slangh"

import Scene.Scene;
import Utils.Math.MathHelpers;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;

struct BuildDualMotionPass
{
    RestirParams params;

    Texture2D<float2> primaryMotion;
    Texture2D<float> previousDepth;

    RWTexture2D<float2> dualMotion;
    RWTexture2D<uint> motionValidity;

    bool unprojectPrevious(uint2 pixel, out float3 worldPos)
    {
        worldPos = 0.f;
        const float depth = previousDepth[pixel];
        if (!isfinite(depth))
            return false;

        const float2 uv = (float2(pixel) + 0.5f) / float2(params.frameDim);
#ifdef FALCOR_FLIP_Y
        const float2 ndc = float2(2.f, 2.f) * uv - 1.f;
#else
        const float2 ndc = float2(2.f, -2.f) * uv + float2(-1.f, 1.f);
#endif
        const float4 clip = float4(ndc, depth, 1.f);
        const float4 worldH = mul(params.previousInvViewProj, clip);
        if (abs(worldH.w) < 1e-8f)
            return false;
        worldPos = worldH.xyz / worldH.w;
        return all(isfinite(worldPos));
    }

    void execute(uint2 pixel)
    {
        if (!params.isPixelInBounds(pixel))
            return;

        const float2 primary = primaryMotion[pixel];
        uint2 previousPixel;
        const bool primaryInBounds = params.reprojectPixel(pixel, primary, previousPixel);

        float2 dual = primary;
        bool dualValid = false;

        if (params.hasFlag(kEnableDualMotion) && params.hasFlag(kHistoryValid))
        {
            float3 occWorld;
            if (unprojectPrevious(pixel, occWorld))
            {
                const float4 currH = mul(gScene.camera.data.viewProjMatNoJitter, float4(occWorld, 1.f));
                if (abs(currH.w) > 1e-8f)
                {
                    const float2 currNdc = currH.xy / currH.w;
#ifdef FALCOR_FLIP_Y
                    const float2 currUv = currNdc * 0.5f + 0.5f;
#else
                    const float2 currUv = float2(currNdc.x * 0.5f + 0.5f, -currNdc.y * 0.5f + 0.5f);
#endif
                    const float2 currPos = currUv * float2(params.frameDim);
                    const float2 occMotion = currPos - (float2(pixel) + 0.5f);
                    dual = -occMotion / float2(params.frameDim);
                    uint2 dualPrev;
                    dualValid = params.reprojectPixel(pixel, dual, dualPrev);
                }
            }
        }

        dualMotion[pixel] = dual;
        uint validity = primaryInBounds ? kMotionPrimaryValid : 0u;
        if (dualValid)
            validity |= kMotionDualValid;
        motionValidity[pixel] = validity;
    }
};

ParameterBlock<BuildDualMotionPass> gPass;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.xy);
}
