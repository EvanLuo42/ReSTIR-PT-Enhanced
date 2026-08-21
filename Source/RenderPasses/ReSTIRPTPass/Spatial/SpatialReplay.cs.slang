#include "Scene/SceneDefines.slangh"

import Scene.RaytracingInline;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.Surface;
import RenderPasses.ReSTIRPTPass.Common.HybridShift;

struct SpatialReplayPass
{
    RestirParams params;
    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;

    StructuredBuffer<PathReservoir> currentReservoirs;
    Texture2D<PackedHitInfo> vbuffer;
    StructuredBuffer<ReplayWorkItem> replayQueue;
    RWStructuredBuffer<ShiftResult> spatialShifts;
    RWStructuredBuffer<uint> counters;

    void execute(uint directionIndex)
    {
        const uint queueIndex = directionIndex >> 1u;
        const uint count = min(
            counters[(uint)CounterIndex::SpatialReplay],
            params.spatialReplayCapacity
        );
        if (queueIndex >= count)
            return;

        const ReplayWorkItem item = replayQueue[queueIndex];
        const bool reverse = (directionIndex & 1u) != 0u;
        const uint sourceIndex = reverse ? item.pixelB : item.pixelA;
        const uint targetIndex = reverse ? item.pixelA : item.pixelB;
        const uint recordIndex = item.recordIndex + sourceIndex;

        Surface targetSurface;
        const uint2 targetPixel = params.pixelCoord(targetIndex);
        if (!loadSurfaceFromVBuffer(vbuffer, targetPixel, params, targetSurface, false))
        {
            spatialShifts[recordIndex] = makeInvalidShiftResult();
            return;
        }

        spatialShifts[recordIndex] = evaluateHybridShift(
            currentReservoirs[sourceIndex].sample,
            targetSurface,
            params,
            targetPixel,
            envMapSampler,
            emissiveSampler,
            GeometryTime::Current,
            GeometryTime::Current
        );
    }
};

ParameterBlock<SpatialReplayPass> gPass;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.x);
}
