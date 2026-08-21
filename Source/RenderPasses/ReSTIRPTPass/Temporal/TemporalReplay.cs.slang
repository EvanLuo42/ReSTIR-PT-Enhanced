#include "Scene/SceneDefines.slangh"

import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.Surface;
import RenderPasses.ReSTIRPTPass.Common.HybridShift;

struct TemporalReplayPass
{
    RestirParams params;
    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;

    StructuredBuffer<PathReservoir> currentReservoirs;
    StructuredBuffer<PathReservoir> historyReservoirs;
    Texture2D<PackedHitInfo> currentVBuffer;
    Texture2D<PackedHitInfo> previousVBuffer;
    StructuredBuffer<ReplayWorkItem> replayQueue;
    RWStructuredBuffer<TemporalShiftRecord> shiftRecords;
    RWStructuredBuffer<uint> counters;

    void execute(uint queueIndex)
    {
        const uint count = min(
            counters[(uint)CounterIndex::TemporalReplay],
            params.temporalReplayCapacity
        );
        if (queueIndex >= count)
            return;

        const ReplayWorkItem item = replayQueue[queueIndex];
        TemporalShiftRecord record = shiftRecords[item.recordIndex];
        if (record.historyPixel == kInvalidIndex)
            return;

        const uint currentIndex = item.pixelA;
        const uint2 currentPixel = params.pixelCoord(currentIndex);
        const uint2 historyPixel = params.pixelCoord(record.historyPixel);

        Surface current;
        Surface history;
        if (!loadSurfaceFromVBuffer(currentVBuffer, currentPixel, params, current, false) ||
            !loadPreviousSurfaceFromVBuffer(previousVBuffer, historyPixel, params, history, false))
        {
            record.currentToHistory = makeFailedShift();
            record.historyToCurrent = makeFailedShift();
            shiftRecords[item.recordIndex] = record;
            return;
        }

        const PathReservoir currentR = currentReservoirs[currentIndex];
        const PathReservoir historyR = historyReservoirs[record.historyPixel];
        record.currentToHistory = evaluateHybridShift(
            currentR.sample, history, params, historyPixel, envMapSampler, emissiveSampler,
            GeometryTime::Current, GeometryTime::Previous
        );
        record.historyToCurrent = evaluateHybridShift(
            historyR.sample, current, params, currentPixel, envMapSampler, emissiveSampler,
            GeometryTime::Previous, GeometryTime::Current
        );
        shiftRecords[item.recordIndex] = record;
    }
};

ParameterBlock<TemporalReplayPass> gPass;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.x);
}
