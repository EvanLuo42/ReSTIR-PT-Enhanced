#include "Scene/SceneDefines.slangh"

import Scene.RaytracingInline;
import Utils.Sampling.SampleGenerator;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.Surface;
import RenderPasses.ReSTIRPTPass.Common.HybridShift;

struct TemporalShiftPass
{
    RestirParams params;
    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;

    Texture2D<float2> primaryMotion;
    Texture2D<float2> dualMotion;
    Texture2D<uint> motionValidity;
    Texture2D<PackedHitInfo> currentVBuffer;
    Texture2D<PackedHitInfo> previousVBuffer;
    StructuredBuffer<PathReservoir> currentReservoirs;
    StructuredBuffer<PathReservoir> historyReservoirs;

    RWStructuredBuffer<TemporalShiftRecord> shiftRecords;
    RWStructuredBuffer<ReplayWorkItem> replayQueue;
    RWStructuredBuffer<uint> counters;

    bool selectHistoryPixel(uint2 pixel, SurfaceGeometry current, out uint2 previousPixel, out SurfaceGeometry history)
    {
        previousPixel = uint2(0u);
        history = {};
        uint2 primaryPrev;
        bool primaryValid = params.reprojectPixel(pixel, primaryMotion[pixel], primaryPrev);

        if (params.hasFlag(kEnableDualMotion))
        {
            const uint validity = motionValidity[pixel];
            primaryValid = primaryValid && ((validity & kMotionPrimaryValid) != 0u);
        }

        if (primaryValid)
        {
            SurfaceGeometry candidate;
            if (loadPreviousSurfaceGeometryFromVBuffer(previousVBuffer, primaryPrev, candidate) &&
                temporalSurfacesCompatible(current, candidate, params))
            {
                previousPixel = primaryPrev;
                history = candidate;
                return true;
            }
        }

        if (params.hasFlag(kEnableDualMotion) && ((motionValidity[pixel] & kMotionDualValid) != 0u))
        {
            uint2 dualPrev;
            if (params.reprojectPixel(pixel, dualMotion[pixel], dualPrev))
            {
                SurfaceGeometry candidate;
                if (loadPreviousSurfaceGeometryFromVBuffer(previousVBuffer, dualPrev, candidate) &&
                    temporalSurfacesCompatible(current, candidate, params))
                {
                    previousPixel = dualPrev;
                    history = candidate;
                    return true;
                }
            }
        }

        return false;
    }

    void enqueueReplay(uint index, uint historyIndex, inout TemporalShiftRecord record)
    {
        uint queueIndex;
        InterlockedAdd(counters[(uint)CounterIndex::TemporalReplay], 1u, queueIndex);
        if (queueIndex >= params.temporalReplayCapacity)
        {
            InterlockedAdd(counters[(uint)CounterIndex::QueueOverflow], 1u);
            record.currentToHistory.flags = kRecordReplayFailed;
            record.historyToCurrent.flags = kRecordReplayFailed;
            shiftRecords[index] = record;
            return;
        }

        ReplayWorkItem item = {};
        item.recordIndex = index;
        item.pixelA = index;
        item.pixelB = historyIndex;
        replayQueue[queueIndex] = item;
        shiftRecords[index] = record;
    }

    void execute(uint2 pixel)
    {
        if (!params.isPixelInBounds(pixel))
            return;

        const uint index = params.pixelIndex(pixel);
        TemporalShiftRecord record = {};
        record.historyPixel = kInvalidIndex;
        record.currentToHistory = makeInvalidShiftResult();
        record.historyToCurrent = makeInvalidShiftResult();

        if (!params.hasFlag(kHistoryValid))
            return;

        SurfaceGeometry currentGeom;
        if (!loadSurfaceGeometryFromVBuffer(currentVBuffer, pixel, params.frameDim, currentGeom))
        {
            shiftRecords[index] = record;
            return;
        }

        uint2 previousPixel;
        SurfaceGeometry historyGeom;
        if (!selectHistoryPixel(pixel, currentGeom, previousPixel, historyGeom))
        {
            shiftRecords[index] = record;
            return;
        }

        const uint historyIndex = params.pixelIndex(previousPixel);
        record.historyPixel = historyIndex;

        const PathReservoir currentR = currentReservoirs[index];
        const PathReservoir historyR = historyReservoirs[historyIndex];

        if (all(previousPixel == pixel) && isIdentitySurfaceMapping(currentGeom, historyGeom))
        {
            record.currentToHistory = isValidPath(currentR.sample)
                ? identityShift(currentR.sample)
                : makeFailedShift();
            record.historyToCurrent = isValidPath(historyR.sample)
                ? identityShift(historyR.sample)
                : makeFailedShift();
            shiftRecords[index] = record;
            return;
        }

        const bool needReplay =
            needsRandomReplay(currentR.sample) || needsRandomReplay(historyR.sample);

        if (needReplay)
        {
            record.currentToHistory.flags = kRecordNeedsReplay;
            record.historyToCurrent.flags = kRecordNeedsReplay;
            enqueueReplay(index, historyIndex, record);
            return;
        }

        if (!isValidPath(currentR.sample) && !isValidPath(historyR.sample))
        {
            record.currentToHistory = makeFailedShift();
            record.historyToCurrent = makeFailedShift();
            shiftRecords[index] = record;
            return;
        }

        Surface current;
        Surface history;
        if (!loadSurfaceFromVBuffer(currentVBuffer, pixel, params, current, false) ||
            !loadPreviousSurfaceFromVBuffer(previousVBuffer, previousPixel, params, history, false))
        {
            record.currentToHistory = makeFailedShift();
            record.historyToCurrent = makeFailedShift();
            shiftRecords[index] = record;
            return;
        }

        record.currentToHistory = evaluateHybridShift(
            currentR.sample, history, params, previousPixel, envMapSampler, emissiveSampler,
            GeometryTime::Current, GeometryTime::Previous
        );
        record.historyToCurrent = evaluateHybridShift(
            historyR.sample, current, params, pixel, envMapSampler, emissiveSampler,
            GeometryTime::Previous, GeometryTime::Current
        );
        shiftRecords[index] = record;
    }
};

ParameterBlock<TemporalShiftPass> gPass;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.xy);
}
