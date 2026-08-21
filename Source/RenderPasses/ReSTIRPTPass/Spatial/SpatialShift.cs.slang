#include "Scene/SceneDefines.slangh"

import Scene.RaytracingInline;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.Surface;
import RenderPasses.ReSTIRPTPass.Common.HybridShift;

struct SpatialShiftPass
{
    RestirParams params;
    EnvMapSampler envMapSampler;
    EmissiveLightSampler emissiveSampler;

    StructuredBuffer<PathReservoir> currentReservoirs;
    StructuredBuffer<uint> pairingPartners;
    Texture2D<PackedHitInfo> vbuffer;

    RWStructuredBuffer<ShiftResult> spatialShifts;
    RWStructuredBuffer<ReplayWorkItem> replayQueue;
    RWStructuredBuffer<uint> counters;

    void execute(uint3 threadID)
    {
        const uint2 pixel = threadID.xy;
        const uint slot = threadID.z;
        if (!params.isPixelInBounds(pixel) || slot >= params.pairingSlotCount)
            return;

        const uint pixelIndex = params.pixelIndex(pixel);
        const uint partner = pairingPartners[slot * params.pixelCount() + pixelIndex];
        if (partner >= params.pixelCount())
            return;

        const uint pixelA = min(pixelIndex, partner);
        const uint pixelB = max(pixelIndex, partner);
        if (pixelIndex != pixelA)
            return;

        if (pixelA == pixelB)
            return;

        const uint recordBase = slot * params.pixelCount();
        const uint recordA = recordBase + pixelA;
        const uint recordB = recordBase + pixelB;
        const ShiftResult invalid = makeInvalidShiftResult();

        const PathReservoir resA = currentReservoirs[pixelA];
        const PathReservoir resB = currentReservoirs[pixelB];

        const uint2 coordA = params.pixelCoord(pixelA);
        const uint2 coordB = params.pixelCoord(pixelB);
        SurfaceGeometry geomA;
        SurfaceGeometry geomB;
        if (!loadSurfaceGeometryFromVBuffer(vbuffer, coordA, params.frameDim, geomA) ||
            !loadSurfaceGeometryFromVBuffer(vbuffer, coordB, params.frameDim, geomB) ||
            !surfacesCompatible(geomA, geomB, params))
        {
            spatialShifts[recordA] = invalid;
            spatialShifts[recordB] = invalid;
            return;
        }

        const bool needReplay = needsRandomReplay(resA.sample) || needsRandomReplay(resB.sample);
        if (needReplay)
        {
            uint queueIndex;
            InterlockedAdd(counters[(uint)CounterIndex::SpatialReplay], 1u, queueIndex);
            if (queueIndex >= params.spatialReplayCapacity)
            {
                InterlockedAdd(counters[(uint)CounterIndex::QueueOverflow], 1u);
                spatialShifts[recordA] = invalid;
                spatialShifts[recordB] = invalid;
                return;
            }

            ShiftResult queued = invalid;
            queued.flags = kRecordNeedsReplay;
            spatialShifts[recordA] = queued;
            spatialShifts[recordB] = queued;

            ReplayWorkItem item = {};
            item.recordIndex = recordBase;
            item.pixelA = pixelA;
            item.pixelB = pixelB;
            replayQueue[queueIndex] = item;
            return;
        }

        if (!isValidPath(resA.sample) && !isValidPath(resB.sample))
        {
            spatialShifts[recordA] = makeFailedShift();
            spatialShifts[recordB] = makeFailedShift();
            return;
        }

        Surface surfA;
        Surface surfB;
        if (!loadSurfaceFromVBuffer(vbuffer, coordA, params, surfA, false) ||
            !loadSurfaceFromVBuffer(vbuffer, coordB, params, surfB, false))
        {
            spatialShifts[recordA] = invalid;
            spatialShifts[recordB] = invalid;
            return;
        }

        spatialShifts[recordA] = evaluateHybridShift(
            resA.sample, surfB, params, coordB, envMapSampler, emissiveSampler,
            GeometryTime::Current, GeometryTime::Current
        );
        spatialShifts[recordB] = evaluateHybridShift(
            resB.sample, surfA, params, coordA, envMapSampler, emissiveSampler,
            GeometryTime::Current, GeometryTime::Current
        );
    }
};

ParameterBlock<SpatialShiftPass> gPass;

[numthreads(8, 8, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID);
}
