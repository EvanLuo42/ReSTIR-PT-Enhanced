import Utils.Sampling.SampleGenerator;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.GRIS;

struct TemporalReusePass
{
    RestirParams params;

    RWStructuredBuffer<PathReservoir> currentReservoirs;
    StructuredBuffer<PathReservoir> historyReservoirs;
    StructuredBuffer<TemporalShiftRecord> shiftRecords;
    Texture2D<float> previousDuplicationMap;

    void execute(uint2 pixel)
    {
        if (!params.isPixelInBounds(pixel))
            return;

        const uint index = params.pixelIndex(pixel);
        if (!params.hasFlag(kHistoryValid))
            return;

        const TemporalShiftRecord shifts = shiftRecords[index];
        if (shifts.historyPixel == kInvalidIndex)
            return;

        const PathReservoir current = currentReservoirs[index];
        const PathReservoir history = historyReservoirs[shifts.historyPixel];
        const uint2 historyPixel = params.pixelCoord(shifts.historyPixel);
        const float cCap = params.hasFeatureFlag(kEnableDuplicationMap)
            ? getDuplicationConfidenceCap(previousDuplicationMap[historyPixel], params)
            : params.cDefault;

        const float cCurrent = max(current.confidence, 1.f);
        const float cHistory = min(max(history.confidence, 1.f), cCap);

        const bool currentCandidate = isUsableReservoir(current);
        const bool currentMap = currentCandidate &&
            (shifts.currentToHistory.flags & kRecordValid) != 0u;
        const bool historyCandidate = isUsableReservoir(history) &&
            (shifts.historyToCurrent.flags & kRecordValid) != 0u &&
            shifts.historyToCurrent.target > 0.f &&
            shifts.historyToCurrent.jacobian > 0.f;

        if (!historyCandidate)
            return;

        float mCurrent = 0.f;
        if (currentCandidate)
        {

            mCurrent = currentMap ? pairwiseMISWeight(
                cCurrent,
                current.sample.target,
                cHistory,
                shifts.currentToHistory.target,
                shifts.currentToHistory.jacobian
            ) : 1.f;
        }

        float mHistory = 0.f;
        float jHistory = 0.f;
        if (historyCandidate)
        {
            jHistory = shifts.historyToCurrent.jacobian;
            const float invJ = 1.f / jHistory;
            mHistory = pairwiseMISWeight(
                cHistory,
                shifts.historyToCurrent.target,
                cCurrent,
                history.sample.target,
                invJ
            );
        }

        const float wCurrent = mCurrent * current.sample.target * max(current.ucw, 0.f);
        const float wHistory = mHistory * shifts.historyToCurrent.target * max(history.ucw, 0.f) * jHistory;

        SampleGenerator sg = SampleGenerator(pixel, params.frameIndex * 17u + 3u);
        GRISState gris = makeGRISState();
        if (grisSelect(gris, wCurrent, cCurrent, currentCandidate, sg))
            gris.reservoir.sample = current.sample;
        if (grisSelect(gris, wHistory, cHistory, true, sg))
            gris.reservoir.sample = applyShiftResult(history.sample, shifts.historyToCurrent);

        currentReservoirs[index] = finalizeGRIS(gris);
    }
};

ParameterBlock<TemporalReusePass> gPass;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.xy);
}
