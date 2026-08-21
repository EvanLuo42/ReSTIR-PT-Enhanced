import Utils.Sampling.SampleGenerator;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.GRIS;

struct SpatialResamplePass
{
    RestirParams params;

    StructuredBuffer<PathReservoir> currentReservoirs;
    StructuredBuffer<uint> pairingPartners;
    StructuredBuffer<ShiftResult> spatialShifts;

    RWStructuredBuffer<PathReservoir> historyReservoirs;
    RWStructuredBuffer<BirthIdentity> birthIdentities;

    RWTexture2D<float4> outputColor;

    void execute(uint2 pixel)
    {
        if (!params.isPixelInBounds(pixel))
            return;

        const uint index = params.pixelIndex(pixel);
        const PathReservoir current = currentReservoirs[index];
        const float3 primaryEmission = outputColor[pixel].rgb;
        const bool haveCurrent = isUsableReservoir(current);
        const float cCurrent = max(current.confidence, 1.f);

        uint partnerCount = 0u;
        float mCurrent = 0.f;
        const uint slotCount = params.pairingSlotCount;
        const uint pixelCount = params.pixelCount();
        for (uint slot = 0u; slot < slotCount; ++slot)
        {
            const uint partner = pairingPartners[slot * pixelCount + index];
            if (partner >= pixelCount || partner == index)
                continue;

            const uint awayIndex = slot * pixelCount + index;
            const uint towardIndex = slot * pixelCount + partner;
            const ShiftResult away = spatialShifts[awayIndex];
            const ShiftResult toward = spatialShifts[towardIndex];

            if (!isShiftEvaluated(away.flags) && !isShiftEvaluated(toward.flags))
                continue;

            if (haveCurrent)
            {
                const float cN = max(currentReservoirs[partner].confidence, 1.f);
                if ((away.flags & kRecordValid) != 0u)
                {
                    mCurrent += pairwiseMISWeight(
                        cCurrent,
                        current.sample.target,
                        cN,
                        away.target,
                        away.jacobian
                    );
                }
                else
                {

                    mCurrent += 1.f;
                }
            }
            partnerCount += 1u;
        }

        if (partnerCount == 0u)
        {
            historyReservoirs[index] = current;
            birthIdentities[index] = getBirthIdentity(current.sample);
            outputColor[pixel] = float4(primaryEmission + reservoirScalarContribution(current), 0.f);
            return;
        }

        const float n = float(partnerCount);
        SampleGenerator sg = SampleGenerator(pixel, params.frameIndex * 13u + 7u);
        GRISState gris = makeGRISState();
        float3 vectorSum = 0.f;
        if (haveCurrent)
            mCurrent /= n;

        const float wCurrent = mCurrent * current.sample.target * max(current.ucw, 0.f);
        const float3 vCurrent = current.sample.contribution * mCurrent * max(current.ucw, 0.f);
        if (grisSelect(gris, wCurrent, cCurrent, haveCurrent, sg))
            gris.reservoir.sample = current.sample;
        vectorSum += vCurrent;

        for (uint slot = 0u; slot < slotCount; ++slot)
        {
            const uint partner = pairingPartners[slot * pixelCount + index];
            if (partner >= pixelCount || partner == index)
                continue;

            const uint awayIndex = slot * pixelCount + index;
            const uint shiftIndex = slot * pixelCount + partner;
            const ShiftResult toward = spatialShifts[shiftIndex];
            const ShiftResult away = spatialShifts[awayIndex];
            if (!isShiftEvaluated(away.flags) && !isShiftEvaluated(toward.flags))
                continue;

            const PathReservoir neighbor = currentReservoirs[partner];
            const float cN = max(neighbor.confidence, 1.f);
            const bool towardValid = (toward.flags & kRecordValid) != 0u &&
                toward.target > 0.f && isUsableReservoir(neighbor) && toward.jacobian > 0.f;

            float wN = 0.f;
            if (towardValid)
            {
                const float jN = toward.jacobian;
                const float invJ = 1.f / jN;
                const float mN = pairwiseMISWeight(
                    cN,
                    toward.target,
                    cCurrent,
                    neighbor.sample.target,
                    invJ
                ) / n;
                wN = mN * toward.target * max(neighbor.ucw, 0.f) * jN;
                vectorSum += toward.contribution * mN * max(neighbor.ucw, 0.f) * jN;
            }

            if (grisSelect(gris, wN, cN, towardValid, sg))
                gris.reservoir.sample = applyShiftResult(neighbor.sample, toward);
        }

        PathReservoir output = finalizeGRIS(gris);

        float3 color = primaryEmission;
        if (params.hasFeatureFlag(kEnableVectorWeights) && any(vectorSum != 0.f))
            color += vectorSum;
        else if (isValidPath(output.sample))
            color += output.sample.contribution * max(output.ucw, 0.f);
        else
            color += vectorSum;

        historyReservoirs[index] = output;
        birthIdentities[index] = getBirthIdentity(output.sample);
        outputColor[pixel] = float4(color, 0.f);
    }
};

ParameterBlock<SpatialResamplePass> gPass;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.xy);
}
