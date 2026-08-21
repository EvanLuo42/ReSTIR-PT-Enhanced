import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.Pairing;

struct BuildPairingPass
{
    RestirParams params;

    StructuredBuffer<int2> pairingBaseOffsets;
    StructuredBuffer<PairingSetDesc> pairingSets;
    RWStructuredBuffer<uint> pairingPartners;

    void execute(uint3 threadID)
    {
        const uint2 pixel = threadID.xy;
        const uint slot = threadID.z;
        if (!params.isPixelInBounds(pixel) || slot >= params.pairingSlotCount)
            return;

        const uint index = params.pixelIndex(pixel);
        const uint outputIndex = slot * params.pixelCount() + index;
        pairingPartners[outputIndex] = resolvePairingPartner(
            pixel,
            slot,
            params,
            pairingBaseOffsets,
            pairingSets
        );
    }
};

ParameterBlock<BuildPairingPass> gPass;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID);
}
