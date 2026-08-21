import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;
import RenderPasses.ReSTIRPTPass.Common.GRIS;

struct ResolvePass
{
    RestirParams params;

    StructuredBuffer<PathReservoir> currentReservoirs;
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

        historyReservoirs[index] = current;
        birthIdentities[index] = getBirthIdentity(current.sample);
        outputColor[pixel] = float4(primaryEmission + reservoirScalarContribution(current), 0.f);
    }
};

ParameterBlock<ResolvePass> gPass;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.xy);
}
