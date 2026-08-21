import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;

struct ReflectTypesPass
{
    RWStructuredBuffer<PathReservoir> pathReservoirs;
    RWStructuredBuffer<BirthIdentity> birthIdentities;
    RWStructuredBuffer<TemporalShiftRecord> temporalShifts;
    RWStructuredBuffer<ShiftResult> spatialShifts;
    RWStructuredBuffer<ReplayWorkItem> replayQueue;
    RWStructuredBuffer<LightTileEntry> lightTiles;
    RWStructuredBuffer<uint> pairingPartners;
    RWStructuredBuffer<int2> pairingBaseOffsets;
    RWStructuredBuffer<PairingSetDesc> pairingSets;
    RWStructuredBuffer<uint> counters;
};

ParameterBlock<ReflectTypesPass> gPass;

[numthreads(1, 1, 1)]
void main() {}
