struct BuildDispatchArgsPass
{
    uint counterIndex;
    uint queueCapacity;
    uint consumerGroupSize;
    uint argsByteOffset;
    uint workMultiplier;

    RWStructuredBuffer<uint> counters;
    RWByteAddressBuffer dispatchArgs;

    void execute(uint threadIndex)
    {
        if (threadIndex != 0u)
            return;

        const uint itemCount = min(counters[counterIndex], queueCapacity) * max(1u, workMultiplier);
        const uint groupSize = max(1u, consumerGroupSize);
        const uint groupCount = max(1u, (itemCount + groupSize - 1u) / groupSize);
        dispatchArgs.Store3(argsByteOffset, uint3(groupCount, 1u, 1u));
    }
};

ParameterBlock<BuildDispatchArgsPass> gPass;

[numthreads(1, 1, 1)]
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    gPass.execute(dispatchThreadID.x);
}
