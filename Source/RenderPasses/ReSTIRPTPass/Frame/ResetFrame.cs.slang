import RenderPasses.ReSTIRPTPass.Params;

struct ResetFramePass
{
    RWStructuredBuffer<uint> counters;

    void execute()
    {
        [unroll]
        for (uint i = 0u; i < (uint)CounterIndex::Count; ++i)
            counters[i] = 0u;
    }
};

ParameterBlock<ResetFramePass> gPass;

[numthreads(1, 1, 1)]
void main()
{
    gPass.execute();
}
