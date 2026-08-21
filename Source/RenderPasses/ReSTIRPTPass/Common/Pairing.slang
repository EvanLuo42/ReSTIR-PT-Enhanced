import RenderPasses.ReSTIRPTPass.Params;

int wrapCoord(int v, int m)
{
    int r = v % m;
    return r < 0 ? r + m : r;
}

int2 applyPairingTransform(int2 p, uint transform)
{
    if ((transform & 1u) != 0u)
        p.x = -p.x;
    if ((transform & 2u) != 0u)
        p.y = -p.y;
    if ((transform & 4u) != 0u)
        p = p.yx;
    return p;
}

int2 applyPairingInverse(int2 p, uint transform)
{
    if ((transform & 4u) != 0u)
        p = p.yx;
    if ((transform & 1u) != 0u)
        p.x = -p.x;
    if ((transform & 2u) != 0u)
        p.y = -p.y;
    return p;
}

int2 samplePairingDelta(
    uint2 pixel,
    PairingSetDesc set,
    StructuredBuffer<int2> pairingBaseOffsets
)
{
    const int2 p = int2(pixel) + int2(set.frameOffsetX, set.frameOffsetY);
    const int2 q = applyPairingInverse(p, set.transform);
    const int2 dim = int2(set.dimX, set.dimY);
    const uint2 uv = uint2(wrapCoord(q.x, dim.x), wrapCoord(q.y, dim.y));
    const int2 delta = pairingBaseOffsets[set.baseOffset + uv.y * set.dimX + uv.x];
    return applyPairingTransform(delta, set.transform);
}

uint resolvePairingPartner(
    uint2 pixel,
    uint slot,
    RestirParams params,
    StructuredBuffer<int2> pairingBaseOffsets,
    StructuredBuffer<PairingSetDesc> pairingSets
)
{
    const uint self = params.pixelIndex(pixel);
    if (slot >= params.pairingSlotCount)
        return self;

    const PairingSetDesc set = pairingSets[slot];
    const int2 delta = samplePairingDelta(pixel, set, pairingBaseOffsets);
    const int2 partnerCoord = int2(pixel) + delta;
    if (any(partnerCoord < int2(0)) || any(partnerCoord >= int2(params.frameDim)))
        return self;

    const uint2 partnerPixel = uint2(partnerCoord);
    const int2 back = samplePairingDelta(partnerPixel, set, pairingBaseOffsets);
    if (any(int2(partnerPixel) + back != int2(pixel)))
        return self;
    return params.pixelIndex(partnerPixel);
}
