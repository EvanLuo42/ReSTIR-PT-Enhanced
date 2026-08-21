import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;

static const uint kDupGroupWidth = 16u;
static const uint kDupGroupHeight = 16u;
static const uint kDupTileRadius = 8u;
static const uint kDupTileWidth = kDupGroupWidth + 2u * kDupTileRadius;
static const uint kDupTileHeight = kDupGroupHeight + 2u * kDupTileRadius;
static const uint kDupTileSize = kDupTileWidth * kDupTileHeight;
static const uint kDupGroupSize = kDupGroupWidth * kDupGroupHeight;
static const uint kDupBucketCount = 2048u;
static const uint kDupBucketMask = kDupBucketCount - 1u;

uint hashBirthIdentity(uint4 value)
{
    uint h = 2166136261u;
    h = (h ^ value.x) * 16777619u;
    h = (h ^ value.y) * 16777619u;
    h = (h ^ value.z) * 16777619u;
    h = (h ^ value.w) * 16777619u;
    return h;
}

struct DuplicationMapPass
{
    RestirParams params;
    StructuredBuffer<BirthIdentity> birthIdentities;
    RWTexture2D<float> duplicationMap;

    static groupshared uint4 sIdentity[kDupTileSize];
    static groupshared uint sNext[kDupTileSize];
    static groupshared uint sBucketHeads[kDupBucketCount];

    void execute(uint2 pixel, uint2 groupID, uint2 groupThreadID)
    {
        const bool pixelInBounds = params.isPixelInBounds(pixel);
        if (!params.hasFeatureFlag(kEnableDuplicationMap))
        {
            if (pixelInBounds)
                duplicationMap[pixel] = 0.f;
            return;
        }

        const uint localThread = groupThreadID.y * kDupGroupWidth + groupThreadID.x;
        const int2 tileOrigin = int2(groupID * uint2(kDupGroupWidth, kDupGroupHeight)) -
            int2(kDupTileRadius);

        for (uint bucket = localThread; bucket < kDupBucketCount; bucket += kDupGroupSize)
            sBucketHeads[bucket] = kInvalidIndex;
        GroupMemoryBarrierWithGroupSync();

        for (uint tileIndex = localThread; tileIndex < kDupTileSize; tileIndex += kDupGroupSize)
        {
            const int2 tileCoord = int2(tileIndex % kDupTileWidth, tileIndex / kDupTileWidth);
            const int2 sourceCoord = tileOrigin + tileCoord;
            BirthIdentity identity = {};
            if (all(sourceCoord >= int2(0)) && all(sourceCoord < int2(params.frameDim)))
                identity = birthIdentities[params.pixelIndex(uint2(sourceCoord))];

            sIdentity[tileIndex] = identity.value;
            const uint bucket = hashBirthIdentity(identity.value) & kDupBucketMask;
            uint previous;
            InterlockedExchange(sBucketHeads[bucket], tileIndex, previous);
            sNext[tileIndex] = previous;
        }
        GroupMemoryBarrierWithGroupSync();

        if (!pixelInBounds)
            return;

        const uint centerIndex = (groupThreadID.y + kDupTileRadius) * kDupTileWidth +
            groupThreadID.x + kDupTileRadius;
        BirthIdentity center;
        center.value = sIdentity[centerIndex];
        if (!isValidBirthIdentity(center))
        {
            duplicationMap[pixel] = 0.f;
            return;
        }

        const uint radius = params.duplicationRadius;
        uint duplicateCount = 0u;
        if (radius <= kDupTileRadius)
        {
            const uint bucket = hashBirthIdentity(center.value) & kDupBucketMask;
            uint neighborIndex = sBucketHeads[bucket];
            const int2 centerCoord = int2(
                groupThreadID.x + kDupTileRadius,
                groupThreadID.y + kDupTileRadius
            );
            while (neighborIndex != kInvalidIndex)
            {
                if (neighborIndex != centerIndex)
                {
                    const int2 neighborCoord = int2(
                        neighborIndex % kDupTileWidth,
                        neighborIndex / kDupTileWidth
                    );
                    const int2 delta = abs(neighborCoord - centerCoord);
                    if (all(delta <= int2(radius)) &&
                        all(sIdentity[neighborIndex] == center.value))
                    {
                        duplicateCount += 1u;
                    }
                }
                neighborIndex = sNext[neighborIndex];
            }
        }
        else
        {

            for (int y = -int(radius); y <= int(radius); ++y)
            {
                for (int x = -int(radius); x <= int(radius); ++x)
                {
                    if (x == 0 && y == 0)
                        continue;
                    const int2 neighbor = int2(pixel) + int2(x, y);
                    if (any(neighbor < int2(0)) || any(neighbor >= int2(params.frameDim)))
                        continue;
                    const BirthIdentity candidate = birthIdentities[params.pixelIndex(uint2(neighbor))];
                    duplicateCount += hasSameBirth(center, candidate) ? 1u : 0u;
                }
            }
        }

        const uint diameter = 2u * radius + 1u;
        const uint neighborCount = max(1u, diameter * diameter - 1u);
        duplicationMap[pixel] = float(duplicateCount) / float(neighborCount);
    }
};

ParameterBlock<DuplicationMapPass> gPass;

[numthreads(kDupGroupWidth, kDupGroupHeight, 1)]
void main(
    uint3 dispatchThreadID : SV_DispatchThreadID,
    uint3 groupID : SV_GroupID,
    uint3 groupThreadID : SV_GroupThreadID
)
{
    gPass.execute(dispatchThreadID.xy, groupID.xy, groupThreadID.xy);
}
