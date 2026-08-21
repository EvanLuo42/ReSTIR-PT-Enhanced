import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import Rendering.Materials.LobeType;

uint packTechniqueKey(TechniqueKind kind, uint s, uint t, uint pathLength)
{
    return ((uint)kind & 0xfu) |
        ((s & 0xfu) << 4u) |
        ((t & 0xfu) << 8u) |
        ((pathLength & 0xfu) << 12u);
}

uint getPathLength(PathSample sample)
{
    return (sample.techniqueKey >> 12u) & 0xfu;
}

bool isValidPath(PathSample sample)
{
    return (sample.flags & kPathValid) != 0u;
}

uint getPathRisM(PathSample sample)
{
    return max(1u, (sample.flags & kPathRisMMask) >> kPathRisMShift);
}

uint packPathRisM(uint flags, uint risM)
{
    return (flags & ~kPathRisMMask) | ((min(risM, 255u) << kPathRisMShift) & kPathRisMMask);
}

uint neePathFlags(uint lightType)
{
    uint flags = kPathValid | kPathEndsWithNEE;
    if (lightType == 0u)
        flags |= kPathEndsOnEnvironment;
    if (lightType == 2u)
        flags |= kPathEndsOnAnalytic;
    return flags;
}

uint cameraHitFlags(bool envMiss)
{
    return envMiss ? (kPathValid | kPathEndsOnEnvironment) : kPathValid;
}

bool isLightEndpointReconnect(PathSample sample)
{
    return ((sample.flags & kPathEndsWithNEE) != 0u) &&
        sample.reconnectionVertex != kInvalidIndex &&
        sample.reconnectionVertex == getPathLength(sample);
}

bool isBsdfHitEmissiveEndpointReconnect(PathSample sample)
{
    return ((sample.flags & kPathEndsWithNEE) == 0u) &&
        ((sample.flags & kPathHasReconnectBSDF) == 0u) &&
        sample.reconnectionVertex != kInvalidIndex &&
        sample.reconnectionVertex == getPathLength(sample);
}

BirthIdentity getBirthIdentity(PathSample sample)
{
    BirthIdentity identity;
    const uint validBit = isValidPath(sample) ? 0x80000000u : 0u;
    const uint packed = validBit | sample.techniqueKey;
    identity.value = uint4(sample.randomSeed, sample.candidateID, packed);
    return identity;
}

bool isValidBirthIdentity(BirthIdentity identity)
{
    return (identity.value.w & 0x80000000u) != 0u;
}

bool hasSameBirth(BirthIdentity a, BirthIdentity b)
{
    return all(a.value == b.value);
}

float3 safeComponentRatio(float3 numerator, float3 denominator)
{

    return select(denominator > 1e-8f, numerator / denominator, float3(0.f));
}

ShiftResult encodeShiftResult(PathSample shifted, float jacobian)
{
    ShiftResult result = {};
    result.contribution = shifted.contribution;
    result.target = shifted.target;
    result.proposalPdf = shifted.proposalPdf;
    result.reconnectDensity = shifted.reconnectDensity;
    result.jacobian = jacobian;
    result.reconnectionVertex = shifted.reconnectionVertex;
    result.flags = kRecordValid;
    return result;
}

PathSample applyShiftResult(PathSample source, ShiftResult result)
{
    PathSample shifted = source;
    shifted.contribution = result.contribution;
    shifted.target = result.target;
    shifted.proposalPdf = result.proposalPdf;
    shifted.reconnectDensity = result.reconnectDensity;
    shifted.reconnectionVertex = result.reconnectionVertex;
    return shifted;
}

bool isShiftEvaluated(uint flags)
{
    return (flags & (kRecordValid | kRecordReplayFailed)) != 0u;
}

bool passesEnhancedReconnectionGate(
    float3 prefixPosition,
    float3 prefixNormal,
    float3 reconnectionPosition,
    float3 reconnectionNormal,
    float pFrom,
    float pAt,
    bool skipInverse,
    float primaryFootprint,
    float reconnectionRoughness,
    RestirParams params,
    out float forwardGeometry
)
{
    forwardGeometry = geometryTerm(prefixPosition, reconnectionPosition, reconnectionNormal);
    const float fwdDen = pFrom * forwardGeometry;
    const float forwardFootprint = fwdDen > 0.f ? rcp(fwdDen) : 0.f;
    float inverseFootprint = 0.f;
    if (!skipInverse)
    {
        const float gInv = geometryTerm(reconnectionPosition, prefixPosition, prefixNormal);
        const float invDen = pAt * gInv;
        inverseFootprint = invDen > 0.f ? rcp(invDen) : 0.f;
    }
    const float dualFootprint = skipInverse
        ? forwardFootprint
        : min(forwardFootprint, inverseFootprint);

    const float scale = params.footprintScale > 1e-6f ? params.footprintScale : 0.0002f;
    const bool footprintAccepted = dualFootprint >= scale * primaryFootprint;
    const bool roughnessAccepted = reconnectionRoughness >= params.minReconnectionRoughness;
    return isfinite(dualFootprint) && isfinite(primaryFootprint) && footprintAccepted && roughnessAccepted;
}

uint2 packLobe(uint2 packed, uint surfaceVertex, uint lobe)
{
    if (surfaceVertex == 0u || surfaceVertex > 8u)
        return packed;
    const uint word = (surfaceVertex - 1u) >> 2u;
    const uint shift = ((surfaceVertex - 1u) & 3u) * 8u;
    const uint mask = 0xffu << shift;
    packed[word] = (packed[word] & ~mask) | ((lobe & 0xffu) << shift);
    return packed;
}

uint unpackLobe(uint2 packed, uint surfaceVertex)
{
    if (surfaceVertex == 0u || surfaceVertex > 8u)
        return 0u;
    const uint word = (surfaceVertex - 1u) >> 2u;
    const uint shift = ((surfaceVertex - 1u) & 3u) * 8u;
    return (packed[word] >> shift) & 0xffu;
}

static const uint kConcreteLobeMask =
    (uint)LobeType::DiffuseReflection |
    (uint)LobeType::SpecularReflection |
    (uint)LobeType::DeltaReflection |
    (uint)LobeType::DiffuseTransmission |
    (uint)LobeType::SpecularTransmission |
    (uint)LobeType::DeltaTransmission;

uint concreteLobeCount(uint lobeTypes)
{
    return countbits(lobeTypes & kConcreteLobeMask);
}

bool hasExactSingleLobe(uint lobeTypes, uint sampledLobe)
{
    return sampledLobe != 0u &&
        (sampledLobe & (uint)LobeType::Delta) == 0u &&
        concreteLobeCount(lobeTypes) == 1u &&
        (lobeTypes & sampledLobe) != 0u;
}

bool isDiffuseLobe(uint lobe)
{
    return (lobe & (uint)LobeType::Diffuse) != 0u;
}

bool isDeltaLobe(uint lobe)
{
    return (lobe & (uint)LobeType::Delta) != 0u;
}

float sampledLobeAlpha(uint lobe, float materialAlpha)
{
    if (isDeltaLobe(lobe))
        return 0.f;
    if (isDiffuseLobe(lobe))
        return 1.f;
    return max(materialAlpha, 0.f);
}

float geometryTerm(float3 fromPos, float3 toPos, float3 toNormal)
{
    const float3 delta = toPos - fromPos;
    const float dist2 = max(dot(delta, delta), 1e-12f);
    const float invDist = rsqrt(dist2);

    return max(dot(toNormal, -delta), 0.f) * invDist * invDist * invDist;
}

float primaryRayFootprint(float3 cameraPos, float3 primaryPos, float3 primaryNormal)
{
    const float3 toCamera = cameraPos - primaryPos;
    const float dist2 = max(dot(toCamera, toCamera), 1e-12f);
    const float nDotOmega = max(abs(dot(primaryNormal, normalize(toCamera))), 1e-6f);
    return dist2 / (nDotOmega / (4.f * 3.141592653589793f));
}

float lightSolidAnglePdfFromArea(
    float areaPdf,
    float3 prefixPosition,
    float3 lightPosition,
    float3 lightNormal
)
{
    const float g = geometryTerm(prefixPosition, lightPosition, lightNormal);
    return areaPdf > 0.f && g > 0.f ? areaPdf / g : 0.f;
}

float getDuplicationConfidenceCap(float duplication, RestirParams params)
{
    if (!params.hasFeatureFlag(kEnableDuplicationMap))
        return params.cDefault;

    const float d = saturate(duplication);
    return lerp(params.cDefault, params.cMin, pow(d, params.duplicationExponent));
}
