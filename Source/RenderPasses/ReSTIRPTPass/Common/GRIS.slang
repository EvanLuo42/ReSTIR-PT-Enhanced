import Utils.Sampling.SampleGenerator;
import Utils.Color.ColorHelpers;
import RenderPasses.ReSTIRPTPass.Common.Types;
import RenderPasses.ReSTIRPTPass.Params;
import RenderPasses.ReSTIRPTPass.Common.EnhancedPolicy;

float targetPdf(float3 contribution)
{
    const float t = luminance(contribution);
    return isfinite(t) && t > 0.f ? t : 0.f;
}

bool isFiniteFloat3(float3 v)
{
    return all(isfinite(v));
}

bool isUsableReservoir(PathReservoir r)
{
    return isValidPath(r.sample) && isfinite(r.ucw) && r.ucw >= 0.f;
}

float3 reservoirScalarContribution(PathReservoir r)
{
    return isUsableReservoir(r) ? r.sample.contribution * max(r.ucw, 0.f) : 0.f;
}

float pairwiseMISWeight(float cI, float pI, float cJ, float pJShifted, float jacobianIToJ)
{
    const float num = max(cI, 0.f) * pI;
    const float den = num + max(cJ, 0.f) * max(pJShifted, 0.f) * max(jacobianIToJ, 0.f);
    const bool ok = pI > 0.f && den > 0.f && isfinite(pI) && isfinite(den);
    return ok ? num / den : 0.f;
}

struct GRISState
{
    PathReservoir reservoir;
    float weightSum;
    float confidenceSum;
    bool hasSample;
};

GRISState makeGRISState()
{
    GRISState s = {};
    s.reservoir = makeEmptyPathReservoir();
    return s;
}

bool grisSelect(
    inout GRISState s,
    float weight,
    float confidence,
    bool valid,
    inout SampleGenerator sg
)
{

    if (isfinite(confidence))
        s.confidenceSum += max(confidence, 0.f);

    if (!valid || !(weight > 0.f) || !isfinite(weight))
        return false;

    s.weightSum += weight;

    const float p = s.weightSum;
    const bool selected = sampleNext1D(sg) * p < weight;
    s.hasSample = s.hasSample || selected;
    return selected;
}

void grisUpdate(
    inout GRISState s,
    PathSample sample,
    float weight,
    float confidence,
    inout SampleGenerator sg
)
{
    if (grisSelect(s, weight, confidence, isValidPath(sample), sg))
        s.reservoir.sample = sample;
}

PathReservoir finalizeGRIS(GRISState s)
{
    PathReservoir r = s.reservoir;
    r.confidence = s.confidenceSum;
    if (!s.hasSample || !(r.sample.target > 0.f) || !isValidPath(r.sample))
    {
        PathReservoir empty = makeEmptyPathReservoir();
        empty.confidence = r.confidence;
        return empty;
    }

    r.ucw = s.weightSum / r.sample.target;
    if (!isfinite(r.ucw) || r.ucw < 0.f)
    {
        PathReservoir empty = makeEmptyPathReservoir();
        empty.confidence = r.confidence;
        return empty;
    }
    return r;
}
