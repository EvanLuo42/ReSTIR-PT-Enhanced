static const uint kInvalidIndex = 0xffffffffu;
static const uint kMaxPathVertices = 9u;

static const uint kMotionPrimaryValid = 1u << 0;
static const uint kMotionDualValid = 1u << 1;

static const uint kRecordValid = 1u << 0;
static const uint kRecordNeedsReplay = 1u << 1;
static const uint kRecordReplayFailed = 1u << 2;

static const uint kPathValid = 1u << 0;
static const uint kPathEndsWithNEE = 1u << 1;
static const uint kPathEndsOnEnvironment = 1u << 2;
static const uint kPathHasReconnectBSDF = 1u << 3;
static const uint kPathEndsOnAnalytic = 1u << 4;
static const uint kPathRisMShift = 8u;
static const uint kPathRisMMask = 0xffu << kPathRisMShift;

enum class TechniqueKind : uint
{
    Invalid,
    CameraHit,
    NEE,
};

struct PathSample
{
    uint techniqueKey;

    uint2 randomSeed;
    uint reconnectionVertex;
    uint flags;

    uint lightTriangleIndex;
    uint candidateID;

    uint2 sampledLobes;

    float proposalPdf;
    float target;
    float reconnectDensity;

    float3 incidentDirection;
    float3 contribution;
    float3 suffixThroughput;

    float3 reconnectionPosition;
    float3 reconnectionNormal;

    uint4 reconnectionHit;
};

struct PathReservoir
{
    PathSample sample;
    float ucw;
    float confidence;
};

struct BirthIdentity
{
    uint4 value;
};

struct ShiftResult
{
    float3 contribution;
    float target;

    float proposalPdf;
    float reconnectDensity;
    float jacobian;
    uint reconnectionVertex;
    uint flags;
};

struct TemporalShiftRecord
{
    uint historyPixel;
    ShiftResult currentToHistory;
    ShiftResult historyToCurrent;
};

struct ReplayWorkItem
{
    uint recordIndex;
    uint pixelA;
    uint pixelB;
    uint _padding;
};

struct LightTileEntry
{
    uint lightIndex;
    float selectionPdf;
    float2 sample;
};

PathReservoir makeEmptyPathReservoir()
{
    PathReservoir r = {};
    r.sample.reconnectionVertex = kInvalidIndex;
    return r;
}

ShiftResult makeInvalidShiftResult()
{
    ShiftResult result = {};
    result.reconnectionVertex = kInvalidIndex;
    return result;
}
