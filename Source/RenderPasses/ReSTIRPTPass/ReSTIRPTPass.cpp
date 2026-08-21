/***************************************************************************
 # Copyright (c) 2015-24, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "ReSTIRPTPass.h"
#include "RenderGraph/RenderPassHelpers.h"
#include "RenderGraph/RenderPassStandardFlags.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EmissiveUniformSampler.h"
#include "Utils/Sampling/SampleGeneratorType.slangh"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <string>
#include <vector>

namespace
{
const std::string kColor = "color";
const std::string kVBuffer = "vbuffer";
const std::string kDepth = "depth";
const std::string kMVec = "mvec";
const std::string kShaderRoot = "RenderPasses/ReSTIRPTPass/";

const Falcor::ChannelList kInputChannels = {
    {kVBuffer, "gVBuffer", "Visibility buffer in packed format"},
    {kMVec, "gMotionVector", "Primary current-to-previous motion (normalized)"},
    {kDepth, "gDepth", "Depth buffer (NDC)", false, ResourceFormat::R32Float},
};

const Falcor::ChannelList kOutputChannels = {
    {kColor, "gOutputColor", "HDR color", false, ResourceFormat::RGBA32Float},
};

const uint32_t kPairingSizes[] = {254, 230, 210};
const uint32_t kPairingSetCount = 3;
const float kPairingSigma = 16.f;

uint32_t nSigmaIterations(float sigma)
{
    const float s = std::max(sigma, 0.8f);
    return (uint32_t)std::floor(s * s * 0.5f + 1.46f / s + 1.76f / (s * s) + 0.656f / (s * s * s) + 0.5f);
}

void shuffle2x2(std::vector<int32_t>& links, uint32_t width, uint32_t height, int32_t offset, std::mt19937& rng)
{
    for (uint32_t y = 0; y < height; y += 2)
    {
        for (uint32_t x = 0; x < width; x += 2)
        {
            int32_t idx[4];
            int32_t vals[4];
            for (int i = 0; i < 4; ++i)
            {
                const int32_t px = int32_t((x + offset + (i & 1)) % width);
                const int32_t py = int32_t((y + offset + ((i >> 1) & 1)) % height);
                idx[i] = py * int32_t(width) + px;
                vals[i] = links[idx[i]];
            }
            std::shuffle(std::begin(vals), std::end(vals), rng);
            for (int i = 0; i < 4; ++i)
                links[idx[i]] = vals[i];
        }
    }
}

std::vector<int2> generatePairingTexture(uint32_t size, float sigma, uint32_t seed)
{
    const uint32_t n = size * size;
    std::vector<int32_t> links(n);
    for (uint32_t i = 0; i < n; ++i)
        links[i] = int32_t(i / 2);

    std::mt19937 rng(seed);
    const uint32_t iters = nSigmaIterations(sigma);
    for (uint32_t i = 0; i < iters; ++i)
        shuffle2x2(links, size, size, int32_t(i & 1), rng);

    const uint32_t pairCount = n / 2;
    std::vector<int32_t> first(pairCount, -1);
    std::vector<int32_t> second(pairCount, -1);
    for (uint32_t i = 0; i < n; ++i)
    {
        const int32_t li = links[i];
        if (first[li] < 0)
            first[li] = int32_t(i);
        else
            second[li] = int32_t(i);
    }

    std::vector<int2> offsets(n);
    const int32_t half = int32_t(size / 2);
    for (uint32_t i = 0; i < n; ++i)
    {
        const int32_t li = links[i];
        const int32_t partner = (first[li] == int32_t(i)) ? second[li] : first[li];
        int32_t dx = int32_t(partner % size) - int32_t(i % size);
        int32_t dy = int32_t(partner / size) - int32_t(i / size);
        if (dx > half)
            dx -= int32_t(size);
        if (dx < -half)
            dx += int32_t(size);
        if (dy > half)
            dy -= int32_t(size);
        if (dy < -half)
            dy += int32_t(size);
        offsets[i] = int2(dx, dy);
    }
    return offsets;
}
static_assert(sizeof(RestirParams) == 192, "RestirParams must match Params.slang");
static_assert(sizeof(PairingSetDesc) == 32, "PairingSetDesc must match Params.slang");
}

extern "C" FALCOR_API_EXPORT void registerPlugin(Falcor::PluginRegistry& registry)
{
    registry.registerClass<RenderPass, ReSTIRPTPass>();
}

ReSTIRPTPass::ReSTIRPTPass(ref<Device> pDevice, const Properties& props) : RenderPass(pDevice)
{
    parseProperties(props);
    mpSampleGenerator = SampleGenerator::create(mpDevice, SAMPLE_GENERATOR_TINY_UNIFORM);
    preparePairingTextures();
}

void ReSTIRPTPass::parseProperties(const Properties& props)
{
    for (const auto& [key, value] : props)
    {
        if (key == "maxPathLength")
            mParams.maxPathLength = value;
        else if (key == "maxDiffuseBounces")
            mParams.maxDiffuseBounces = value;
        else if (key == "maxSpecularBounces")
            mParams.maxSpecularBounces = value;
        else if (key == "maxTransmissionBounces")
            mParams.maxTransmissionBounces = value;
        else if (key == "enableDuplicationMap")
            mEnableDuplicationMap = value;
        else if (key == "enableRussianRoulette")
            mEnableRussianRoulette = value;
        else if (key == "enableVectorWeights")
            mEnableVectorWeights = value;
        else if (key == "enableDualMotion")
            mEnableDualMotion = value;
        else if (key == "enableForcedNEEReconnect")
            mEnableForcedNEE = value;
        else if (key == "enablePresampledLights")
            mEnablePresampledLights = value;
        else if (key == "enableTemporalReuse")
            mEnableTemporalReuse = value;
        else if (key == "enableSpatialReuse")
            mEnableSpatialReuse = value;
        else if (key == "enableNEE")
            mEnableNEE = value;
        else if (key == "enableMIS")
            mEnableMIS = value;
        else if (key == "enableBSDFSampling")
            mEnableBSDFSampling = value;
        else if (key == "disableCaustics")
            mDisableCaustics = value;
        else if (key == "useLightsInDielectricVolumes")
            mUseLightsInDielectricVolumes = value;
        else if (key == "disableDirectIllumination")
            mDisableDirectIllumination = value;
        else if (key == "adjustShadingNormals")
            mAdjustShadingNormals = value;
        else if (key == "useFixedSeed")
            mUseFixedSeed = value;
        else if (key == "fixedSeed")
            mFixedSeed = value;
        else if (key == "nestedMaterialSlots")
            mNestedMaterialSlots = value;
        else if (key == "sampleGenerator")
            mSampleGeneratorType = value;
        else if (key == "primaryNEECandidateCount")
            mParams.primaryNEECandidateCount = value;
        else if (key == "pairingSlotCount")
            mParams.pairingSlotCount = value;
        else if (key == "lightTileCount")
            mParams.lightTileCount = value;
        else if (key == "lightsPerTile")
            mParams.lightsPerTile = value;
        else if (key == "initialRussianRouletteStart")
            mParams.initialRussianRouletteStart = value;
        else if (key == "initialRussianRouletteMinSurvival")
            mParams.initialRussianRouletteMinSurvival = value;
        else if (key == "duplicationRadius")
            mParams.duplicationRadius = value;
        else if (key == "temporalDepthThreshold")
            mParams.temporalDepthThreshold = value;
        else if (key == "spatialNormalThreshold")
            mParams.spatialNormalThreshold = value;
        else if (key == "spatialDepthThreshold")
            mParams.spatialDepthThreshold = value;
        else if (key == "specularRoughnessThreshold")
            mParams.specularRoughnessThreshold = value;
        else if (key == "cDefault")
            mParams.cDefault = value;
        else if (key == "cMin")
            mParams.cMin = value;
        else if (key == "duplicationExponent")
            mParams.duplicationExponent = value;
        else if (key == "footprintScale")
            mParams.footprintScale = value;
        else if (key == "minReconnectionRoughness")
            mParams.minReconnectionRoughness = value;
        else if (key == "lodBias")
            mParams.lodBias = value;
        else if (key == "primaryLodMode")
        {
            const TexLODMode mode = value;
            mParams.primaryLodMode = (uint32_t)mode;
        }
        else
            logWarning("Unknown property '{}' in ReSTIRPTPass properties.", key);
    }
}

Properties ReSTIRPTPass::getProperties() const
{
    Properties props;
    props["maxPathLength"] = mParams.maxPathLength;
    props["maxDiffuseBounces"] = mParams.maxDiffuseBounces;
    props["maxSpecularBounces"] = mParams.maxSpecularBounces;
    props["maxTransmissionBounces"] = mParams.maxTransmissionBounces;
    props["enableDuplicationMap"] = mEnableDuplicationMap;
    props["enableRussianRoulette"] = mEnableRussianRoulette;
    props["enableVectorWeights"] = mEnableVectorWeights;
    props["enableDualMotion"] = mEnableDualMotion;
    props["enableForcedNEEReconnect"] = mEnableForcedNEE;
    props["enablePresampledLights"] = mEnablePresampledLights;
    props["enableTemporalReuse"] = mEnableTemporalReuse;
    props["enableSpatialReuse"] = mEnableSpatialReuse;
    props["enableNEE"] = mEnableNEE;
    props["enableMIS"] = mEnableMIS;
    props["enableBSDFSampling"] = mEnableBSDFSampling;
    props["disableCaustics"] = mDisableCaustics;
    props["useLightsInDielectricVolumes"] = mUseLightsInDielectricVolumes;
    props["disableDirectIllumination"] = mDisableDirectIllumination;
    props["adjustShadingNormals"] = mAdjustShadingNormals;
    props["useFixedSeed"] = mUseFixedSeed;
    props["fixedSeed"] = mFixedSeed;
    props["nestedMaterialSlots"] = mNestedMaterialSlots;
    props["sampleGenerator"] = mSampleGeneratorType;
    props["primaryNEECandidateCount"] = mParams.primaryNEECandidateCount;
    props["pairingSlotCount"] = mParams.pairingSlotCount;
    props["lightTileCount"] = mParams.lightTileCount;
    props["lightsPerTile"] = mParams.lightsPerTile;
    props["initialRussianRouletteStart"] = mParams.initialRussianRouletteStart;
    props["initialRussianRouletteMinSurvival"] = mParams.initialRussianRouletteMinSurvival;
    props["duplicationRadius"] = mParams.duplicationRadius;
    props["temporalDepthThreshold"] = mParams.temporalDepthThreshold;
    props["spatialNormalThreshold"] = mParams.spatialNormalThreshold;
    props["spatialDepthThreshold"] = mParams.spatialDepthThreshold;
    props["specularRoughnessThreshold"] = mParams.specularRoughnessThreshold;
    props["cDefault"] = mParams.cDefault;
    props["cMin"] = mParams.cMin;
    props["duplicationExponent"] = mParams.duplicationExponent;
    props["footprintScale"] = mParams.footprintScale;
    props["minReconnectionRoughness"] = mParams.minReconnectionRoughness;
    props["lodBias"] = mParams.lodBias;
    props["primaryLodMode"] = TexLODMode(mParams.primaryLodMode);
    return props;
}

RenderPassReflection ReSTIRPTPass::reflect(const CompileData& compileData)
{
    RenderPassReflection reflector;
    addRenderPassInputs(reflector, kInputChannels);
    addRenderPassOutputs(reflector, kOutputChannels);
    return reflector;
}

void ReSTIRPTPass::compile(RenderContext* pRenderContext, const CompileData& compileData)
{
    mFrameDim = compileData.defaultTexDims;
}

void ReSTIRPTPass::setScene(RenderContext* pRenderContext, const ref<Scene>& pScene)
{
    mpScene = pScene;
    mpEnvMapSampler.reset();
    mpEmissiveSampler.reset();
    mpPowerSampler.reset();
    mFrameIndex = 0;
    recreatePrograms();
    invalidateHistory();
}

void ReSTIRPTPass::invalidateHistory()
{
    mHistoryValid = false;
}

DefineList ReSTIRPTPass::getDefines() const
{
    DefineList defines;
    defines.add(mpSampleGenerator->getDefines());
    if (mpEmissiveSampler)
        defines.add(mpEmissiveSampler->getDefines());
    else
        defines.add("_EMISSIVE_LIGHT_SAMPLER_TYPE", std::to_string((uint32_t)EmissiveLightSamplerType::Null));

    defines.add("USE_ENV_LIGHT", "0");
    defines.add("USE_ANALYTIC_LIGHTS", "0");
    defines.add("USE_EMISSIVE_LIGHTS", "0");
    defines.add("INTERIOR_LIST_SLOT_COUNT", std::to_string(std::clamp(mNestedMaterialSlots, 2u, 4u)));

    if (mpScene)
    {
        defines.add(mpScene->getSceneDefines());
        defines.add("USE_ENV_LIGHT", mpScene->useEnvLight() ? "1" : "0");
        defines.add("USE_ANALYTIC_LIGHTS", mpScene->useAnalyticLights() ? "1" : "0");
        defines.add("USE_EMISSIVE_LIGHTS", mpScene->useEmissiveLights() ? "1" : "0");
    }
    return defines;
}

void ReSTIRPTPass::recreatePrograms()
{
    mpReflectTypes = nullptr;
    mpResetFrame = nullptr;
    mpBuildDispatchArgs = nullptr;
    mpPresampleLights = nullptr;
    mpBuildPairing = nullptr;
    mpBuildDualMotion = nullptr;
    mpInitialSample = nullptr;
    mpTemporalShift = nullptr;
    mpTemporalReplay = nullptr;
    mpTemporalReuse = nullptr;
    mpSpatialShift = nullptr;
    mpSpatialReplay = nullptr;
    mpSpatialResample = nullptr;
    mpResolve = nullptr;
    mpDuplicationMapPass = nullptr;
    mRecompile = true;
}

static ref<ComputePass> createComputeStage(ref<Device> pDevice, const ref<Scene>& pScene, const std::string& path, const DefineList& defines, bool withScene)
{
    ProgramDesc desc;
    if (withScene && pScene)
    {
        desc.addShaderModules(pScene->getShaderModules());
        desc.addShaderLibrary(path).csEntry("main");
        desc.addTypeConformances(pScene->getTypeConformances());
    }
    else
    {
        desc.addShaderLibrary(path).csEntry("main");
    }
    return ComputePass::create(pDevice, desc, defines, true);
}

void ReSTIRPTPass::prepareLighting(RenderContext* pRenderContext)
{
    if (!mpScene)
        return;

    if (mpScene->useEnvLight())
    {
        if (!mpEnvMapSampler)
        {
            mpEnvMapSampler = std::make_unique<EnvMapSampler>(mpDevice, mpScene->getEnvMap());
            mRecompile = true;
        }
    }
    else if (mpEnvMapSampler)
    {
        mpEnvMapSampler.reset();
        mRecompile = true;
    }

    if (mpScene->getRenderSettings().useEmissiveLights)
        mpScene->getILightCollection(pRenderContext);

    if (mpScene->useEmissiveLights())
    {
        auto pLights = mpScene->getILightCollection(pRenderContext);
        if (!mpEmissiveSampler)
        {
            mpEmissiveSampler = std::make_unique<LightBVHSampler>(pRenderContext, pLights);
            mRecompile = true;
        }

        mpEmissiveSampler->update(pRenderContext, pLights);

        const auto& tris = pLights->getMeshLightTriangles(pRenderContext);
        if (!tris.empty())
        {
            if (!mpPowerSampler)
            {
                mpPowerSampler = std::make_unique<EmissivePowerSampler>(pRenderContext, pLights);
                mRecompile = true;
            }
            mpPowerSampler->update(pRenderContext, pLights);

            double fluxSum = 0.0;
            for (const auto& tri : tris)
                fluxSum += double(tri.flux);
            mParams.lightPowerInvWeightsSum = fluxSum > 0.0 ? float(1.0 / fluxSum) : 0.f;
        }
        else if (mpPowerSampler)
        {
            mpPowerSampler.reset();
            mParams.lightPowerInvWeightsSum = 0.f;
            mRecompile = true;
        }
    }
    else if (mpEmissiveSampler || mpPowerSampler)
    {
        mpEmissiveSampler.reset();
        mpPowerSampler.reset();
        mParams.lightPowerInvWeightsSum = 0.f;
        mRecompile = true;
    }
}

void ReSTIRPTPass::fillPairingDescs(PairingSetDesc* descs, bool randomizeTransforms) const
{
    std::mt19937 rng(mParams.frameIndex * 747796405u + 2891336453u);
    std::uniform_int_distribution<int> bit(0, 1);
    std::uniform_int_distribution<int> off(-64, 64);
    uint32_t offset = 0;
    for (uint32_t i = 0; i < kPairingSetCount; ++i)
    {
        descs[i] = {};
        descs[i].baseOffset = offset;
        descs[i].dimX = kPairingSizes[i];
        descs[i].dimY = kPairingSizes[i];
        if (randomizeTransforms)
        {
            descs[i].transform = uint32_t(bit(rng) | (bit(rng) << 1) | (bit(rng) << 2));
            descs[i].frameOffsetX = off(rng);
            descs[i].frameOffsetY = off(rng);
        }
        offset += kPairingSizes[i] * kPairingSizes[i];
    }
}

void ReSTIRPTPass::preparePairingTextures()
{
    std::vector<int2> packed;
    PairingSetDesc descs[kPairingSetCount];
    fillPairingDescs(descs, false);
    for (uint32_t i = 0; i < kPairingSetCount; ++i)
    {
        auto tex = generatePairingTexture(kPairingSizes[i], kPairingSigma, 0xC0FFEEu + i * 17u);
        packed.insert(packed.end(), tex.begin(), tex.end());
    }

    mpPairingBaseOffsets = mpDevice->createStructuredBuffer(
        sizeof(int2),
        uint32_t(packed.size()),
        ResourceBindFlags::ShaderResource,
        MemoryType::DeviceLocal,
        packed.data()
    );
    mpPairingSets = mpDevice->createStructuredBuffer(
        sizeof(PairingSetDesc),
        kPairingSetCount,
        ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
        MemoryType::DeviceLocal,
        descs
    );
}

void ReSTIRPTPass::updatePairingTransforms()
{
    if (!mpPairingSets)
        return;

    PairingSetDesc descs[kPairingSetCount];
    fillPairingDescs(descs, true);
    mpPairingSets->setBlob(descs, 0, sizeof(descs));
}

void ReSTIRPTPass::prepareResources(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pVBuffer = renderData.getTexture(kVBuffer);
    FALCOR_ASSERT(pVBuffer);
    const uint2 dim = uint2(pVBuffer->getWidth(), pVBuffer->getHeight());
    if (any(dim != mFrameDim))
    {
        mFrameDim = dim;
        invalidateHistory();
    }

    mParams.pairingSlotCount = std::clamp(mParams.pairingSlotCount, 1u, kPairingSetCount);
    mParams.lightTileCount = std::max(mParams.lightTileCount, 1u);
    mParams.lightsPerTile = std::max(mParams.lightsPerTile, 1u);

    const uint32_t pixelCount = mFrameDim.x * mFrameDim.y;
    mParams.frameDim = mFrameDim;
    mParams.temporalReplayCapacity = pixelCount;

    mParams.spatialReplayCapacity = mParams.pairingSlotCount * ((pixelCount + 1u) / 2u);
    const uint32_t lightTileEntries = mParams.lightTileCount * mParams.lightsPerTile;

    if (!mpReflectTypes)
        mpReflectTypes = createComputeStage(mpDevice, mpScene, kShaderRoot + "Frame/ReflectTypes.cs.slang", getDefines(), false);

    auto var = mpReflectTypes->getRootVar()["gPass"];
    auto makeBuf = [&](const char* name, uint32_t count)
    {
        return mpDevice->createStructuredBuffer(
            var[name],
            std::max(count, 1u),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess,
            MemoryType::DeviceLocal,
            nullptr,
            false
        );
    };

    const bool needReservoirs = !mpCurrentReservoirs || mpCurrentReservoirs->getElementCount() < pixelCount;
    const bool needPairing = mAllocatedPairingSlots != mParams.pairingSlotCount || needReservoirs;
    const bool needTiles = mAllocatedLightTileEntries != lightTileEntries || !mpLightTiles;
    if (needReservoirs || needPairing || needTiles)
    {
        mpCurrentReservoirs = makeBuf("pathReservoirs", pixelCount);
        mpHistoryReservoirs = makeBuf("pathReservoirs", pixelCount);
        mpBirthIdentities = makeBuf("birthIdentities", pixelCount);
        mpTemporalShifts = makeBuf("temporalShifts", pixelCount);
        mpSpatialShifts = makeBuf("spatialShifts", pixelCount * mParams.pairingSlotCount);
        mpTemporalReplayQueue = makeBuf("replayQueue", mParams.temporalReplayCapacity);
        mpSpatialReplayQueue = makeBuf("replayQueue", mParams.spatialReplayCapacity);
        mpLightTiles = makeBuf("lightTiles", lightTileEntries);
        mpPairingPartners = makeBuf("pairingPartners", pixelCount * mParams.pairingSlotCount);
        mpCounters = makeBuf("counters", 16);
        mAllocatedPairingSlots = mParams.pairingSlotCount;
        mAllocatedLightTileEntries = lightTileEntries;
        invalidateHistory();
    }

    if (!mpDispatchArgs)
    {
        const uint32_t initArgs[] = {1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0};
        mpDispatchArgs = mpDevice->createBuffer(
            sizeof(initArgs),
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::IndirectArg,
            MemoryType::DeviceLocal,
            initArgs
        );
    }

    auto makeTex = [&](ResourceFormat format)
    {
        return mpDevice->createTexture2D(
            mFrameDim.x,
            mFrameDim.y,
            format,
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
    };

    if (!mpDuplicationMap || mpDuplicationMap->getWidth() != mFrameDim.x || mpDuplicationMap->getHeight() != mFrameDim.y)
    {
        mpDuplicationMap = makeTex(ResourceFormat::R32Float);
        mpDualMotion = makeTex(ResourceFormat::RG32Float);
        mpMotionValidity = makeTex(ResourceFormat::R32Uint);
        mpPrevVBuffer = mpDevice->createTexture2D(
            mFrameDim.x,
            mFrameDim.y,
            pVBuffer->getFormat(),
            1,
            1,
            nullptr,
            ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess
        );
        mpPrevDepth = makeTex(ResourceFormat::R32Float);
        invalidateHistory();
    }
}

void ReSTIRPTPass::bindScene(RenderContext* pRenderContext, const ref<ComputePass>& pPass, bool raytracing)
{
    if (!mpScene || !pPass)
        return;
    auto root = pPass->getRootVar();
    if (raytracing)
        mpScene->bindShaderDataForRaytracing(pRenderContext, root["gScene"]);
    else
        mpScene->bindShaderData(root["gScene"]);
}

void ReSTIRPTPass::bindParams(const ShaderVar& var)
{
    var["params"].setBlob(mParams);
}

void ReSTIRPTPass::bindLights(const ShaderVar& var)
{
    if (mpEnvMapSampler && var.hasMember("envMapSampler"))
        mpEnvMapSampler->bindShaderData(var["envMapSampler"]);
    if (mpEmissiveSampler && var.hasMember("emissiveSampler"))
        mpEmissiveSampler->bindShaderData(var["emissiveSampler"]);
    if (var.hasMember("lightTiles"))
        var["lightTiles"] = mpLightTiles;
}

ShaderVar ReSTIRPTPass::bindComputePass(
    RenderContext* pRenderContext,
    const ref<ComputePass>& pPass,
    bool bindSceneData,
    bool raytracing,
    bool bindLightData
)
{
    if (bindSceneData)
        bindScene(pRenderContext, pPass, raytracing);
    auto var = pPass->getRootVar()["gPass"];
    bindParams(var);
    if (bindLightData)
        bindLights(var);
    return var;
}

void ReSTIRPTPass::dispatchPixels(RenderContext* pRenderContext, const ref<ComputePass>& pPass, uint32_t z)
{
    pPass->execute(pRenderContext, mFrameDim.x, mFrameDim.y, z);
}

void ReSTIRPTPass::dispatchCount(RenderContext* pRenderContext, const ref<ComputePass>& pPass, uint32_t count)
{
    pPass->execute(pRenderContext, count, 1, 1);
}

void ReSTIRPTPass::buildDispatchArgs(
    RenderContext* pRenderContext,
    CounterIndex counterIndex,
    uint32_t capacity,
    uint32_t groupSize,
    uint32_t argsOffset,
    uint32_t workMultiplier
)
{
    auto var = mpBuildDispatchArgs->getRootVar()["gPass"];
    var["counterIndex"] = (uint32_t)counterIndex;
    var["queueCapacity"] = capacity;
    var["consumerGroupSize"] = groupSize;
    var["argsByteOffset"] = argsOffset;
    var["workMultiplier"] = workMultiplier;
    var["counters"] = mpCounters;
    var["dispatchArgs"] = mpDispatchArgs;
    pRenderContext->uavBarrier(mpCounters.get());
    mpBuildDispatchArgs->execute(pRenderContext, 1, 1, 1);
    pRenderContext->uavBarrier(mpDispatchArgs.get());
}

void ReSTIRPTPass::execute(RenderContext* pRenderContext, const RenderData& renderData)
{
    const auto& pOutput = renderData.getTexture(kColor);
    if (!mpScene)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));
        return;
    }

    const auto sceneUpdates = mpScene->getUpdates();
    if (is_set(sceneUpdates, IScene::UpdateFlags::RecompileNeeded) ||
        is_set(sceneUpdates, IScene::UpdateFlags::GeometryChanged) ||
        is_set(sceneUpdates, IScene::UpdateFlags::SDFGridConfigChanged))
    {
        recreatePrograms();
    }

    if (is_set(sceneUpdates, IScene::UpdateFlags::EnvMapChanged) ||
        is_set(sceneUpdates, IScene::UpdateFlags::RenderSettingsChanged))
    {
        mpEnvMapSampler.reset();
        mRecompile = true;
    }
    if (is_set(sceneUpdates, IScene::UpdateFlags::LightCollectionChanged) ||
        is_set(sceneUpdates, IScene::UpdateFlags::RenderSettingsChanged))
    {
        mpEmissiveSampler.reset();
        mpPowerSampler.reset();
        mRecompile = true;
    }

    const auto reprojectableUpdates = IScene::UpdateFlags::CameraMoved |
        IScene::UpdateFlags::CameraPropertiesChanged | IScene::UpdateFlags::GeometryMoved |
        IScene::UpdateFlags::LightsMoved | IScene::UpdateFlags::SceneGraphChanged |
        IScene::UpdateFlags::CurvesMoved | IScene::UpdateFlags::CustomPrimitivesMoved |
        IScene::UpdateFlags::GridVolumesMoved | IScene::UpdateFlags::MeshesChanged;
    if ((sceneUpdates & ~reprojectableUpdates) != IScene::UpdateFlags::None)
        invalidateHistory();
    if (mOptionsChanged)
    {
        invalidateHistory();
        mOptionsChanged = false;
    }

    prepareLighting(pRenderContext);

    const auto& pVBuffer = renderData.getTexture(kVBuffer);
    const auto& pMVec = renderData.getTexture(kMVec);
    const auto& pDepth = renderData.getTexture(kDepth);
    if (!pVBuffer || !pMVec || !pDepth)
    {
        pRenderContext->clearUAV(pOutput->getUAV().get(), float4(0.f));
        logWarning("ReSTIRPTPass: missing required vbuffer, mvec, or NDC depth input.");
        invalidateHistory();
        return;
    }

    prepareResources(pRenderContext, renderData);

    const auto defines = getDefines();
    auto ensure = [&](ref<ComputePass>& pass, const std::string& relativePath, bool withScene)
    {
        if (!pass)
            pass = createComputeStage(mpDevice, mpScene, kShaderRoot + relativePath, defines, withScene);
        else if (mRecompile)
        {
            pass->getProgram()->setDefines(defines);
            pass->setVars(nullptr);
        }
    };

    ensure(mpResetFrame, "Frame/ResetFrame.cs.slang", false);
    ensure(mpBuildDispatchArgs, "Frame/BuildDispatchArgs.cs.slang", false);
    ensure(mpPresampleLights, "Frame/PresampleLights.cs.slang", true);
    ensure(mpBuildPairing, "Spatial/BuildPairing.cs.slang", false);
    ensure(mpBuildDualMotion, "Frame/BuildDualMotion.cs.slang", true);
    ensure(mpInitialSample, "Path/InitialSample.cs.slang", true);
    ensure(mpTemporalShift, "Temporal/TemporalShift.cs.slang", true);
    ensure(mpTemporalReplay, "Temporal/TemporalReplay.cs.slang", true);
    ensure(mpTemporalReuse, "Temporal/TemporalReuse.cs.slang", false);
    ensure(mpSpatialShift, "Spatial/SpatialShift.cs.slang", true);
    ensure(mpSpatialReplay, "Spatial/SpatialReplay.cs.slang", true);
    ensure(mpSpatialResample, "Spatial/SpatialResample.cs.slang", false);
    ensure(mpResolve, "Frame/Resolve.cs.slang", false);
    ensure(mpDuplicationMapPass, "Frame/DuplicationMap.cs.slang", false);
    mRecompile = false;

    mParams.frameDim = mFrameDim;
    mParams.frameIndex = mUseFixedSeed ? mFixedSeed : mFrameIndex++;
    mParams.flags = 0;
    if (mEnableDualMotion)
        mParams.flags |= kEnableDualMotion;
    if (mHistoryValid && mEnableTemporalReuse)
        mParams.flags |= kHistoryValid;

    mParams.featureFlags = 0;
    if (mEnablePresampledLights)
        mParams.featureFlags |= kEnablePresampledLights;
    if (mEnableDuplicationMap)
        mParams.featureFlags |= kEnableDuplicationMap;
    if (mEnableForcedNEE)
        mParams.featureFlags |= kEnableForcedNEEReconnect;
    if (mEnableRussianRoulette)
        mParams.featureFlags |= kEnableInitialRussianRoulette;
    if (mEnableVectorWeights)
        mParams.featureFlags |= kEnableVectorWeights;
    if (mEnableNEE)
        mParams.featureFlags |= kEnableNEE;
    if (mEnableMIS)
        mParams.featureFlags |= kEnableMIS;
    if (mEnableBSDFSampling)
        mParams.featureFlags |= kEnableBSDFSampling;
    if (mDisableCaustics)
        mParams.featureFlags |= kDisableCaustics;
    if (mUseLightsInDielectricVolumes)
        mParams.featureFlags |= kUseLightsInDielectricVolumes;
    if (mDisableDirectIllumination)
        mParams.featureFlags |= kDisableDirectIllumination;
    if (mAdjustShadingNormals)
        mParams.featureFlags |= kAdjustShadingNormals;

    mParams.pairingSlotCount = std::clamp(mParams.pairingSlotCount, 1u, kPairingSetCount);
    mParams.maxPathLength = std::clamp(mParams.maxPathLength, 2u, 8u);
    mParams.maxDiffuseBounces = std::min(mParams.maxDiffuseBounces, 8u);
    mParams.maxSpecularBounces = std::min(mParams.maxSpecularBounces, 8u);
    mParams.maxTransmissionBounces = std::min(mParams.maxTransmissionBounces, 8u);
    if (mParams.cMin > mParams.cDefault)
        std::swap(mParams.cMin, mParams.cDefault);
    mParams.previousInvViewProj = inverse(mpScene->getCamera()->getData().prevViewProjMatNoJitter);
    updatePairingTransforms();

    {
        FALCOR_PROFILE(pRenderContext, "ResetFrame");
        auto var = mpResetFrame->getRootVar()["gPass"];
        var["counters"] = mpCounters;
        dispatchCount(pRenderContext, mpResetFrame, 1);
        pRenderContext->uavBarrier(mpCounters.get());
    }

    if (mEnablePresampledLights)
    {
        FALCOR_PROFILE(pRenderContext, "PresampleLights");
        auto var = bindComputePass(pRenderContext, mpPresampleLights, true, false, false);
        if (mpPowerSampler && var.hasMember("powerSampler"))
            mpPowerSampler->bindShaderData(var["powerSampler"]);
        var["lightTiles"] = mpLightTiles;
        dispatchCount(pRenderContext, mpPresampleLights, mParams.lightTileCount * mParams.lightsPerTile);
    }

    if (mEnableSpatialReuse)
    {
        FALCOR_PROFILE(pRenderContext, "BuildPairing");
        auto var = bindComputePass(pRenderContext, mpBuildPairing, false, false, false);
        var["pairingBaseOffsets"] = mpPairingBaseOffsets;
        var["pairingSets"] = mpPairingSets;
        var["pairingPartners"] = mpPairingPartners;
        dispatchPixels(pRenderContext, mpBuildPairing, mParams.pairingSlotCount);
    }

    if (mEnableTemporalReuse && mEnableDualMotion && mHistoryValid)
    {
        FALCOR_PROFILE(pRenderContext, "BuildDualMotion");
        auto var = bindComputePass(pRenderContext, mpBuildDualMotion, true, false, false);
        var["primaryMotion"] = pMVec;
        var["previousDepth"] = mpPrevDepth;
        var["dualMotion"] = mpDualMotion;
        var["motionValidity"] = mpMotionValidity;
        dispatchPixels(pRenderContext, mpBuildDualMotion);
    }

    {
        FALCOR_PROFILE(pRenderContext, "InitialSample");
        auto var = bindComputePass(pRenderContext, mpInitialSample, true, true, true);
        var["vbuffer"] = pVBuffer;
        var["currentReservoirs"] = mpCurrentReservoirs;
        var["outputColor"] = pOutput;
        dispatchPixels(pRenderContext, mpInitialSample);
        pRenderContext->uavBarrier(pOutput.get());
    }

    if (mEnableTemporalReuse)
    {
        {
            FALCOR_PROFILE(pRenderContext, "TemporalShift");
            auto var = bindComputePass(pRenderContext, mpTemporalShift, true, true, true);
            var["primaryMotion"] = pMVec;
            var["dualMotion"] = mpDualMotion;
            var["motionValidity"] = mpMotionValidity;
            var["currentVBuffer"] = pVBuffer;
            var["previousVBuffer"] = mpPrevVBuffer;
            var["currentReservoirs"] = mpCurrentReservoirs;
            var["historyReservoirs"] = mpHistoryReservoirs;
            var["shiftRecords"] = mpTemporalShifts;
            var["replayQueue"] = mpTemporalReplayQueue;
            var["counters"] = mpCounters;
            dispatchPixels(pRenderContext, mpTemporalShift);
            pRenderContext->uavBarrier(mpCounters.get());
            pRenderContext->uavBarrier(mpTemporalReplayQueue.get());
            pRenderContext->uavBarrier(mpTemporalShifts.get());
        }

        buildDispatchArgs(
            pRenderContext,
            CounterIndex::TemporalReplay,
            mParams.temporalReplayCapacity,
            64,
            kTemporalArgsOffset
        );
        {
            FALCOR_PROFILE(pRenderContext, "TemporalReplay");
            auto var = bindComputePass(pRenderContext, mpTemporalReplay, true, true, true);
            var["currentReservoirs"] = mpCurrentReservoirs;
            var["historyReservoirs"] = mpHistoryReservoirs;
            var["currentVBuffer"] = pVBuffer;
            var["previousVBuffer"] = mpPrevVBuffer;
            var["replayQueue"] = mpTemporalReplayQueue;
            var["shiftRecords"] = mpTemporalShifts;
            var["counters"] = mpCounters;
            mpTemporalReplay->executeIndirect(pRenderContext, mpDispatchArgs.get(), kTemporalArgsOffset);
            pRenderContext->uavBarrier(mpTemporalShifts.get());
        }

        {
            FALCOR_PROFILE(pRenderContext, "TemporalReuse");
            auto var = bindComputePass(pRenderContext, mpTemporalReuse, false, false, false);
            var["currentReservoirs"] = mpCurrentReservoirs;
            var["historyReservoirs"] = mpHistoryReservoirs;
            var["shiftRecords"] = mpTemporalShifts;
            var["previousDuplicationMap"] = mpDuplicationMap;
            dispatchPixels(pRenderContext, mpTemporalReuse);
            pRenderContext->uavBarrier(mpCurrentReservoirs.get());
        }
    }

    if (mEnableSpatialReuse)
    {
        {
            FALCOR_PROFILE(pRenderContext, "SpatialShift");
            auto var = bindComputePass(pRenderContext, mpSpatialShift, true, true, true);
            var["currentReservoirs"] = mpCurrentReservoirs;
            var["pairingPartners"] = mpPairingPartners;
            var["vbuffer"] = pVBuffer;
            var["spatialShifts"] = mpSpatialShifts;
            var["replayQueue"] = mpSpatialReplayQueue;
            var["counters"] = mpCounters;
            dispatchPixels(pRenderContext, mpSpatialShift, mParams.pairingSlotCount);
            pRenderContext->uavBarrier(mpCounters.get());
            pRenderContext->uavBarrier(mpSpatialReplayQueue.get());
            pRenderContext->uavBarrier(mpSpatialShifts.get());
        }

        buildDispatchArgs(
            pRenderContext,
            CounterIndex::SpatialReplay,
            mParams.spatialReplayCapacity,
            64,
            kSpatialArgsOffset,
            2
        );
        {
            FALCOR_PROFILE(pRenderContext, "SpatialReplay");
            auto var = bindComputePass(pRenderContext, mpSpatialReplay, true, true, true);
            var["currentReservoirs"] = mpCurrentReservoirs;
            var["vbuffer"] = pVBuffer;
            var["replayQueue"] = mpSpatialReplayQueue;
            var["spatialShifts"] = mpSpatialShifts;
            var["counters"] = mpCounters;
            mpSpatialReplay->executeIndirect(pRenderContext, mpDispatchArgs.get(), kSpatialArgsOffset);
            pRenderContext->uavBarrier(mpSpatialShifts.get());
        }

        {
            FALCOR_PROFILE(pRenderContext, "SpatialResample");
            auto var = bindComputePass(pRenderContext, mpSpatialResample, false, false, false);
            var["currentReservoirs"] = mpCurrentReservoirs;
            var["pairingPartners"] = mpPairingPartners;
            var["spatialShifts"] = mpSpatialShifts;
            var["historyReservoirs"] = mpHistoryReservoirs;
            var["birthIdentities"] = mpBirthIdentities;
            var["outputColor"] = pOutput;
            dispatchPixels(pRenderContext, mpSpatialResample);
            pRenderContext->uavBarrier(mpHistoryReservoirs.get());
            pRenderContext->uavBarrier(mpBirthIdentities.get());
            pRenderContext->uavBarrier(pOutput.get());
        }
    }
    else
    {
        FALCOR_PROFILE(pRenderContext, "Resolve");
        auto var = bindComputePass(pRenderContext, mpResolve, false, false, false);
        var["currentReservoirs"] = mpCurrentReservoirs;
        var["historyReservoirs"] = mpHistoryReservoirs;
        var["birthIdentities"] = mpBirthIdentities;
        var["outputColor"] = pOutput;
        dispatchPixels(pRenderContext, mpResolve);
        pRenderContext->uavBarrier(mpHistoryReservoirs.get());
        pRenderContext->uavBarrier(mpBirthIdentities.get());
        pRenderContext->uavBarrier(pOutput.get());
    }

    if (mEnableDuplicationMap)
    {
        FALCOR_PROFILE(pRenderContext, "DuplicationMap");
        auto var = bindComputePass(pRenderContext, mpDuplicationMapPass, false, false, false);
        var["birthIdentities"] = mpBirthIdentities;
        var["duplicationMap"] = mpDuplicationMap;
        dispatchPixels(pRenderContext, mpDuplicationMapPass);
    }

    pRenderContext->copyResource(mpPrevVBuffer.get(), pVBuffer.get());
    if (mEnableDualMotion)
        pRenderContext->copyResource(mpPrevDepth.get(), pDepth.get());
    mHistoryValid = true;
}

void ReSTIRPTPass::renderUI(Gui::Widgets& widget)
{
    bool dirty = false;
    bool recompile = false;

    widget.text("ReSTIR PT Enhanced");
    widget.tooltip("Runtime options match Falcor PathTracer and original ReSTIR PT where the algorithm supports them.");

    if (auto group = widget.group("Path tracing", true))
    {
        if (widget.var("Max path length", mParams.maxPathLength, 2u, 8u))
        {
            mParams.maxDiffuseBounces = std::min(mParams.maxDiffuseBounces, mParams.maxPathLength);
            mParams.maxSpecularBounces = std::min(mParams.maxSpecularBounces, mParams.maxPathLength);
            mParams.maxTransmissionBounces = std::min(mParams.maxTransmissionBounces, mParams.maxPathLength);
            dirty = true;
        }
        widget.tooltip("Maximum camera-path vertices after the camera (x1..x8). Glass needs at least 3.");

        dirty |= widget.var("Max diffuse bounces", mParams.maxDiffuseBounces, 0u, 8u);
        widget.tooltip("0 = no diffuse continuation (direct only for diffuse).");
        dirty |= widget.var("Max specular bounces", mParams.maxSpecularBounces, 0u, 8u);
        dirty |= widget.var("Max transmission bounces", mParams.maxTransmissionBounces, 0u, 8u);
        widget.tooltip("0 = no refraction/transmission. Raise this for glass.");

        dirty |= widget.checkbox("BSDF importance sampling", mEnableBSDFSampling);
        widget.tooltip("If disabled, cosine-weighted hemisphere sampling is used (debug).");

        dirty |= widget.checkbox("Russian roulette", mEnableRussianRoulette);
        if (mEnableRussianRoulette)
        {
            dirty |= widget.var("RR start bounce", mParams.initialRussianRouletteStart, 1u, 8u);
            dirty |= widget.var("RR min survival", mParams.initialRussianRouletteMinSurvival, 0.f, 1.f);
        }

        dirty |= widget.checkbox("Disable direct illumination", mDisableDirectIllumination);
        widget.tooltip("Skip next-event estimation on the primary hit. BSDF-sampled lights still count.");
        dirty |= widget.checkbox("Disable caustics", mDisableCaustics);
        widget.tooltip("After a diffuse bounce, only sample diffuse lobes (no specular/glass caustics).");
        dirty |= widget.var("Specular roughness threshold", mParams.specularRoughnessThreshold, 0.f, 1.f);
        widget.tooltip("Events rougher than this are classified as diffuse (caustics / bounce limits).");

        if (widget.dropdown("Sample generator", SampleGenerator::getGuiDropdownList(), mSampleGeneratorType))
        {
            mpSampleGenerator = SampleGenerator::create(mpDevice, mSampleGeneratorType);
            recompile = true;
            dirty = true;
        }
    }

    if (auto group = widget.group("Lighting", true))
    {
        dirty |= widget.checkbox("Next-event estimation (NEE)", mEnableNEE);
        widget.tooltip("Shadowed light samples at each non-delta path vertex.");
        if (mEnableNEE)
        {
            dirty |= widget.checkbox("Multiple importance sampling (MIS)", mEnableMIS);
            dirty |= widget.var("Primary NEE candidates", mParams.primaryNEECandidateCount, 1u, 32u);
            widget.tooltip("N at the primary hit. Later vertices use max(1, N / bounce^2).");
            dirty |= widget.checkbox("Presampled light tiles", mEnablePresampledLights);
            if (mEnablePresampledLights)
            {
                dirty |= widget.var("Light tile count", mParams.lightTileCount, 1u, 512u);
                dirty |= widget.var("Lights per tile", mParams.lightsPerTile, 1u, 4096u);
            }
        }
        dirty |= widget.checkbox("Lights inside dielectrics", mUseLightsInDielectricVolumes);
        widget.tooltip("Allow NEE/emission while the path is inside a transmissive volume.");

        if (mpEmissiveSampler)
        {
            if (auto samplerGroup = widget.group("Emissive sampler"))
            {
                if (mpEmissiveSampler->renderUI(samplerGroup))
                    dirty = true;
            }
        }
    }

    if (auto group = widget.group("ReSTIR reuse", true))
    {
        dirty |= widget.checkbox("Temporal reuse", mEnableTemporalReuse);
        dirty |= widget.checkbox("Spatial reuse", mEnableSpatialReuse);
        dirty |= widget.checkbox("Dual motion vectors", mEnableDualMotion);
        widget.tooltip(
            "Second temporal candidate after disocclusion."
        );
        dirty |= widget.checkbox("Vector shading weights", mEnableVectorWeights);
        dirty |= widget.checkbox("Forced NEE reconnection", mEnableForcedNEE);
        widget.tooltip("Reconnect NEE endpoints even when the footprint gate fails.");

        if (mEnableSpatialReuse)
        {
            dirty |= widget.var("Spatial pairs", mParams.pairingSlotCount, 1u, kPairingSetCount);
            widget.tooltip("How many reciprocal neighbor pairs (1–3). Same as ReSTIR PT neighbor count.");
            dirty |= widget.var("Normal similarity", mParams.spatialNormalThreshold, 0.f, 1.f);
            dirty |= widget.var("Depth similarity", mParams.spatialDepthThreshold, 0.f, 1.f);
        }
        if (mEnableTemporalReuse)
            dirty |= widget.var("Temporal depth similarity", mParams.temporalDepthThreshold, 0.f, 1.f);
    }

    if (auto group = widget.group("Reconnection / Enhanced"))
    {
        dirty |= widget.var("Footprint scale (c/100)", mParams.footprintScale, 0.0f, 0.01f, 0.00001f, false, "%.6f");
        widget.tooltip("Paper reconnection gate. 0 is clamped back to the default 0.0002.");
        dirty |= widget.var("Min reconnection alpha", mParams.minReconnectionRoughness, 0.0f, 1.f);
        dirty |= widget.checkbox("Duplication map", mEnableDuplicationMap);
        if (mEnableDuplicationMap)
        {
            dirty |= widget.var("c_default", mParams.cDefault, 1.f, 64.f);
            dirty |= widget.var("c_min", mParams.cMin, 1.f, 20.f);
            dirty |= widget.var("Duplication exponent", mParams.duplicationExponent, 0.01f, 4.f);
            dirty |= widget.var("Duplication radius", mParams.duplicationRadius, 1u, 16u);
        }
    }

    if (auto group = widget.group("Materials"))
    {
        dirty |= widget.checkbox("Adjust shading normals (secondary)", mAdjustShadingNormals);
        widget.tooltip("Same as PathTracer secondary-hit normal adjustment. Primary hits follow the VBuffer.");
        if (widget.var("Max nested materials", mNestedMaterialSlots, 2u, 4u))
        {
            recompile = true;
            dirty = true;
        }
        widget.tooltip("Interior-list size for nested glass / volumes. Changing this recompiles shaders.");
        {
            TexLODMode mode = TexLODMode(mParams.primaryLodMode);
            if (widget.dropdown("Primary LOD mode", mode))
            {
                mParams.primaryLodMode = (uint32_t)mode;
                dirty = true;
            }
            widget.tooltip("Mip0 matches PathTracer default. RayDiffs uses screen-space gradients at the primary hit.");
        }
        dirty |= widget.var("TexLOD bias", mParams.lodBias, -16.f, 16.f, 0.01f);
        widget.tooltip("LOD offset for secondary hits.");
    }

    if (auto group = widget.group("Debugging"))
    {
        dirty |= widget.checkbox("Use fixed seed", mUseFixedSeed);
        widget.tooltip("Repeat the same RNG every frame.");
        if (mUseFixedSeed)
            dirty |= widget.var("Seed", mFixedSeed);
        if (widget.button("Reset history"))
            invalidateHistory();
        widget.text("Frame index: " + std::to_string(mFrameIndex));
    }

    if (recompile)
        recreatePrograms();
    if (dirty)
        mOptionsChanged = true;
}
