/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
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
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 # OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 # HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 # STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 # IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 # POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#pragma once
#include "Falcor.h"
#include "RenderGraph/RenderPass.h"
#include "Utils/Sampling/SampleGenerator.h"
#include "Rendering/Lights/EmissiveLightSampler.h"
#include "Rendering/Lights/EmissivePowerSampler.h"
#include "Rendering/Lights/EnvMapSampler.h"
#include "Rendering/Lights/LightBVHSampler.h"
#include "Rendering/Materials/TexLODTypes.slang"
#include "Utils/Sampling/SampleGeneratorType.slangh"

#include "Params.slang"

using namespace Falcor;

class ReSTIRPTPass : public RenderPass
{
public:
    FALCOR_PLUGIN_CLASS(ReSTIRPTPass, "ReSTIRPTPass", "ReSTIR PT Enhanced");

    static ref<ReSTIRPTPass> create(ref<Device> pDevice, const Properties& props)
    {
        return make_ref<ReSTIRPTPass>(pDevice, props);
    }

    ReSTIRPTPass(ref<Device> pDevice, const Properties& props);

    virtual Properties getProperties() const override;
    virtual RenderPassReflection reflect(const CompileData& compileData) override;
    virtual void compile(RenderContext* pRenderContext, const CompileData& compileData) override;
    virtual void execute(RenderContext* pRenderContext, const RenderData& renderData) override;
    virtual void renderUI(Gui::Widgets& widget) override;
    virtual void setScene(RenderContext* pRenderContext, const ref<Scene>& pScene) override;
    virtual bool onMouseEvent(const MouseEvent& mouseEvent) override { return false; }
    virtual bool onKeyEvent(const KeyboardEvent& keyEvent) override { return false; }

private:
    void parseProperties(const Properties& props);
    DefineList getDefines() const;
    void recreatePrograms();
    void prepareLighting(RenderContext* pRenderContext);
    void prepareResources(RenderContext* pRenderContext, const RenderData& renderData);
    void preparePairingTextures();
    void fillPairingDescs(PairingSetDesc* descs, bool randomizeTransforms) const;
    void updatePairingTransforms();
    void bindScene(RenderContext* pRenderContext, const ref<ComputePass>& pPass, bool raytracing);
    void bindParams(const ShaderVar& var);
    void bindLights(const ShaderVar& var);
    ShaderVar bindComputePass(
        RenderContext* pRenderContext,
        const ref<ComputePass>& pPass,
        bool bindSceneData,
        bool raytracing,
        bool bindLightData
    );
    void dispatchPixels(RenderContext* pRenderContext, const ref<ComputePass>& pPass, uint32_t z = 1);
    void dispatchCount(RenderContext* pRenderContext, const ref<ComputePass>& pPass, uint32_t count);
    void buildDispatchArgs(
        RenderContext* pRenderContext,
        CounterIndex counterIndex,
        uint32_t capacity,
        uint32_t groupSize,
        uint32_t argsOffset,
        uint32_t workMultiplier = 1
    );
    void invalidateHistory();

    ref<Scene> mpScene;
    ref<SampleGenerator> mpSampleGenerator;
    std::unique_ptr<EnvMapSampler> mpEnvMapSampler;
    std::unique_ptr<EmissiveLightSampler> mpEmissiveSampler;
    std::unique_ptr<EmissivePowerSampler> mpPowerSampler;

    ref<ComputePass> mpReflectTypes;
    ref<ComputePass> mpResetFrame;
    ref<ComputePass> mpBuildDispatchArgs;
    ref<ComputePass> mpPresampleLights;
    ref<ComputePass> mpBuildPairing;
    ref<ComputePass> mpBuildDualMotion;
    ref<ComputePass> mpInitialSample;
    ref<ComputePass> mpTemporalShift;
    ref<ComputePass> mpTemporalReplay;
    ref<ComputePass> mpTemporalReuse;
    ref<ComputePass> mpSpatialShift;
    ref<ComputePass> mpSpatialReplay;
    ref<ComputePass> mpSpatialResample;
    ref<ComputePass> mpResolve;
    ref<ComputePass> mpDuplicationMapPass;

    ref<Buffer> mpCurrentReservoirs;
    ref<Buffer> mpHistoryReservoirs;
    ref<Buffer> mpBirthIdentities;
    ref<Buffer> mpTemporalShifts;
    ref<Buffer> mpSpatialShifts;
    ref<Buffer> mpTemporalReplayQueue;
    ref<Buffer> mpSpatialReplayQueue;
    ref<Buffer> mpLightTiles;
    ref<Buffer> mpPairingBaseOffsets;
    ref<Buffer> mpPairingSets;
    ref<Buffer> mpPairingPartners;
    ref<Buffer> mpCounters;
    ref<Buffer> mpDispatchArgs;

    ref<Texture> mpDuplicationMap;
    ref<Texture> mpDualMotion;
    ref<Texture> mpMotionValidity;
    ref<Texture> mpPrevVBuffer;
    ref<Texture> mpPrevDepth;

    RestirParams mParams;
    uint2 mFrameDim = uint2(0);
    uint32_t mFrameIndex = 0;
    bool mHistoryValid = false;
    bool mOptionsChanged = false;
    bool mRecompile = true;

    bool mEnableDuplicationMap = true;
    bool mEnableRussianRoulette = true;
    bool mEnableVectorWeights = true;
    bool mEnableDualMotion = true;
    bool mEnableForcedNEE = true;
    bool mEnablePresampledLights = true;
    bool mEnableTemporalReuse = true;
    bool mEnableSpatialReuse = true;
    bool mEnableNEE = true;
    bool mEnableMIS = true;
    bool mEnableBSDFSampling = true;
    bool mDisableCaustics = false;
    bool mUseLightsInDielectricVolumes = false;
    bool mDisableDirectIllumination = false;
    bool mAdjustShadingNormals = false;
    bool mUseFixedSeed = false;
    uint32_t mFixedSeed = 1;
    uint32_t mNestedMaterialSlots = 2;
    uint32_t mSampleGeneratorType = SAMPLE_GENERATOR_TINY_UNIFORM;
    uint32_t mAllocatedPairingSlots = 0;
    uint32_t mAllocatedLightTileEntries = 0;

    static constexpr uint32_t kTemporalArgsOffset = 0;
    static constexpr uint32_t kSpatialArgsOffset = 16;
};
