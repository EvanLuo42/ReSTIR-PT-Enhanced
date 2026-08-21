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
#pragma once
#include "Utils/HostDeviceShared.slangh"

BEGIN_NAMESPACE_FALCOR

/** Frame flags stored in RestirParams.flags.
*/
static const uint kEnableDualMotion = 1u << 0;
static const uint kHistoryValid = 1u << 1;

/** Feature flags stored in RestirParams.featureFlags.
*/
static const uint kEnablePresampledLights = 1u << 0;
static const uint kEnableDuplicationMap = 1u << 1;
static const uint kEnableForcedNEEReconnect = 1u << 2;
static const uint kEnableInitialRussianRoulette = 1u << 3;
static const uint kEnableVectorWeights = 1u << 4;
static const uint kEnableNEE = 1u << 5;
static const uint kEnableMIS = 1u << 6;
static const uint kEnableBSDFSampling = 1u << 7;
static const uint kDisableCaustics = 1u << 8;
static const uint kUseLightsInDielectricVolumes = 1u << 9;
static const uint kDisableDirectIllumination = 1u << 10;
static const uint kAdjustShadingNormals = 1u << 11;

enum class CounterIndex : uint32_t
{
    TemporalReplay,
    SpatialReplay,
    QueueOverflow,
    Count,
};

struct PairingSetDesc
{
    uint baseOffset;
    uint dimX;
    uint dimY;
    uint transform;
    int frameOffsetX;
    int frameOffsetY;
    uint _padding0;
    uint _padding1;
};

/** ReSTIR PT parameters shared between host and device.
    Layout follows HLSL packing rules; the struct is uploaded as a memory blob.
    Do not use bool: it is 1 byte in MSVC and 4 bytes in HLSL.
*/
struct RestirParams
{
    uint2 frameDim = { 0, 0 };
    uint frameIndex = 0;
    uint flags = 0;

    uint featureFlags = 0;
    uint maxPathLength = 6;
    uint primaryNEECandidateCount = 32;
    uint initialRussianRouletteStart = 2;

    uint pairingSlotCount = 3;
    uint lightTileCount = 128;
    uint lightsPerTile = 1024;
    uint temporalReplayCapacity = 0;

    uint spatialReplayCapacity = 0;
    float lightPowerInvWeightsSum = 0.f;
    uint primaryLodMode = 0;
    float cDefault = 20.f;

    float cMin = 1.f;
    float duplicationExponent = 0.1f;
    float temporalDepthThreshold = 0.1f;
    float footprintScale = 0.0002f;

    float minReconnectionRoughness = 0.2f;
    float spatialNormalThreshold = 0.9f;
    float spatialDepthThreshold = 0.1f;
    float initialRussianRouletteMinSurvival = 0.05f;

    uint duplicationRadius = 8;
    uint _padding1 = 0;
    float lodBias = 0.f;
    float specularRoughnessThreshold = 0.25f;

    uint maxDiffuseBounces = 8;
    uint maxSpecularBounces = 8;
    uint maxTransmissionBounces = 8;
    uint _padding0 = 0;

    float4x4 previousInvViewProj;

#ifndef HOST_CODE
    bool isPixelInBounds(uint2 pixel)
    {
        return all(pixel < frameDim);
    }

    uint pixelCount()
    {
        return frameDim.x * frameDim.y;
    }

    uint pixelIndex(uint2 pixel)
    {
        return pixel.y * frameDim.x + pixel.x;
    }

    uint2 pixelCoord(uint index)
    {
        return uint2(index % frameDim.x, index / frameDim.x);
    }

    bool hasFlag(uint flag)
    {
        return (flags & flag) != 0u;
    }

    bool hasFeatureFlag(uint flag)
    {
        return (featureFlags & flag) != 0u;
    }

    uint neeCandidateCount(uint bounceIndex)
    {
        const uint b = max(1u, bounceIndex);
        return max(1u, primaryNEECandidateCount / (b * b));
    }

    bool reprojectPixel(uint2 pixel, float2 motion, out uint2 previousPixel)
    {
        const float2 previousPosition = float2(pixel) + 0.5f + motion * float2(frameDim);
        const int2 p = int2(floor(previousPosition));
        const bool valid = all(p >= int2(0)) && all(p < int2(frameDim));
        previousPixel = valid ? uint2(p) : uint2(0u);
        return valid;
    }
#endif
};

END_NAMESPACE_FALCOR
