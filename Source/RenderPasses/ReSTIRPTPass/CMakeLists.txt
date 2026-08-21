add_plugin(ReSTIRPTPass)

target_sources(ReSTIRPTPass PRIVATE
    ReSTIRPTPass.cpp
    ReSTIRPTPass.h
    Params.slang

    Common/Types.slang
    Common/EnhancedPolicy.slang
    Common/SurfaceTypes.slang
    Common/SurfaceLoad.slang
    Common/SurfaceTrace.slang
    Common/Surface.slang
    Common/Lights.slang
    Common/GRIS.slang
    Common/Pairing.slang
    Common/HybridShift.slang

    Path/PathSampling.slang
    Path/InitialSample.cs.slang

    Temporal/TemporalShift.cs.slang
    Temporal/TemporalReplay.cs.slang
    Temporal/TemporalReuse.cs.slang

    Spatial/SpatialShift.cs.slang
    Spatial/SpatialReplay.cs.slang
    Spatial/SpatialResample.cs.slang
    Spatial/BuildPairing.cs.slang

    Frame/ReflectTypes.cs.slang
    Frame/ResetFrame.cs.slang
    Frame/BuildDispatchArgs.cs.slang
    Frame/PresampleLights.cs.slang
    Frame/BuildDualMotion.cs.slang
    Frame/DuplicationMap.cs.slang
    Frame/Resolve.cs.slang
)

target_copy_shaders(ReSTIRPTPass RenderPasses/ReSTIRPTPass)

target_source_group(ReSTIRPTPass "RenderPasses")
