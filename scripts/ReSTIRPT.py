from falcor import *

def render_graph_ReSTIRPT():
    g = RenderGraph("ReSTIRPT")
    ReSTIRPT = createPass(
        "ReSTIRPTPass",
        {
            "maxPathLength": 6,
            "enableDuplicationMap": True,
            "enableRussianRoulette": True,
            "enableVectorWeights": True,
            "enableDualMotion": True,
            "enableForcedNEEReconnect": True,
            "enablePresampledLights": True,
        },
    )
    g.addPass(ReSTIRPT, "ReSTIRPT")
    VBufferRT = createPass(
        "VBufferRT",
        {
            "samplePattern": "Halton",
            "sampleCount": 16,
            "useAlphaTest": True,
        },
    )
    g.addPass(VBufferRT, "VBufferRT")
    DLSS = createPass(
        "DLSSPass",
        {
            "enabled": False,
            "profile": "Balanced",
            "motionVectorScale": "Relative",
            "isHDR": True,
            "sharpness": 0.0,
            "exposure": 0.0,
        },
    )
    g.addPass(DLSS, "DLSS")
    AccumulatePass = createPass("AccumulatePass", {"enabled": False, "precisionMode": "Single"})
    g.addPass(AccumulatePass, "AccumulatePass")
    ToneMapper = createPass("ToneMapper", {"autoExposure": False, "exposureCompensation": 0.0})
    g.addPass(ToneMapper, "ToneMapper")
    g.addEdge("VBufferRT.vbuffer", "ReSTIRPT.vbuffer")
    g.addEdge("VBufferRT.mvec", "ReSTIRPT.mvec")
    g.addEdge("VBufferRT.depth", "ReSTIRPT.depth")
    g.addEdge("ReSTIRPT.color", "DLSS.color")
    g.addEdge("VBufferRT.mvec", "DLSS.mvec")
    g.addEdge("VBufferRT.depth", "DLSS.depth")
    g.addEdge("DLSS.output", "AccumulatePass.input")
    g.addEdge("AccumulatePass.output", "ToneMapper.src")
    g.markOutput("ToneMapper.dst")
    g.markOutput("ReSTIRPT.color")
    return g

ReSTIRPT = render_graph_ReSTIRPT()
try:
    m.addGraph(ReSTIRPT)
except NameError:
    None
