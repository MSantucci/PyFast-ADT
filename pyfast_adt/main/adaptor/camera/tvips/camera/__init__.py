import sys
sys.coinit_flags = 0

import os
try:
    import comtypes #type: ignore
    import comtypes.client #type: ignore
except OSError as e:
    #ipython imports comtypes package without coinit_flags set, so undo that flag and try importing it
    #Note: this might only happen for ipython and python 3.7?
    del sys.coinit_flags
    import comtypes #type: ignore
    import comtypes.client #type: ignore

if sys.maxsize < 2**32 + 1:
    print("It's strongly recommended to use a 64 bit enabled python distribution, especially if you are interested in burst mode. \nOtherwise consider reducing the data using real-time drift correction on the GPU")

CAMC_PATH = r"c:\tvips\emmenu\bin\camc4.exe"


if not os.path.exists(CAMC_PATH):
    raise RuntimeError("CAMC4 not found")

CAMCLIB = comtypes.client.GetModule(CAMC_PATH)


def ConstructCamera(id):
    from tvips.camera.camcGPU import camcGPU
    class CamcGPUCamera(camcGPU):
        _cameratype=id

    return CamcGPUCamera 

XF416 = ConstructCamera(16)
XF416R = ConstructCamera(17)
Simulator = ConstructCamera(0)

