#####################################################################################
# Copyright 2020 by Marco Oster/TVIPS GmbH
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation and/or
#    other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
######################################################################################
import logging
LOG = logging.getLogger(__name__)

import comtypes, comtypes.client
import comtypes.server, comtypes.server.localserver
import mmap
import os


import threading
import time
import logging
import numpy as np # type: ignore

    
class GPUCameraBase:
    cameratype = None

    def __init__(self):
        self.camc = CamcGPU()
        self.memreader = CamcSharedMemReader()

        self.HadCAMCOnce = False #keep track of having had camc once, since the Islocked routine might return "Leginon" also from a former session and bypass initialization

        #ccdcamera.CCDCamera specific initialization copied from tietz2.py
        self.nframes = 1 #TODO: is that the number of frames requested?

        # set binning first so we can use it
        self.initSettings()

   
    '''
    @RequestCamera
    def acquireImage(self):
        self.camc.camera.Format(0,0,4096,4096,1,1)
        print("format set")
        self.camc.camera.AcquireImageAsync(40)
        print("Acquiring image")
    '''


    @RequestCamera
    def _getImage(self):
        """Get a single image."""


        self._setFormat()

        self.camc.camera.LParam[camclib.cpBurstNumImages] = 1
        self.camc.camera.LParam[camclib.cpReadoutMode] = 1 #Beam blanking mode for single images

        #TODO: Revisit this if Leginon is ever ported to python3 and use async await
        im = None
        ImageReceivedEvent = threading.Event()

        self.camc.camccallback.SetSingleImageCallback(ImageReceivedEvent.set)

        self.camc.camera.AcquireImageAsync(int(self.exposure)) 

        ImageReceivedEvent.wait(0.001 * self.exposure + 10) #timeout

        if not ImageReceivedEvent.isSet():
            raise Exception("Could not acquire image in reasonable time")

        im = self.memreader.GetLastImage(self.dimension['x'], self.dimension['y'])

        return im

    @RequestCamera
    def _getBurstImage(self, n=10):

        self._setFormat()

        self.camc.camera.LParam[camclib.cpBurstNumImages] = n 
        self.camc.camera.LParam[camclib.cpReadoutMode] = 3 #RS mode for burst
        self.camc.camera.LParam[camclib.cpBurstUseProcessing] = 0

        expOffset = self._getMinExpOffset()

        camcExpTime = max(expOffset, int(self.exposure) - expOffset)

        #allocate
        import gc; gc.collect()
        im = np.empty((n, self.dimension['x'], self.dimension['y']), dtype=np.uint16)

        evt = threading.Event()
        self.camc.camccallback.SetBurstImageCallback(evt.set)

        self.camc.camera.AcquireImageAsync(camcExpTime)
        seen = np.zeros(n, dtype=np.bool)
        for j in range(n):
            evt.wait(3)
            seen[j] = not evt.isSet() #keep track of missing events
            im[j] = self.memreader.GetLastImage(self.dimension['x'], self.dimension['y'])
            evt.clear()

        missed = np.count_nonzero(seen)
        if missed > 0:
            print("TVIPS XF416: missed {:d} burst images at {:d}({:d})ms exposure time".format(missed, int(self.exposure), camcExpTime))


        return im 

    @RequestCamera
    def _getDriftCorrectedImage(self, n=10):

        self._setFormat()

        self.camc.camera.LParam[camclib.cpBurstNumImages] = n 
        self.camc.camera.LParam[camclib.cpReadoutMode] = 3 #RS mode for burst

        expOffset = self._getMinExpOffset()

        camcExpTime = max(expOffset, int(self.exposure) - expOffset)

        #set up drift correction parameters
        """
        8001 cpXcfCutAcp Long DriftDoCutACP 
        8002 cpXcfAcpPixels Long DriftCutACPPixels 
        8003 cpXcfWidth Long DriftXcfWidth 
        8004 cpXcfHeight Long DriftXcfHeight 
        8005 cpXcfOffsetX Long DriftXcfOffsetX 
        8006 cpXcfOffsetY Long DriftXcfOffsetY 
        8007 cpXcfApplyBandPass Long DriftXcfApplyBandpass 
        8008 cpXcfBandLow Long DriftXcfBandLow 
        8009 cpXcfBandHigh Long DriftXcfBandHigh 
        8010 cpXcfDoPhaseCorrelation Long Do Phase Correlation 
        8011 cpXcfDoFitPeak Long Fit XCF Peak 
        8012 cpXcfBinning Long Calculate XCF on binned image
        """

        self.camc.camera.LParam[camclib.cpXcfCutAcp] = 0
        self.camc.camera.LParam[camclib.cpXcfWidth] = self.dimension['x']
        self.camc.camera.LParam[camclib.cpXcfHeight] = self.dimension['y']
        self.camc.camera.LParam[camclib.cpXcfOffsetX] = self.offset['x']
        self.camc.camera.LParam[camclib.cpXcfOffsetY] = self.offset['y']

        self.camc.camera.LParam[camclib.cpXcfApplyBandPass] = 1
        self.camc.camera.LParam[camclib.cpXcfBandLow] = 0
        self.camc.camera.LParam[camclib.cpXcfBandHigh] = 3

        self.camc.camera.LParam[camclib.cpXcfDoFitPeak] = 1

        self.camc.camera.LParam[camclib.cpBurstUseProcessing] = 1
        self.camc.camera.LParam[camclib.cpBurstWhatProcessing] = 2 #(0=sum, 1=avr, 2=align, 3=counting )

        ImageReceivedEvent = threading.Event()

        self.camc.camccallback.SetBurstImageCallback(ImageReceivedEvent.set)
        self.camc.camera.AcquireImageAsync(camcExpTime) 

        ImageReceivedEvent.wait(0.001 * self.exposure * n + 10) #timeout

        if not ImageReceivedEvent.isSet():
            raise Exception("Could not acquire image in reasonable time")

        im = self.memreader.GetLastImage(self.dimension['x'], self.dimension['y'])

        return im



class XF416(GPUCameraBase):
    name = 'TVIPS XF416'
    camera_name = 'TVIPS XF416'
    intensity_averaged = False #TODO: what does that mean?
    binning_limits = [1, 2, 4, 8]

    cameratype = camclib.ctXF416e_GPU

class XF416R(GPUCameraBase):
    name = 'TVIPS XF416 retractable'
    camera_name = 'TVIPS XF416 retractable'
    intensity_averaged = False #TODO: what does that mean?
    binning_limits = [1, 2, 4, 8]

    cameratype = camclib.ctXF416e_GPUr

    #todo: insert code for retracting/inserting




