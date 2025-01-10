import sys; sys.coinit_flags=0
import os
import logging

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
import time
import threading
import comtypes # type: ignore
import comtypes.server.localserver #type: ignore
from tvips.camera import CAMCLIB, MemReader
from tvips.common.types import Axes2D, CameraConfiguration, CameraConfigurationXF416, CameraFormat, CameraSpeed, CameraDynamic, CameraReadoutMode, BurstParameters, ProcessingMode, CameraType, CorrectionMode, Shutter

LOG = logging.getLogger(__name__)

class camcCallBack(comtypes.COMObject):
    _com_interfaces_ = [CAMCLIB.ICAMCCallBack, CAMCLIB.ICAMCImageCallBack]
    _reg_clsid_ = "{B06DDC47-93F5-4236-890F-CE8E3FAB2DD5}"
    _reg_threading_ = "Both"
    _reg_progid_ = "TVIPS.PYCAMCCallBack.1"
    _reg_novers_progid_ = "TVIPS.PYCAMCCallBack"
    _reg_desc_ = "Python CAMC callback"
    _reg_clsctx_ = comtypes.CLSCTX_INPROC_SERVER | comtypes.CLSCTX_LOCAL_SERVER
    _regcls_ = comtypes.server.localserver.REGCLS_MULTIPLEUSE

    #callbacks
    _singleImageCallback = None
    _burstImageCallback = None

    #variable to communicate to any instance requesting the lock
    yieldLock = 1

    #methods for ICAMCCallBack

    def LivePing(self):
        #print("Ping")
        pass

    def RequestLock(self):
        LOG.debug("LockRequest from external")
        return self.yieldLock

    #methods for ICAMCImageCallBack
    def BurstImageAvailable(self):
        LOG.debug("Burst image available")
        if self._burstImageCallback is not None:
            self._burstImageCallback()


    def SingleImageAvailable(self):
        LOG.debug("Single Image Available")
        if self._singleImageCallback is not None:
            self._singleImageCallback()
            #reset callback
            self._singleImageCallback = None

    #additional methods

    def SetSingleImageCallback(self, func):
        if self._singleImageCallback is not None:
            LOG.debug("Callback already installed, overwriting")
        self._singleImageCallback = func

    def SetBurstImageCallback(self, func):
        #install callback
        if self._burstImageCallback is not None:
            LOG.debug("Callback already installed, overwriting")
        self._burstImageCallback = func

 #function decorator.  Needs to be called from an instance which has
 #defined self.cameratype
def EnsureCamera(func: Callable) -> Callable:
    def wrap(self, *args, **kwargs):
        self._threadLock.acquire()
        LOG.debug("Threadlock acquired")
        self.camccallback.yieldLock = 0

        try:
            culprit, state = self.camcCamera.IsLocked
            LOG.debug("CAMC lock state: {} by {}".format("unlocked" if state == 0 else "locked", culprit))

            if not (state == 1 and culprit == self._getID() and self._hadCAMCOnce): #as long as I have the lock, I'm also initialized
                tries = 100
                success = -1
                while (success != CAMCLIB.crSucceed and tries > 0):
                    success = self.camcCamera.RequestLock()
                    LOG.debug("Tried to acquire lock ({} tries left): {}".format(tries, success))
                    if success == CAMCLIB.crSucceed:
                        break #don't lose time then
                    tries -= 1
                    time.sleep(0.1)

                if success != CAMCLIB.crSucceed:
                    culprit, state = self.camcCamera.IsLocked
                    raise RuntimeError("Could not get camera lock within 10s. Blame " + culprit)

                self._hadCAMCOnce = True

                self.camcCamera.Initialize(self._cameratype, 0)
                LOG.debug("CAMC initialized after having lost the lock (or first time init)")

            LOG.debug("CAMC lock state after trying to get the lock: {} by {}".format("unlocked" if state == 0 else "locked", culprit))

            #make sure correct camera is selected
            self.camcCamera.ActiveCamera = self._cameratype

            LOG.debug("SetActive cameraid {:d}".format(self._cameratype))

            return func(self, *args, **kwargs)

        finally:
            self.camccallback.yieldLock = 1
            self._threadLock.release()
            LOG.debug("Thread lock released, allowed CAMC to yield camc cameralock")

    return wrap

class camcGPU:

    #static properties
    camcCamera: CAMCLIB.ICameraGpu
    _threadLock = threading.RLock()
    _isLocked: bool = False
    _cameratype: CameraType = CameraType.Simulator
    _memReader: MemReader.MemReader = MemReader.MemReader()

    def __init__(self):
        try:
            comtypes.CoInitializeEx()
        except OSError:
            comtypes.CoInitialize()

        #get ICameraGpu interface
        try:
            self.camcCamera = comtypes.client.CreateObject(CAMCLIB.Camera, interface=CAMCLIB.ICameraGpu)
        except WindowsError as e:
            LOG.error("Unable to instantiate CAMC GPU. Make sure to use CAMC/EMMENU > 5.2.0")
            raise


        #callbacks
        self.camccallback = camcCallBack()
        self.camcCamera.RegisterCAMCCallBack(self.camccallback.QueryInterface(CAMCLIB.ICAMCCallBack), self._getID())
        self.camcCamera.RegisterCAMCImageCallBack(self.camccallback.QueryInterface(CAMCLIB.ICAMCImageCallBack))


        self._hadCAMCOnce = False # type: bool
        self._imageGeometry = 0 # type: int
        self._LCRows = -1 # type:int


    def _getID(self) -> str:
        return 'python (PID: {})'.format(os.getpid())

    def GetMinExposureOffset(self, cameraConfiguration: CameraConfiguration, imageGeometry: Optional[int]) -> int:
        """
        Get minimal exposure time for given format
        Args:
            cameraConfiguration: Configuration the camera is to be queried for
        Returns:
            exposure time offset in ms which is at the same time the value the requested exposure time has to be corrected for
        """
        #this assumes the camera is already configured (Format has been called,
        #mode, geometry and LC_rows selected)

        if imageGeometry is None:
            imageGeometry = self._imageGeometry

        if cameraConfiguration.format is None:
            raise ValueError("camera format needs to be specified")

        return self.camcCamera.RTPROPERTY(CAMCLIB.rtpMinExpTime,
                                          self._cameratype,
                                          cameraConfiguration.format.dimension.y * cameraConfiguration.format.binning.y,
                                          cameraConfiguration.readoutMode,
                                          imageGeometry, #sensor orientation
                                          cameraConfiguration.format.dimension.x * cameraConfiguration.format.binning.x,
                                          self._LCRows,
                                          0, 0, 0)

    @EnsureCamera
    def GetCurrentCameraConfiguration(self) -> CameraConfiguration:

        c = CameraConfiguration(speed=CameraSpeed(self.camcCamera.LParam[CAMCLIB.cpLLSCXAmplifier]),
                                dynamic=CameraDynamic(self.camcCamera.LParam[CAMCLIB.cpCurrentGainIndex]),
                                readoutMode=CameraReadoutMode(self.camcCamera.LParam[CAMCLIB.cpReadoutMode])
                                )

        #c.format = CameraFormatXF416() #there is no way to reading out the last format sent to camc...
        #c.exposureTime = None #no way to read that back
        return c

    @EnsureCamera
    def SetFormat(self, format: Optional[CameraFormat] = None) -> None:
        if format is None:
            format = CameraFormat.GetDefault(self._cameratype)
        self.camcCamera.Format(format.GetUnbinnedOffset().x,
                               format.GetUnbinnedOffset().y,
                               format.dimension.x,
                               format.dimension.y,
                               format.binning.x,
                               format.binning.y)

    @EnsureCamera
    def _setBurstParameters(self, burstParameters: Optional[BurstParameters] = None) -> None:

        if burstParameters is None:
            #configure burst as it was disabled (default BurstParameters)
            burstParameters = BurstParameters()

        self.camcCamera.LParam[CAMCLIB.cpBurstNumImages] = burstParameters.NumBurstImages
        self.camcCamera.LParam[CAMCLIB.cpBurstIgnoreFirstN] = burstParameters.IgnoreFirst
        self.camcCamera.LParam[CAMCLIB.cpBurstIgnoreLastM] = burstParameters.IgnoreLast

        if burstParameters.ProcessingMode == ProcessingMode.NoProcessing:
            self.camcCamera.LParam[CAMCLIB.cpBurstUseProcessing] = 0
            self.camcCamera.LParam[CAMCLIB.cpBurstUseProcessing] = 0
        else:
            self.camcCamera.LParam[CAMCLIB.cpBurstUseProcessing] = 1
            self.camcCamera.LParam[CAMCLIB.cpBurstWhatProcessing] = burstParameters.ProcessingMode

    @EnsureCamera
    def _configureCamera(self, cameraConfiguration: CameraConfiguration):
        self._setBurstParameters(cameraConfiguration.burstParameters)
        self.camcCamera.LParam[CAMCLIB.cpCorrectionMode] = cameraConfiguration.correctionMode
        self.camcCamera.LParam[CAMCLIB.cpCurrentGainIndex] = cameraConfiguration.dynamic 
        self.camcCamera.LParam[CAMCLIB.cpLLSCXAmplifier] = cameraConfiguration.speed 
        self.camcCamera.LParam[CAMCLIB.cpReadoutMode] = cameraConfiguration.readoutMode 
        if isinstance(cameraConfiguration, CameraConfigurationXF416):
            self.camcCamera.LParam[CAMCLIB.cpLCRows] = cameraConfiguration.LCRows

        #Note: everything has to be set before calling format!
        self.SetFormat(cameraConfiguration.format)

    def _readSingleImage(self, cameraFormat: CameraFormat):
        #no need to introduce callbacks here, as we expect only one single frame
        return self._memReader.GetLastImage(cameraFormat.dimension)

    def _getShutter(self) -> Shutter:
        sh = self.camcCamera.ShutterMode()
        if sh == CAMCLIB.smSH:
            return Shutter.SH
        if sh == CAMCLIB.SH_BB:
            return Shutter.SH | Shutter.BB
        if sh == CAMCLIB.smBB:
            return Shutter.BB
        raise NotImplementedError("Unsupported shutter mode")


    def _setShutter(self, shutterMode: Shutter) -> None:
        sh = None
        if shutterMode == Shutter.SH | Shutter.BB:
            sh = CAMCLIB.smSH_BB
        elif shutterMode == Shutter.SH:
            sh = CAMCLIB.smSH
        elif shutterMode == Shutter.BB:
            sh = CAMCLIB.smBB
        else:
            raise NotImplementedError("Shutter mode not implemented")

        self.camcCamera.ShutterMode = sh

    def _setShutterForceBlank(self):
        self.camcCamera.ShutterOverride(CAMCLIB.smBB, 1, 0)
        self.camcCamera.ShutterOverride(CAMCLIB.smSH, 1, 0)

    def _setShutterForceUnblank(self):
        self.camcCamera.ShutterOverride(CAMCLIB.smBB, 1, 1)
        self.camcCamera.ShutterOverride(CAMCLIB.smSH, 1, 1)

    def _setShutterNormal(self):
        self.camcCamera.ShutterOverride(CAMCLIB.smBB, 0, 0)
        self.camcCamera.ShutterOverride(CAMCLIB.smSH, 0, 0)


    @EnsureCamera
    def _acquireImage(self, cameraConfiguration: Optional[CameraConfiguration] = None):
        if cameraConfiguration is None:
            cameraConfiguration = CameraConfiguration.GetDefault(self._cameratype)
        self._configureCamera(cameraConfiguration)
        self.camcCamera.AcquireImageBlocking(cameraConfiguration.exposureTime)
        if cameraConfiguration.burstParameters.NumBurstImages == 1 or cameraConfiguration.burstParameters.ProcessingMode != ProcessingMode.NoProcessing:
            return self._readSingleImage(cameraConfiguration.format)
        else:
            raise NotImplementedError()

    @EnsureCamera
    def AcquireImage(self, cameraConfiguration: Optional[CameraConfiguration] = None):
        self._setShutterNormal()
        return self._acquireImage(cameraConfiguration)

    @EnsureCamera
    def AcquireDark(self, cameraConfiguration: Optional[CameraConfiguration] = None):
        self._setShutterForceBlank()
        return self._acquireImage(cameraConfiguration)

