from __future__ import annotations
from typing import Iterable, Optional
from dataclasses import dataclass, field, asdict
from enum import IntEnum, IntFlag
from abc import ABC, abstractmethod
import copy

CAMERA_READOUT_TIME = 1 #1 second readout time, probably too pessimistic

class CameraSpeed(IntEnum):
    Speed16Mhz = 0
    Speed32Mhz = 1

class CameraDynamic(IntEnum):
    HighSensitivity = 2
    ExtendedDynamic = 3

class CameraReadoutMode(IntEnum):
    RollingShutter = 3
    BeamBlanking = 1

class ProcessingMode(IntEnum):
    NoProcessing = 0
    ClippingSum = 1
    Average = 2
    DriftCorrection = 3
    Counting = 4

class CorrectionMode(IntFlag):
    NoCorrection = 0
    DarkCorrected = 1
    GainNormalized = 2

class Shutter(IntFlag):
    BB = 1
    SH = 2
    

class CameraType(IntEnum):
    Simulator = 0
    DE = 13
    XF416 = 16
    XF416R = 17

@dataclass
class Axes2D:
    x: int
    y: int

@dataclass
class BurstParameters:
    NumBurstImages: int = 1
    IgnoreFirst: int = 0
    IgnoreLast: int = 0
    ProcessingMode: ProcessingMode = ProcessingMode.NoProcessing
    #todo: add parameters for Counting
    #todo: add parameters for configuring the drift correction

@dataclass
class CameraFormat(ABC):
    offset: Axes2D = Axes2D(x=0, y=0)
    binning: Axes2D = Axes2D(x=1, y=1)
    dimension: Axes2D = Axes2D(x=4096, y=4096)

    def GetUnbinnedOffset(self) -> Axes2D:
        return Axes2D(x=self.offset.x * self.binning.x,
                      y=self.offset.y * self.binning.y)

    @abstractmethod
    def Valid(self) -> bool:
        valid = True
        effectiveOffset = self.GetUnbinnedOffset()
        valid &= effectiveOffset.x >= 0
        valid &= effectiveOffset.y >= 0

        return valid

    @classmethod
    def GetDefault(cls, type:CameraType ) -> CameraFormat:
        if type == CameraType.Simulator:
            return CameraFormat()
        elif type == CameraType.XF416 or type == CameraType.XF416R:
            return CameraFormatXF416()
        else:
            raise NotImplementedError("Camera not yet supported")

    

@dataclass
class CameraFormatXF416(CameraFormat):
    addressIncrement: int = 256
    validBinnings: Iterable[int] = field(default=(1, 2, 4, 8))

    def Valid(self) -> bool:
        valid = super(CameraFormatXF416, self).Valid()
        for axis in ('x', 'y'):
            offset = asdict(self.GetUnbinnedOffset())
            valid &= offset[axis] <= 4096 - self.addressIncrement
            valid &= offset[axis] % self.addressIncrement == 0
            valid &= offset[axis] >= 0
            dimension = asdict(self.dimension)
            binning = asdict(self.binning)
            valid &= dimension[axis] % self.addressIncrement == 0
            valid &= dimension[axis] > 0
            valid &= dimension[axis] * binning[axis] + offset[axis] <= 4096

            valid &= binning[axis] in self.validBinnings
        return valid


@dataclass
class CameraConfiguration:
    speed: CameraSpeed = CameraSpeed.Speed16Mhz
    dynamic: CameraDynamic = CameraDynamic.HighSensitivity
    readoutMode: CameraReadoutMode = CameraReadoutMode.RollingShutter
    format: Optional[CameraFormat] = None
    exposureTime: int = 100 
    burstParameters: Optional[BurstParameters] = None
    correctionMode: CorrectionMode = CorrectionMode.DarkCorrected | CorrectionMode.GainNormalized

    def GetEstimatedAcquisitionTime(self) -> float:
        """ Estimated acquisition time in seconds """
        if self.burstParameters is not None:
            return self.burstParameters.NumBurstImages * self.exposureTime / 1000  + CAMERA_READOUT_TIME

        return self.exposureTime / 1000 + CAMERA_READOUT_TIME

    def Valid(self) -> bool:
        return True

    def Clone(self) -> CameraConfiguration:
        return copy.copy(self)

    def GetOutputDimensions(self) -> Optional[Axes2D]:
        if self.format is None:
            return None

        if self.burstParameters is not None:
            return Axes2D(x=self.format.dimension.x,
                          y=self.format.dimension.y * self.burstParameters.NumBurstImages)

        return Axes2D(x=self.format.dimension.x, y=self.format.dimension.y)

    @classmethod
    def GetDefault(cls, type: CameraType) -> CameraConfiguration:
        if type == CameraType.Simulator:
            cc = CameraConfiguration()
            cc.format = CameraFormat.GetDefault(type)
            return cc
        elif type == CameraType.XF416 or type == CameraType.XF416R:
            cc = CameraConfigurationXF416()
            cc.format = CameraFormat.GetDefault(type)
            cc.burstParameters = BurstParameters()
            return cc
        else:
            raise NotImplementedError("Camera not yet supported")


@dataclass
class CameraConfigurationXF416(CameraConfiguration):
    format: Optional[CameraFormatXF416] = None
    LCRows: int = -1
    dynamic: CameraDynamic=CameraDynamic.HighSensitivity
    speed: CameraSpeed = CameraSpeed.Speed16Mhz

    def Valid(self) -> bool:
        valid = super(CameraConfigurationXF416, self).Valid()
        valid &= False if self.format is None else self.format.Valid()
        #valid &= self.numBurst > 0
        valid &= self.LCRows >= -1
        valid &= self.LCRows <= 4095

        return valid
