from .adaptor_cam import Cam_base
import json
import numpy as np
import atexit
import logging
import time
from pathlib import Path
import comtypes.client
from tvips.common.types import Axes2D, CameraConfiguration, CameraConfigurationXF416, CameraFormat, CameraSpeed, \
    CameraDynamic, CameraReadoutMode, BurstParameters, ProcessingMode, CameraType, CorrectionMode, Shutter
from tvips.camera import MemReader
import os, cv2
import imageio
import tkinter
import matplotlib.pyplot as plt
# in this adaptor are present 2 classes, CameraEMMENUGPU and camc_GPU.
# the first one is used to do everything where the second one is used to acquire cred data only!
type_dict = {
    1: 'GetDataByte',
    2: 'GetDataUShort',
    3: 'GetDataShort',
    4: 'GetDataLong',
    5: 'GetDataFloat',
    6: 'GetDataDouble',
    7: 'GetDataComplex',
    8: 'IMG_STRING',  # no method on EMImage
    9: 'GetDataBinary',
    10: 'GetDataRGB8',
    11: 'GetDataRGB16',
    12: 'IMG_EMVECTOR',  # no method on EMImage
}

def EMVector2dict(vec):
    """Convert EMVector object to a Python dictionary."""
    d = {}
    for k in dir(vec):
        if k.startswith('_'):
            continue
        v = getattr(vec, k)
        if isinstance(v, int):
            d[k] = v
        elif isinstance(v, float):
            d[k] = v
        elif isinstance(v, str):
            d[k] = v
        elif isinstance(v, comtypes.Array):
            d[k] = list(v)
        else:
            print(k, v, type(v))

    return d

class Cam_xf416r(Cam_base):
    """Software interface for the EMMENU program and camc5.22 to control live readout of the buffer for acquiring series of images

        Communicates with EMMENU over the COM interface defined by TVIPS, and directly to the camera using the API of TVIPS.
        EMMENU is used to give a friendly way to set up the camera parameters and see the live view of the camera.
        TVIPS API is used to acquire the images from the camera buffer, to reach 20 FPS during acquisition.

        its required to have a folder tree in EMMENU called Diffraction
        drc_name : str
            Set the default folder to store data in
        name : str
            Name of the interface
        """
    def __init__(self, drc_name: str = 'Diffraction', name: str = 'emmenu', instance_gui = None):
        """Initialize camera module."""
        super().__init__()
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        except OSError:
            comtypes.CoInitialize()

        self.name = name  # name of the camera
        self.drc_name = drc_name  # name of the folder where to save the data
        self.exposure = None  # ms
        self.x = None
        self.y = None
        self.processing = None
        self.delay = None  # seconds
        self.binning = None
        self.buffer_size = None
        self.buffer = None
        self.stop_signal = None
        self.table = None
        self.timings = []
        self.load_calibration_table()
        self.instance_gui = instance_gui
        #connecting the camera and setting up the release on disconnection
        self.connect()
        atexit.register(self.releaseConnection)

    def connect(self):
        """Connect to the camera."""
        try:
            ######## started the initialization of the EMMenu ###########
            self._memReader = MemReader.MemReader()
            self.image = None
            self.result_param = []

            self._obj = comtypes.client.CreateObject('EMMENU4.EMMENUApplication.1', comtypes.CLSCTX_ALL)
            self._recording = False

            # get first camera
            self._cam = self._obj.TEMCameras.Item(1)

            # hi-jack first viewport
            self._vp = self._obj.Viewports.Item(1)
            self._vp.SetCaption('FAST-ADT viewport')
            self._vp.FlapState = 2  # pull out the flap, because we can :-) [0, 1, 2]

            self._obj.Option('ClearBufferOnDeleteImage')  # `Delete` -> Clear buffer (preferable)
            # other choices: DeleteBufferOnDeleteImage / Default

            # Image manager for managing image buffers (left panel)
            self._immgr = self._obj.ImageManager

            # for writing tiff files
            self._emf = self._obj.EMFile

            # stores all pointers to image data
            self._emi = self._obj.EMImages

            # set up instamatic data directory
            self.top_drc_index = self._immgr.TopDirectory
            self.top_drc_name = self._immgr.DirectoryName(self.top_drc_index)

            # check if exists
            if not self._immgr.DirectoryExist(self.top_drc_index, self.drc_name):
                if self.getEMMenuVersion().startswith('4.'):
                    self._immgr.CreateNewSubDirectory(self.top_drc_index, self.drc_name, 2, 2)
                else:
                    # creating new subdirectories is bugged in EMMENU 5.0.9.0/5.0.10.0
                    # No work-around -> raise exception for now until it is fixed
                    raise ValueError(
                        f'Directory `{self.drc_name}` does not exist in the EMMENU Image manager. \nPlease create it')

            self.drc_index = self._immgr.DirectoryHandleFromName(self.drc_name)
            self._vp.DirectoryHandle = self.drc_index  # set current directory

            # load initial parameters for camera\emmenu they will be overwrited later
            self.streamable = False
            self.default_exposure = 50
            self.default_binsize = 4
            self.dimensions = 1024

            print('initialization EMMENU Cam done!')
            ######## start initialization of the camc_GPU ###########
            CAMC_PATH = r"c:\tvips\emmenu\bin\camc4.exe"
            if not os.path.exists(CAMC_PATH):
                raise RuntimeError("CAMC4 not found")

            self.CAMCLIB = comtypes.client.GetModule(CAMC_PATH)
            self.cameraGPU = comtypes.client.CreateObject(self.CAMCLIB.Camera, interface=self.CAMCLIB.ICameraGpu)
            self.cameraLiveGPU = comtypes.client.CreateObject(self.CAMCLIB.Camera,
                                                              interface=self.CAMCLIB.ICameraLiveGpu)
        except Exception as err:
            print('Error in the initialization of the camera', err)

    def listConfigs(self) -> list:
        """List the configs from the Configuration Manager."""
        print(f'Configurations for camera {self.name}')
        current = self._vp.Configuration

        lst = []

        for i, cfg in enumerate(self._obj.CameraConfigurations):
            is_selected = (current == cfg.Name)
            end = ' (selected)' if is_selected else ''
            print(f'{i + 1:2d} - {cfg.Name}{end}')
            lst.append(cfg.Name)

        return lst

    def getCurrentConfigName(self) -> str:
        """Return the name of the currently selected configuration in
        EMMENU."""
        cfg = self.getCurrentConfig(as_dict=False)
        return cfg.Name

    def getCurrentConfig(self, as_dict: bool = True) -> dict:
        """Get selected config object currently associated with the
        viewport."""
        vp_cfg_name = self._vp.Configuration
        count = self._obj.CameraConfigurations.Count
        for j in range(1, count + 1):
            cfg = self._obj.CameraConfigurations.Item(j)
            if cfg.Name == vp_cfg_name:
                break

        if as_dict:
            d = {}
            d['Name'] = cfg.Name  # str
            d['CCDOffsetX'] = cfg.CCDOffsetX  # int
            d['CCDOffsetY'] = cfg.CCDOffsetY  # int
            d['DimensionX'] = cfg.DimensionX  # int
            d['DimensionY'] = cfg.DimensionY  # int
            d['BinningX'] = cfg.BinningX  # int
            d['BinningY'] = cfg.BinningY  # int
            d['CameraType'] = cfg.CameraType  # str
            d['GainValue'] = cfg.GainValue  # float
            d['SpeedValue'] = cfg.SpeedValue  # int
            d['FlatMode'] = cfg.FlatMode  # int
            d['FlatModeStr'] = ('Uncorrected', 'Dark subtracted', None, 'Gain corrected')[cfg.FlatMode]  # str, 2 undefined
            d['PreExposureTime'] = cfg.PreExposureTime  # int
            d['UsePreExposure'] = bool(cfg.UsePreExposure)
            d['ReadoutMode'] = cfg.ReadoutMode  # int
            d['ReadoutModeStr'] = (None, 'Normal', 'Frame transfer', 'Rolling shutter')[cfg.ReadoutMode]  # str, 0 undefined
            d['UseRollingAverage'] = bool(cfg.UseRollingAverage)
            d['RollingAverageValue'] = cfg.RollingAverageValue  # int
            d['UseRollingShutter'] = bool(cfg.UseRollingShutter)
            d['UseScriptPreExposure'] = bool(cfg.UseScriptPreExposure)
            d['UseScriptPostExposure'] = bool(cfg.UseScriptPostExposure)
            d['UseScriptPreContinuous'] = bool(cfg.UseScriptPreContinuous)
            d['UseScriptPostContinuous'] = bool(cfg.UseScriptPostContinuous)
            d['ScriptPathPostExposure'] = cfg.ScriptPathPostExposure  # str
            d['ScriptPathPreContinuous'] = cfg.ScriptPathPreContinuous  # str
            d['ScriptPathPostContinuous'] = cfg.ScriptPathPostContinuous  # str
            d['ScriptPathBeforeSeries'] = cfg.ScriptPathBeforeSeries  # str
            d['ScriptPathWithinSeries'] = cfg.ScriptPathWithinSeries  # str
            d['ScriptPathAfterSeries'] = cfg.ScriptPathAfterSeries  # str
            d['SCXAmplifier'] = cfg.SCXAmplifier  # int
            d['SCXAmplifierStr'] = ('Unknown', 'Low noise', 'High capacity')[cfg.SCXAmplifier]  # str
            d['CenterOnChip'] = bool(cfg.CenterOnChip)
            d['SeriesType'] = cfg.SeriesType  # int
            d['SeriesTypeStr'] = ('Single image', 'Delay series', 'Script series')[cfg.SeriesType]  # str
            d['SeriesNumberOfImages'] = cfg.SeriesNumberOfImages
            d['SeriesDelay'] = cfg.SeriesDelay  # int
            d['SeriesAlignImages'] = bool(cfg.SeriesAlignImages)
            d['SeriesIntegrateImages'] = bool(cfg.SeriesIntegrateImages)
            d['SeriesAverageImages'] = bool(cfg.SeriesAverageImages)
            d['SeriesDiscardIndividualImages'] = bool(cfg.SeriesDiscardIndividualImages)
            d['UseScriptBeforeSeries'] = bool(cfg.UseScriptBeforeSeries)
            d['UseScriptWithinSeries'] = bool(cfg.UseScriptWithinSeries)
            d['UseScriptAfterSeries'] = bool(cfg.UseScriptAfterSeries)
            d['ShutterMode'] = cfg.ShutterMode  # int
            d['ShutterModeStr'] = ('None', 'SH', 'BB', 'SH/BB', 'Dark/SH', 'Dark/BB', 'Dark/SH/BB')[cfg.ShutterMode]  # str
            return d
        else:
            return cfg

    def getCurrentCameraInfo(self) -> dict:
        """Gets the current camera object."""
        cam = self._cam

        d = {}
        d['RealSizeX'] = cam.RealSizeX  # int
        d['RealSizeY'] = cam.RealSizeY  # int
        d['MaximumSizeX'] = cam.MaximumSizeX  # int
        d['MaximumSizeY'] = cam.MaximumSizeY  # int
        d['NumberOfGains'] = cam.NumberOfGains  # int
        d['GainValues'] = [cam.GainValue(val) for val in range(cam.NumberOfGains + 1)]
        d['NumberOfSpeeds'] = cam.NumberOfSpeeds  # int
        d['SpeedValues'] = [cam.SpeedValue(val) for val in range(cam.NumberOfSpeeds + 1)]
        d['PixelSizeX'] = cam.PixelSizeX  # int
        d['PixelSizeY'] = cam.PixelSizeY  # int
        d['Dynamic'] = cam.Dynamic  # int
        d['PostMag'] = cam.PostMag  # float
        d['CamCGroup'] = cam.CamCGroup  # int
        return d

    def getCameraType(self) -> str:
        """Get the name of the camera currently in use."""
        cfg = self.getCurrentConfig(as_dict=False)
        return cfg.CameraType

    def getEMMenuVersion(self) -> str:
        """Get the version number of EMMENU."""
        return self._obj.EMMENUVersion

    def lock(self) -> None:
        """Lockdown interactions with emmenu, must call `self.unlock` to
        unlock.

        If EMMenu is locked, no mouse or keyboard input will be accepted
        by the interface. The script calling this function is
        responsible for unlocking EMMenu.
        """
        self._obj.EnableMainframe(1)

    def unlock(self) -> None:
        """Unlock emmenu after it has been locked down with `self.lock`"""
        self._obj.EnableMainframe(0)

    def listDirectories(self) -> dict:
        """List subdirectories of the top directory."""
        top_j = self._immgr.TopDirectory
        top_name = self._immgr.FullDirectoryName(top_j)
        print(f'{top_name} ({top_j})')

        drc_j = self._immgr.SubDirectory(top_j)
        d = {}

        while drc_j:
            drc_name = self._immgr.FullDirectoryName(drc_j)
            print(f'{drc_j} - {drc_name} ')

            d[drc_j] = drc_name

            drc_j = self._immgr.NextDirectory(drc_j)  # get next

        return d

    def getEMVectorByIndex(self, img_index: int, drc_index: int = None) -> dict:
        """Returns the EMVector by index as a python dictionary."""
        p = self.getImageByIndex(img_index, drc_index)
        v = p.EMVector
        d = EMVector2dict(v)
        return d

    def deleteAllImages(self) -> None:
        """Clears all images currently stored in EMMENU buffers."""
        for i, p in enumerate(self._emi):
            try:
                self._emi.DeleteImage(p)
            except BaseException:
                # sometimes EMMenu also loses track of image pointers...
                print(f'Failed to delete buffer {i} ({p})')

    def deleteImageByIndex(self, img_index: int, drc_index: int = None) -> int:
        """Delete the image from EMMENU by its index."""
        p = self.getImageByIndex(img_index, drc_index)
        self._emi.DeleteImage(p)  # alternative: self._emi.Remove(p.ImgHandle)

    def getImageByIndex(self, img_index: int, drc_index: int = None) -> int:
        """Grab data from the image manager by index. Return image pointer
        (COM).

        Not accessible through server.
        """
        if not drc_index:
            drc_index = self.drc_index

        p = self._immgr.Image(drc_index, img_index)

        return p

    def getImageDataByIndex(self, img_index: int, drc_index: int = None) -> 'np.array':
        """Grab data from the image manager by index.

        Return numpy 2D array
        """
        p = self.getImageByIndex(img_index, drc_index)

        tpe = p.DataType
        method = type_dict[tpe]

        f = getattr(p, method)
        arr = f()  # -> tuple of tuples

        return np.array(arr)

    def getCameraDimensions(self) -> (int, int):
        """Get the maximum dimensions reported by the camera."""
        # cfg = self.getCurrentConfig()
        # return cfg.DimensionX, cfg.DimensionY
        return self._cam.RealSizeX, self._cam.RealSizeY
        # return self._cam.MaximumSizeX, self._cam.MaximumSizeY

    def getImageDimensions(self) -> (int, int):
        """Get the dimensions of the image."""
        binning = self.getBinning()
        return int(self._cam.RealSizeX / binning), int(self._cam.RealSizeY / binning)

    def getPhysicalPixelsize(self) -> (int, int):
        """Returns the physical pixel size of the camera nanometers."""
        return self._cam.PixelSizeX, self._cam.PixelSizeY

    def getBinning(self) -> int:
        """Returns the binning corresponding to the currently selected camera
        config."""
        cfg = self.getCurrentConfig(as_dict=False)
        bin_x = cfg.BinningX
        bin_y = cfg.BinningY
        assert bin_x == bin_y, 'Binnings differ in X and Y direction! (X: {bin_x} | Y: {bin_y})'
        return bin_x
    def get_binning(self): self.getBinning()

    def set_binning(self, binning: int):
        print("not possible use emmenu configuration")
        pass

    def set_processing(self, processing: str):
        '''' set the processing of the camera,
        processing = "Unprocessed, Background subtracted, Gain normalized"'''
        try:
            self.processing = str(processing)
        except Exception as err: print(err)

    def get_processing(self):
        '''' get the processing typeof the camera '''
        # xf416r are ('Uncorrected', 'Dark subtracted', None, 'Gain corrected')[cfg.FlatMode]  # str, 2 undefined
        return self.processing

    def getCameraName(self) -> str:
        """Get the name reported by the camera."""
        return self._cam.name

    def writeTiffFromPointer(self, image_pointer, filename: str) -> None:
        """Write tiff file using the EMMENU machinery `image_pointer` is the
        memory address returned by `getImageIndex()`"""
        self._emf.WriteTiff(image_pointer, filename)

    def writeTiff(self, image_index, filename: str) -> None:
        """Write tiff file using the EMMENU machinery `image_index` is the
        index in the current directory of the image to be written."""
        drc_index = self.drc_index
        p = self.getImageByIndex(image_index, drc_index)

        self.writeTiffFromPointer(p, filename)

    def writeTiffs(self, start_index: int, stop_index: int, path: str, clear_buffer: bool = False) -> None:
        """Write a series of data in tiff format and writes them to the given
        `path` using EMMENU machinery."""
        path = Path(path)
        drc_index = self.drc_index

        if stop_index <= start_index:
            raise IndexError(f'`stop_index`: {stop_index} >= `start_index`: {start_index}')

        for i, image_index in enumerate(range(start_index, stop_index + 1)):
            p = self.getImageByIndex(image_index, drc_index)

            fn = str(path / f'{i:04d}.tiff')
            print(f'Image #{image_index} -> {fn}')

            # TODO: wrap writeTiff in try/except
            # writeTiff causes vague error if image does not exist

            self.writeTiffFromPointer(p, fn)

            if clear_buffer:
                # self._immgr.DeleteImageBuffer(drc_index, image_index)  # does not work on 3200
                self._emi.DeleteImage(p)  # also clears from buffer

        print(f'Wrote {i + 1} images to {path}')

    def getImage(self, **kwargs) -> 'np.array':
        """Acquire image through EMMENU and return data as np array."""
        print('camera_emmenu.py line 403 now!')
        self._vp.AcquireAndDisplayImage()
        i = self.get_image_index()
        return self.getImageDataByIndex(i)

    def acquire_image(self, exposure_time: int, binning: int, processing: str):
        self.set_exposure(int(exposure_time))
        #binning = binning, processing = processing todo
        img = self.getImage()
        img = self.rotate_img(img)
        return img

    def acquire_image_and_show(self, binning: int, exposure_time: int, image_size: str):
        self.set_exposure(int(exposure_time))
        # binning = binning, processing = processing todo
        img = self.getImage()
        img = self.rotate_img(img)
        # plt.imshow(img)
        # plt.show()
        # new line for streaming
        if self.is_cam_streaming():
            self.image_to_streaming(img)
        return img

    def rotate_img(self, img, times=None, flip_h=None, flip_v=None, flip_diag = None):
        times = times if times is not None else int(self.table["cam_rotate90x"])
        flip_h = flip_h if flip_h is not None else self.table["cam_flip_h"]
        flip_v = flip_v if flip_v is not None else self.table["cam_flip_v"]
        flip_diag = flip_diag if flip_diag is not None else self.table["cam_flip_diag"]

        if times is not None and times != 0:
            img = np.rot90(img, times)
        if flip_h:
            img = np.fliplr(img)
        if flip_v:
            img = np.flipud(img)
        if flip_diag:
            img = np.transpose(img)
            #img = np.flipud(img)
        return img

    def acquireImage(self, **kwargs) -> int: # to delete?
        """Acquire image through EMMENU and store in the Image Manager Returns
        the image index."""
        print('camera_emmenu.py line 411 now!')
        self._vp.AcquireAndDisplayImage()
        return self.get_image_index()

    def set_image_index(self, index: int) -> None:
        """Change the currently selected buffer by the image index Note that
        the interface here is 0-indexed, whereas the image manager is 1-indexed
        (FIXME)"""
        self._vp.IndexInDirectory = index

    def get_image_index(self) -> int:
        """Retrieve the index of the currently selected buffer, 0-indexed."""
        return self._vp.IndexInDirectory

    def get_next_empty_image_index(self) -> int:
        """Get the next empty buffer in the image manager, 0-indexed."""
        i = self.get_image_index()
        while not self._immgr.ImageEmpty(self.drc_index, i):
            i += 1
        return i

    def stop_record(self) -> int:
        i = self.get_image_index()
        print(f'Stop recording (Image index={i})')
        self._vp.StopRecorder()
        self._recording = False
        return i

    def start_record(self) -> int:
        i = self.get_image_index()
        print(f'Start recording (Image index={i})')
        self._vp.StartRecorder()
        self._recording = True
        return i

    def stop_liveview(self) -> None:
        print('Stop live view')
        self._vp.StopContinuous()
        self._recording = False
        # StopRecorder normally defaults to top directory
        self._vp.DirectoryHandle = self.drc_index

    def start_liveview(self, delay: float = 3.0) -> None:
        print('Start live view')
        try:
            self._vp.StartContinuous()
        except comtypes.COMError as e:
            print(f'{e.details[1]}: {e.details[0]}')
        else:
            # sleep for a few seconds to ensure live view is running
            time.sleep(delay)

    def set_exposure(self, exposure_time: int) -> None:
        """Set exposure time in ms.

        It will be set to the lowest allowed value by EMMenu if the
        given exposure time is too low.
        """
        self._vp.ExposureTime = exposure_time

    def get_exposure(self) -> int:
        """Return exposure time in ms."""
        return self._vp.ExposureTime

    def set_autoincrement(self, toggle: bool) -> None:
        """Tell EMMENU to autoincrement the index number (True/False)"""
        if toggle:
            self._vp.AutoIncrement = 1
        else:
            self._vp.AutoIncrement = 0

    def get_timestamps(self, start_index: int, end_index: int) -> list:
        """Get timestamps in seconds for given image index range."""
        drc_index = self.drc_index
        timestamps = []
        for i, image_index in enumerate(range(start_index, end_index + 1)):
            p = self.getImageByIndex(image_index, drc_index)
            t = p.EMVector.lImgCreationTime
            timestamps.append(t)
        return timestamps

    def releaseConnection(self) -> None:
        """Release the connection to the camera."""
        self.stop_liveview()

        self._vp.DirectoryHandle = self.top_drc_index
        self._vp.SetCaption('Image')
        self.set_image_index(0)
        # self._immgr.DeleteDirectory(self.drc_index)  # bugged in EMMENU 5.0.9.0/5.0.10.0, FIXME later

        msg = f'Connection to camera `{self.getCameraName()}` ({self.name}) released'
        print(msg)

        comtypes.CoUninitialize()
    def release_connection(self): self.releaseConnection()

    def GetCurrentCameraConfiguration(self) -> CameraConfiguration:
        c = CameraConfiguration(speed=CameraSpeed(self.cameraGPU.LParam[self.CAMCLIB.cpLLSCXAmplifier]),
                                dynamic=CameraDynamic(self.cameraGPU.LParam[self.CAMCLIB.cpCurrentGainIndex]),
                                readoutMode=CameraReadoutMode(self.cameraGPU.LParam[self.CAMCLIB.cpReadoutMode]))
        return c


    def _readSingleImage(self, cameraFormat: CameraFormat):
        return self._memReader.GetLastImage(cameraFormat.dimension)

        ## from here there is the real data acquisition that should stay in another file! #### maybe i will do it in future?

    def acquire_series_images(self, exposure_time: int, binning: int, processing: str, buffer_size: int, stop_signal, display = False):
        print("not implemented yet")
        pass

    def prepare_acquisition_cRED_data(self, camera, binning, exposure, buffer_size,
                                      FPS_devider=1):  # before there was also camera
        """"    camera = XF416 or XF416R,
                binning = one of the available binning for the choosen camera,
                exposure = choosen exposure,
                buffer_size = dimension of the stack where saving the output images"""

        if camera == "XF416" or camera == "xf416":
            self.cam_format = CameraFormat.GetDefault(CameraType.XF416)
        elif camera == "XF416R" or camera == "xf416r":
            self.cam_format = CameraFormat.GetDefault(CameraType.XF416R)

        self.exposure = exposure
        self.buffer_size = buffer_size
        self.binning = int(binning)
        # copy original size 4k by 4k
        # self.original_cam_format_x = self.cam_format.dimension.x
        # self.original_cam_format_y = self.cam_format.dimension.y
        self.original_cam_format_x = 4096
        self.original_cam_format_y = 4096
        # rescale by binning
        self.cam_format.dimension.x = int(self.original_cam_format_x / self.binning)
        self.cam_format.dimension.y = int(self.original_cam_format_y / self.binning)
        ################################################################################################################
        # print("cam parameters:\n", "self.original_cam_format(x,y):", (self.original_cam_format_x, self.original_cam_format_y),
        #       "\nafter binning:", (self.cam_format.dimension.x, self.cam_format.dimension.y),
        #       "\nself.binning:", self.binning, "binning:", binning)

        self.buffer = np.zeros((self.buffer_size, self.cam_format.dimension.x, self.cam_format.dimension.y),
                               dtype=np.uint16)
        self.buffer_1 = np.zeros((self.buffer_size, self.cam_format.dimension.x, self.cam_format.dimension.y),
                                 dtype=np.uint16)
        self.zero_buffer = np.zeros((self.cam_format.dimension.x, self.cam_format.dimension.y), dtype=np.uint16)

        # response of the XF416R considering the exposure and the readoutime in rolling_shutter_mode
        # self.FPS_exp = 1.018*(self.exposure**-1.057)
        self.FPS_exp = (678.62 * (self.exposure ** -0.943)) / FPS_devider

        self.sleeping = 1.018 * (self.FPS_exp ** -1.057)
        print("ready to acquire cRED data sleep: ", self.sleeping)
        print("FPS devider: ", FPS_devider)

        self.stop_liveview()
        self.set_exposure(self.exposure)
        self.start_liveview()
        print('Live mode on, Ready for the collection...')


    def acquisition_cRED_data(self, stage_thread=None, timer = None, event = None, stop_event = None):
        # need to be moved in the acquisition section
        self.i = 0
        self.stopper = False

        if timer == None:
            self.t1 = time.monotonic_ns()
        else:
            self.ref_timings = {}
            self.t1 = time.monotonic_ns()
            self.ref_time = timer
            self.ref_timings["start_t1_time"] = self.t1
            self.ref_timings["start_ref_time"] = self.ref_time

        #camera_delay = self.table["Camera_delay"]
        self.stage_thread = stage_thread
        if timer != None:
            self.ref_timings["start_stage_thread_time"] = time.monotonic_ns()
        self.stage_thread.start()

        #time.sleep(camera_delay)
        # acquisition loop in self.img passing from the camera buffer
        self.timings = []
        ################################################################################################################
        self.i_ = 0
        try:
            self.buffer_1_used = False
            print("using buffer")
            answer = tkinter.messagebox.askyesno("continuous rotation data acquisition", "ready! press Yes when ready")
            if answer == False:
                print("data acquisition aborted by user")
                self.instance_gui.abort_data_acquisition()
                stop_event.set()
                event.set()
                self.stop_liveview()
                #self.stopper = True # stop camera
                return "aborted"
            # event set decide when starting the rotation and acquisition!
            event.set()
            while self.stopper == False:
                # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.tx = (time.monotonic_ns() - self.t1) / 1000000000
                self.img = self._memReader.GetLastImage(self.cam_format.dimension)
                print("cam grab")
                # 7.01 ms ± 159 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.buffer[self.i, :, :] = self.img
                # new line for streaming
                if self.is_cam_streaming():
                    self.image_to_streaming(self.img)

                self.timings.append(self.tx)
                time.sleep(self.sleeping)
                self.i += 1
                if self.stage_thread.is_alive() == False:
                    self.stopper = True
        except Exception as err:
            print(err)
            print("using buffer_1")
            self.buffer_1_used = True
            while self.stopper == False:
                # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.tx = (time.monotonic_ns() - self.t1) / 1000000000
                self.img = self._memReader.GetLastImage(self.cam_format.dimension)
                print("cam grab")
                # 7.01 ms ± 159 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.buffer_1[self.i_, :, :] = self.img
                # new line for streaming
                if self.is_cam_streaming():
                    self.image_to_streaming(self.img)

                self.timings.append(self.tx)
                time.sleep(self.sleeping)
                self.i_ += 1
                if self.stage_thread.is_alive() == False:
                    self.stopper = True

        self.t2 = (time.monotonic_ns() - self.t1) / 1000000000
        if timer != None:
            self.ref_timings["end_acq_cred_time"] = time.monotonic_ns()

        self.stop_liveview()

        self.support = []
        self.support_1 = []

        i = 0
        for images, images_1 in zip(self.buffer, self.buffer_1):
            if np.array_equal(images, self.zero_buffer) == True:
                self.support.append(i)
            if np.array_equal(images_1, self.zero_buffer) == True:
                self.support_1.append(i)
            i += 1


        # print("support list: ", len(self.support),self.support)
        if len(self.support) > 0 or len(self.support_1) > 0:
            print("shape buffer before: ", np.shape(self.buffer), "shape buffer_1 before: ", np.shape(self.buffer_1))
            if len(self.support) > 0:
                try:
                    print("buffer resize parameters: ", self.support[0], np.shape(self.buffer)[1], np.shape(self.buffer)[2])
                    self.buffer = np.resize(self.buffer, (self.support[0], np.shape(self.buffer)[1], np.shape(self.buffer)[2]))
                except Exception as err:
                    pass
            if len(self.support_1) > 0:
                try:
                    print("buffer_1 resize parameters: ", self.support_1[0], np.shape(self.buffer)[1], np.shape(self.buffer)[2])
                    self.buffer_1 = np.resize(self.buffer_1, (self.support_1[0], np.shape(self.buffer_1)[1], np.shape(self.buffer_1)[2]))
                except Exception as err:
                    pass
            print("shape buffer after resize: ", np.shape(self.buffer))
            print("shape buffer_1 after resize: ", np.shape(self.buffer_1))

        self.buffer_size = np.shape(self.buffer)[0] + np.shape(self.buffer_1)[0]

        self.FPS = self.buffer_size / self.t2

        # need to be moved in the acquisition section
        print("Finished extraction in : ", self.t2)
        print("Frames collected: ", self.buffer_size)
        print("FPS: ", self.FPS)

        self.result_param = [self.t2, self.FPS, self.buffer_size]

        # reset original size camera
        self.cam_format.dimension.x = self.original_cam_format_x
        self.cam_format.dimension.y = self.original_cam_format_y
        # reset exposure to 50 ms
        self.set_exposure(50)
        return self.result_param


    def save_cRED_data(self, savingpath):
        self.ii = 1
        self.saving_dir = str(savingpath)
        self.current_dir = os.getcwd()
        print("saving datacollection in: ", self.saving_dir)
        if len(str(self.buffer.shape[0]+self.buffer_1.shape[0])) <= 4:
            self.name_zeros = 4
        else:
            self.name_zeros = len(str(self.buffer.shape[0]+self.buffer_1.shape[0]))


        for image in self.buffer:
            if not self.current_dir == self.saving_dir:
                os.chdir(self.saving_dir)
            else:
                pass

            # format tiff uncompressed data
            self.image_name = str('%s.tif' % (format(self.ii, '.0f').rjust(self.name_zeros, '0')))
            # print("saving: ", self.image_name)
            imageio.imwrite(self.image_name, image)
            self.ii += 1

        if self.buffer_1_used == True:
            for image in self.buffer_1:
                if not self.current_dir == self.saving_dir:
                    os.chdir(self.saving_dir)
                else:
                    pass

                # format tiff uncompressed data
                self.image_name = str('%s.tif' % (format(self.ii, '.0f').rjust(self.name_zeros, '0')))
                # print("saving: ", self.image_name)
                imageio.imwrite(self.image_name, image)
                self.ii += 1

        os.chdir(self.current_dir)


    # to add and thread in the acquisition module
    # prepare_acquisition_cRED_data(self, camera, binning, exposure, buffer_size)
    # acquisition_cRED_data(self)
    # save_cRED_data(self, savingpath)
    def get_camera_characteristic(self):
        pixelsize = 15.5
        max_image_pixels = 4096
        print("camera:", self.name)
        print("physical camera pixelsize in um", pixelsize)
        print("max image pixels^2", max_image_pixels)
        return pixelsize, max_image_pixels

    def load_calibration_table(self):
        cwd = os.getcwd()
        path = cwd+os.sep+r"adaptor/camera/lookup_table/xf416r_table.txt"
        # table = {"IMAGING", {},
        #          "DIFFRACTION", {}}
        with open(path, 'r') as file:
            self.table = json.load(file)
        self.table["bottom_mounted"] = eval(self.table["bottom_mounted"])
        self.table["cam_flip_h"] = eval(self.table["cam_flip_h"])
        self.table["cam_flip_v"] = eval(self.table["cam_flip_v"])
        self.table["cam_flip_diag"] = eval(self.table["cam_flip_diag"])
        self.table["streamable"] = eval(self.table["streamable"])
        return self.table


    def is_cam_streaming(self):
        return self.table["streamable"]

    def is_cam_bottom_mounted(self):
        return self.table["bottom_mounted"]

    def enable_streaming(self):
        if self.is_cam_streaming():
            from ....server_sharedmemory_stream import camera_server
            self.server = camera_server(shared_mem_name = "camera_stream", frame_shape = (1024, 1024))
            return self.server
        else:
            return False

    def image_to_streaming(self, img):
        if self.server.frame_shape != img.shape:
            img = cv2.resize(img, self.server.frame_shape)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.server.update_buffer(img)


if __name__ == '__main__':
    cam = Cam_xf416r() # to replace CameraEMMENUGPU()
    #cam = CameraEMMENUGPU()
    #cam_camc = camc_GPU() now this is inside Cam_xf416r
    #cam_format = CameraFormat.GetDefault(CameraType.XF416R)

