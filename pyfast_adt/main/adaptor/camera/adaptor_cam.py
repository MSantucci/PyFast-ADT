from abc import ABC, abstractmethod
import numpy as np
# base Class for cameras, all the specific adaptor are child of this class.
# here are defined all the methods that the camera should have to work with FastADT.
class Cam_base(ABC):
    def __init__(self):
        self.name = None # name of the camera
        self.exposure = None # ms
        self.x = None
        self.y = None
        self.processing = None
        self.delay = None # seconds
        self.binning = None
        self.buffer_size = None
        self.stop_signal = None
        self.buffer = None
        self.table = None
        self.timings = []

        #connecting the camera and setting up the release on disconnection

    @abstractmethod
    def connect(self):
        '''' connection with the device '''
        pass
    @abstractmethod
    def release_connection(self):
        '''' release the connection with the device '''
        pass

    @abstractmethod
    def set_exposure(self, exposure_time: int):
        '''' set the exposure time in ms for the camera '''
        pass

    @abstractmethod
    def get_exposure(self):
        '''' get the exposure time in ms for the camera '''
        pass

    @abstractmethod
    def start_liveview(self, delay: float):
        '''' start the live view of the camera '''
        pass

    @abstractmethod
    def stop_liveview(self):
        '''' stop the live view of the camera '''
        pass

    @abstractmethod
    def set_binning(self, binning: int):
        '''' set the binning of the camera, common parameters are 1, 2, 4, 8 '''
        pass

    @abstractmethod
    def get_binning(self):
        '''' get the binning of the camera '''
        pass

    @abstractmethod
    def acquire_image(self, exposure_time: int, binning: int, processing: str):
        """Acquire image through its adaptor and return it as np.array."""
        # set_exposure(exposure_time)
        # set_binning(binning)
        # set_processing(processing)
        pass

    @abstractmethod
    def rotate_img(self, img, times=None, flip_h=None, flip_v=None):
        times = times if times is not None else int(self.table["cam_rotate90x"])
        flip_h = flip_h if flip_h is not None else self.table["cam_flip_h"]
        flip_v = flip_v if flip_v is not None else self.table["cam_flip_v"]

        if times is not None and times != 0:
            img = np.rot90(img, times)
        if flip_h:
            img = np.fliplr(img)
        if flip_v:
            img = np.flipud(img)
        return img

    @abstractmethod
    def set_processing(self, processing: str):
        '''' set the processing of the camera,
        processing = "Unprocessed, Background subtracted, Gain normalized"'''
        pass

    @abstractmethod
    def get_processing(self):
        '''' get the processing typeof the camera '''
        # xf416r are ('Uncorrected', 'Dark subtracted', None, 'Gain corrected')[cfg.FlatMode]  # str, 2 undefined
        pass

    @abstractmethod
    def acquire_series_images(self, exposure_time: int, binning: int, processing: str, buffer_size: int, stop_signal, display = False):
        pass

    @abstractmethod
    def prepare_acquisition_cRED_data(self, camera:str, binning:int, exposure:int, buffer_size, FPS_devider = 1):
        pass

    @abstractmethod
    def acquisition_cRED_data(self, stage_thread=None):
        ''' Acquire images into the buffer up to the thread is alive, usually the stage thread is passed for cRED experiments '''
        pass

    @abstractmethod
    def save_cRED_data(self, savingpath:str):
        pass

    @abstractmethod
    def get_camera_characteristic(self):
        pass

    @abstractmethod
    def load_calibration_table(self):
        pass

    @abstractmethod
    def is_cam_streaming(self):
        """True is the camera have a live mode where you can retrieve the images from the memory like the xf416r, otherwise False like the timepix1"""
        pass
    @abstractmethod
    def is_cam_bottom_mounted(self):
        """True if the camera is mounted on the bottom of the microscope, otherwise False"""
        pass





