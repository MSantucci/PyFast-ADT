# interface to simulate a fake camera for debugging purposes.

try:
    from .adaptor_cam import Cam_base
except:
    from adaptor_cam import Cam_base

import json
import imageio
import time
import numpy as np
import cv2
import os
import atexit
import matplotlib.pyplot as plt
import tkinter
import tkinter.messagebox
import socket
from typing import Any

class Cam_simulator(Cam_base):
    """Fake Camera interface for debugging purposes."""

    def __init__(self, instance_gui=None):
        super().__init__()
        self.name = None  # name of the camera
        self.exposure = None  # ms
        self.x = None
        self.y = None
        self.processing = None
        self.delay = None  # seconds
        self.binning = None
        self.buffer_size = None
        self.stop_signal = None
        self.buffer = None
        self.table = None
        self.timings = []

        # connecting the camera and setting up the release on disconnection


    def connect(self):
        '''' connection with the device '''
        pass


    def release_connection(self):
        '''' release the connection with the device '''
        pass


    def set_exposure(self, exposure_time: int):
        '''' set the exposure time in ms for the camera '''
        pass


    def get_exposure(self):
        '''' get the exposure time in ms for the camera '''
        pass


    def start_liveview(self, delay: float):
        '''' start the live view of the camera '''
        pass


    def stop_liveview(self):
        '''' stop the live view of the camera '''
        pass


    def set_binning(self, binning: int):
        '''' set the binning of the camera, common parameters are 1, 2, 4, 8 '''
        pass


    def get_binning(self):
        '''' get the binning of the camera '''
        pass


    def acquire_image(self, exposure_time: int, binning: int, processing: str):
        """Acquire image through its adaptor and return it as np.array."""
        # set_exposure(exposure_time)
        # set_binning(binning)
        # set_processing(processing)
        pass


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


    def set_processing(self, processing: str):
        '''' set the processing of the camera,
        processing = "Unprocessed, Background subtracted, Gain normalized"'''
        pass


    def get_processing(self):
        '''' get the processing typeof the camera '''
        # xf416r are ('Uncorrected', 'Dark subtracted', None, 'Gain corrected')[cfg.FlatMode]  # str, 2 undefined
        pass


    def acquire_series_images(self, exposure_time: int, binning: int, processing: str, buffer_size: int, stop_signal,
                              display=False):
        pass


    def prepare_acquisition_cRED_data(self, camera: str, binning: int, exposure: int, buffer_size, FPS_devider=1):
        pass


    def acquisition_cRED_data(self, stage_thread=None):
        ''' Acquire images into the buffer up to the thread is alive, usually the stage thread is passed for cRED experiments '''
        pass


    def save_cRED_data(self, savingpath: str):
        pass


    def get_camera_characteristic(self):
        pixelsize = 15.5
        max_image_pixels = 4096
        print("camera:", self.name)
        print("physical camera pixelsize in um", pixelsize)
        print("max image pixels^2", max_image_pixels)
        return pixelsize, max_image_pixels


    def load_calibration_table(self):
        cwd = os.getcwd()
        path = cwd + os.sep + r"adaptor/camera/lookup_table/xf416r_table.txt"
        # table = {"IMAGING", {},
        #          "DIFFRACTION", {}}
        with open(path, 'r') as file:
            self.table = json.load(file)
        self.table["bottom_mounted"] = eval(self.table["bottom_mounted"])
        self.table["cam_flip_h"] = eval(self.table["cam_flip_h"])
        self.table["cam_flip_v"] = eval(self.table["cam_flip_v"])
        self.table["cam_flip_diag"] = eval(self.table["cam_flip_diag"])
        self.table["streamable"] = False
        return self.table


    def is_cam_streaming(self):
        """True is the camera have a live mode where you can retrieve the images from the memory like the xf416r, otherwise False like the timepix1"""
        return self.table["streamable"]


    def is_cam_bottom_mounted(self):
        """True if the camera is mounted on the bottom of the microscope, otherwise False"""
        return self.table["bottom_mounted"]

    def enable_streaming(self):
        # check line 153, self.table["streamable"] = False or True
        if self.is_cam_streaming():
            from server_sharedmemory_stream import camera_server
            self.server = camera_server(shared_mem_name = "camera_stream", frame_shape = (1024, 1024))
            # enable a process that send continuously the images to the shared memory for debug purposes
            from multiprocessing import Process
            p1 = Process(target=self.server.start_simulation)
            atexit.register(p1.terminate)
            p1.start()
            return self.server
        else:
            return False

    def image_to_streaming(self, img):
        if self.server.frame_shape != img.shape:
            img = cv2.resize(img, self.server.frame_shape)

        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.server.update_buffer(img)

if __name__ == '__main__':
    cam = Cam_simulator()

