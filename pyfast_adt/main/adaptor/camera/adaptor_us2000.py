from .adaptor_cam import Cam_base
import json
import numpy as np
import atexit
import logging
import time
from pathlib import Path
# import comtypes.client
import os, cv2
import imageio
import temscript
import matplotlib.pyplot as plt
import tkinter

class Cam_us2000(Cam_base):
    """Software interface for the Gatan us2000 camera passing through temscript interface. in general gatan cameras
    can be used using temscript if they are enabled in Tecnai User Interface"""
    def __init__(self, instance_gui = None):
        """Initialize camera module."""
        super().__init__()
        # try:
        #     comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
        # except OSError:
        #     comtypes.CoInitialize()

        #self.name = name  # name of the camera
        #self.drc_name = drc_name  # name of the folder where to save the data
        self.name = None  # name of the camera
        self.exposure = None  # ms
        self.x = None
        self.y = None
        self.processing = None
        self.delay = None  # seconds
        self.binning = None
        self.buffer_size = None
        self.buffer = None
        self.stop_signal = None
        self.cam = None         # name of the camera in temscript
        self.tem = None         # entry for temscript
        self.table = None   # table with the camera parameters
        self.dark_bin1 = None
        self.dark_bin2 = None
        self.dark_bin4 = None
        self.dark_bin8 = None
        self.flatfield_bin1 = None
        self.flatfield_bin2 = None
        self.flatfield_bin4 = None
        self.flatfield_bin8 = None
        self.last_exposure = 0
        self.load_calibration_table()
        self.instance_gui = instance_gui
        #connecting the camera and setting up the release on disconnection
        self.connect()
        #atexit.register(self.releaseConnection)
    def connect(self):
        """Connect to the camera."""
        try:
            print("connecting to us2000 through temscript")
            self.tem = temscript.RemoteMicroscope((self.table["ip"][0], self.table["ip"][1]))
            self.name = self.initialize_detector()
        except Exception as err:
            print('Error in the initialization of the camera, check that digital micrograph and tia are running\n', err)
    def initialize_detector(self):
        self.name = "BM-UltraScan"  # spirit otherwise the us400 should be "CCD"
        self.cam = self.tem.get_cameras()
        self.load_flatfield()
        return self.name

    def load_flatfield(self):
        if (self.flatfield_bin1 is None or not np.any(self.flatfield_bin1)
                and self.flatfield_bin2 is None or not np.any(self.flatfield_bin2)
                and self.flatfield_bin4 is None or not np.any(self.flatfield_bin4)
                and self.flatfield_bin8 is None or not np.any(self.flatfield_bin8)):

            try:
                path = os.path.split(os.getcwd())[0] + os.sep + r"flatfield/"
                self.flatfield_bin1 = cv2.imread(path + "flatfield_bin1.tif", flags=-1)
                self.flatfield_bin2 = cv2.imread(path + "flatfield_bin2.tif", flags=-1)
                self.flatfield_bin4 = cv2.imread(path + "flatfield_bin4.tif", flags=-1)
                self.flatfield_bin8 = cv2.imread(path + "flatfield_bin8.tif", flags=-1)
            except Exception as err:
                print("error in the loading of the flatfields", err)

    def acquire_and_save_flatfield(self, exposure_time, binning):
        img = self.acquire_image_and_show(exposure_time, binning, processing="Unprocessed")
        img = np.clip(np.round(img), 0, np.iinfo(np.uint16).max)
        img = img.astype(np.uint16)

        path = os.getcwd()
        os.chdir(os.path.split(path)[0] + os.sep + r"flatfield/")
        # format tiff uncompressed data
        self.image_name = str('%s.tif' % ("flatfield_bin" + str(binning)))
        imageio.imwrite(self.image_name, img)
        os.chdir(path)
    def release_connection(self): pass
    def get_binning(self):
        """ get binning value like a conventional CCD camera"""
        cam_param = self.tem.get_camera_param(self.name)
        return cam_param["binning"]

    def set_binning(self, binning: int):
        """ set binning value like a conventional CCD camera"""
        cam_param = self.tem.get_camera_param(self.name)
        cam_param["binning"] = binning
        self.tem.set_camera_param(self.name, cam_param)

    def get_image_size(self):
        """ in temscript image_size control the readout area parameter in the CCD\TV Camera tab,
                from temscript is possible from a 4k**2 cam to reach a 1k**2 cam maximum.
                    image_size == "FULL" (2k), "HALF" (1k), "QUARTER" (512) """
        cam_param = self.tem.get_camera_param(self.name)
        return cam_param["image_size"]

    def set_processing(self, processing: str):
        """ set processing, UNPROCESSED = 0, DEFAULT = 1"""
        cam_param = self.tem.get_camera_param(self.name)
        if processing == "Unprocessed":
            cam_param["correction"] = "UNPROCESSED"
        elif processing == "Gain normalized" or processing == "Background subtracted":
            cam_param["correction"] = "DEFAULT"
        self.tem.set_camera_param(self.name, cam_param)
    def get_processing(self):
        """ get binning value like a conventional CCD camera"""
        cam_param = self.tem.get_camera_param(self.name)
        return cam_param["correction"]

    def acquire_image(self, exposure_time: int, binning: int, processing: str):
        self.set_binning(binning)
        self.set_exposure(int(exposure_time))
        self.set_processing("Unprocessed")
        if self.last_exposure != exposure_time:
            self.dark_bin1 = None
            self.dark_bin2 = None
            self.dark_bin4 = None
            self.dark_bin8 = None
            self.last_exposure = exposure_time
        if binning == 1 and (self.dark_bin1 is None or not np.any(self.dark_bin1)):
            self.tem.set_beam_blanked(True)
            self.dark_bin1 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 2 and (self.dark_bin2 is None or not np.any(self.dark_bin2)):
            self.tem.set_beam_blanked(True)
            self.dark_bin2 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 4 and (self.dark_bin4 is None or not np.any(self.dark_bin4)):
            self.tem.set_beam_blanked(True)
            self.dark_bin4 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 8 and (self.dark_bin8 is None or not np.any(self.dark_bin8)):
            self.tem.set_beam_blanked(True)
            self.dark_bin8 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)

        img = self.tem.acquire(self.name)[self.name]
        if processing == "Background subtracted" or processing == "Gain normalized":
            if binning == 1:
                img = img - self.dark_bin1
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin1 - self.dark_bin1), 0, np.iinfo(np.uint16).max)
            elif binning == 2:
                img = img - self.dark_bin2
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin2 - self.dark_bin2), 0, np.iinfo(np.uint16).max)
            elif binning == 4:
                img = img - self.dark_bin4
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin4 - self.dark_bin4), 0, np.iinfo(np.uint16).max)
            elif binning == 8:
                img = img - self.dark_bin8
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin8 - self.dark_bin8), 0, np.iinfo(np.uint16).max)

            img = np.round(img)
            img = np.clip(img, 0, np.iinfo(np.uint16).max)
            img = img.astype(np.uint16)

        img = self.rotate_img(img)
        if processing == "Gain normalized":
            gain = gain.astype(np.uint16)/np.max(gain)
            img = gain*img
            img = np.round(img)
            img = np.clip(img, 0, np.iinfo(np.uint16).max)
            img = img.astype(np.uint16)

        return img

    def acquire_image_and_show(self, exposure_time: int, binning: int, processing: str):
        self.set_binning(binning)
        self.set_exposure(int(exposure_time))
        self.set_processing("Unprocessed")
        if self.last_exposure != exposure_time:
            self.dark_bin1 = None
            self.dark_bin2 = None
            self.dark_bin4 = None
            self.dark_bin8 = None
            self.last_exposure = exposure_time
        if binning == 1 and (self.dark_bin1 is None or not np.any(self.dark_bin1)):
            self.tem.set_beam_blanked(True)
            self.dark_bin1 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 2 and (self.dark_bin2 is None or not np.any(self.dark_bin2)):
            self.tem.set_beam_blanked(True)
            self.dark_bin2 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 4 and (self.dark_bin4 is None or not np.any(self.dark_bin4)):
            self.tem.set_beam_blanked(True)
            self.dark_bin4 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 8 and (self.dark_bin8 is None or not np.any(self.dark_bin8)):
            self.tem.set_beam_blanked(True)
            self.dark_bin8 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)

        img = self.tem.acquire(self.name)[self.name]
        if processing == "Background subtracted" or processing == "Gain normalized":
            if binning == 1:
                img = img - self.dark_bin1
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin1 - self.dark_bin1), 0, np.iinfo(np.uint16).max)
            elif binning == 2:
                img = img - self.dark_bin2
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin2 - self.dark_bin2), 0, np.iinfo(np.uint16).max)
            elif binning == 4:
                img = img -self.dark_bin4
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin4 - self.dark_bin4), 0, np.iinfo(np.uint16).max)
            elif binning == 8:
                img = img - self.dark_bin8
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin8 - self.dark_bin8), 0, np.iinfo(np.uint16).max)

            img = np.round(img)
            img = np.clip(img, 0, np.iinfo(np.uint16).max)
            img = img.astype(np.uint16)

        img = self.rotate_img(img)
        if processing == "Gain normalized":
            gain = gain.astype(np.uint16)/np.max(gain)
            img = gain*img
            img = np.round(img)
            img = np.clip(img, 0, np.iinfo(np.uint16).max)
            img = img.astype(np.uint16)

        try:
            plt.imshow(img, vmax=np.max(img))
            plt.show()
        except Exception as err:
            print(err)
            plt.imshow(img)
            plt.show()
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

    def stop_liveview(self) -> None:
        print('Stop live view')
        pass

    def start_liveview(self, delay: float = 3.0) -> None:
        print('Start live view')
        pass

    def set_exposure(self, exposure_time: int) -> None:
        """Set exposure time in ms."""
        cam_param = self.tem.get_camera_param(self.name)
        cam_param["exposure(s)"] = exposure_time/1000
        print("exposure to check:", cam_param["exposure(s)"])
        self.tem.set_camera_param(self.name, cam_param)
    def get_exposure(self) -> int:
        """Return exposure time in ms."""
        cam_param = self.tem.get_camera_param(self.name)
        return cam_param["exposure(s)"]*1000
    def acquire_series_images(self, exposure_time: int, binning: int, processing: str, buffer_size: int, stop_signal, display = False):
        print("not implemented yet")
        pass


    def prepare_acquisition_cRED_data(self, camera, binning, exposure, buffer_size, FPS_devider = 1):  # before there was also camera
        """"    camera = us4000,
                binning = one of the available binning for the choosen camera,
                exposure = choosen exposure,
                buffer_size = dimension of the stack where saving the output images"""

        self.exposure = exposure
        self.buffer_size = buffer_size
        self.binning = int(binning)

        self.dimension = int(self.get_camera_characteristic()[1]/self.binning)
        ################################################################################################################

        self.buffer = np.zeros((self.buffer_size, self.dimension, self.dimension), dtype=np.uint16)
        self.buffer_1 = np.zeros((self.buffer_size, self.dimension, self.dimension), dtype=np.uint16)
        self.zero_buffer = np.zeros((self.dimension, self.dimension), dtype=np.uint16)

        # response of the XF416R considering the exposure and the readoutime in rolling_shutter_mode
        # self.FPS_exp = 1.018*(self.exposure**-1.057)
        self.FPS_exp = (678.62 * (self.exposure ** -0.943))/FPS_devider

        self.sleeping = 1.018 * (self.FPS_exp ** -1.057)
        print("ready to acquire cRED data sleep: ", self.sleeping)
        print("FPS devider: ", FPS_devider)

        self.stop_liveview()
        self.set_exposure(self.exposure)
        self.prepare_acquire_image_cred(self.binning, self.exposure)
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

        self.stage_thread = stage_thread
        if timer != None:
            self.ref_timings["start_stage_thread_time"] = time.monotonic_ns()
        self.stage_thread.start()

        self.timings = []

        # acquisition loop in self.img passing from the camera buffer

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
                # self.stopper = True # stop camera
                return "aborted"
            event.set()
            while self.stopper == False:
                # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.tx = (time.monotonic_ns() - self.t1) / 1000000000
                self.img = self.acquire_image_cred()
                # 7.01 ms ± 159 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.buffer[self.i, :, :] = self.img

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
                self.img = self.acquire_image_cred()
                # 7.01 ms ± 159 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.buffer_1[self.i_, :, :] = self.img

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

        return self.result_param


    def save_cRED_data(self, savingpath, processing = "Unprocessed"):
        self.ii = 1
        self.saving_dir = str(savingpath)
        self.current_dir = os.getcwd()
        print("saving datacollection in: ", self.saving_dir)
        if len(str(self.buffer.shape[0]+self.buffer_1.shape[0])) <= 4:
            self.name_zeros = 4
        else:
            self.name_zeros = len(str(self.buffer.shape[0]+self.buffer_1.shape[0]))


        for image in self.buffer:
            image = self.elaborate_image_cred(image, self.binning, processing)
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
                image = self.elaborate_image_cred(image, self.binning, processing)
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
        cam_param = self.tem.get_cameras()[self.name]
        pixelsize = cam_param["pixel_size(um)"]
        max_image_pixels = cam_param["width"]
        print("camera:", self.name)
        print("physical camera pixelsize in um", pixelsize)
        print("max image pixels^2", max_image_pixels)
        return pixelsize, max_image_pixels

    def load_calibration_table(self):
        cwd = os.getcwd()
        path = cwd+os.sep+r"adaptor/camera/lookup_table/us2000_table.txt"
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
        return False
    def is_cam_bottom_mounted(self):
        return self.table["bottom_mounted"]

    def prepare_acquire_image_cred(self, binning, exposure_time):
        self.set_binning(binning)
        self.set_exposure(int(exposure_time))
        self.set_processing("Unprocessed")
        if self.last_exposure != exposure_time:
            self.dark_bin1 = None
            self.dark_bin2 = None
            self.dark_bin4 = None
            self.dark_bin8 = None
            self.last_exposure = exposure_time
        if binning == 1 and (self.dark_bin1 is None or not np.any(self.dark_bin1)):
            self.tem.set_beam_blanked(True)
            self.dark_bin1 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 2 and (self.dark_bin2 is None or not np.any(self.dark_bin2)):
            self.tem.set_beam_blanked(True)
            self.dark_bin2 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 4 and (self.dark_bin4 is None or not np.any(self.dark_bin4)):
            self.tem.set_beam_blanked(True)
            self.dark_bin4 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)
        elif binning == 8 and (self.dark_bin8 is None or not np.any(self.dark_bin8)):
            self.tem.set_beam_blanked(True)
            self.dark_bin8 = self.tem.acquire(self.name)[self.name]
            self.tem.set_beam_blanked(False)

    def acquire_image_cred(self):
        img = self.tem.acquire(self.name)[self.name]
        return img
    def elaborate_image_cred(self, img, binning: int, processing: str):
        if processing == "Background subtracted" or processing == "Gain normalized":
            if binning == 1:
                img = img - self.dark_bin1
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin1 - self.dark_bin1), 0, np.iinfo(np.uint16).max)
            elif binning == 2:
                img = img - self.dark_bin2
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin2 - self.dark_bin2), 0, np.iinfo(np.uint16).max)
            elif binning == 4:
                img = img - self.dark_bin4
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin4 - self.dark_bin4), 0, np.iinfo(np.uint16).max)
            elif binning == 8:
                img = img - self.dark_bin8
                if processing == "Gain normalized":
                    gain = np.clip(np.round(self.flatfield_bin8 - self.dark_bin8), 0, np.iinfo(np.uint16).max)

            img = np.round(img)
            img = np.clip(img, 0, np.iinfo(np.uint16).max)
            img = img.astype(np.uint16)

        img = self.rotate_img(img)
        if processing == "Gain normalized":
            gain = gain.astype(np.uint16)/np.max(gain)
            img = gain*img
            img = np.round(img)
            img = np.clip(img, 0, np.iinfo(np.uint16).max)
            img = img.astype(np.uint16)

        return img

if __name__ == '__main__':
    cam = Cam_us2000() # to replace CameraEMMENUGPU()


