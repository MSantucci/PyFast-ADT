from .adaptor_cam import Cam_base
import json
import numpy as np
import atexit
import logging
import time
from pathlib import Path
import os, cv2
import imageio
from temscript import RemoteMicroscope
from ..microscope.temspy_socket import SocketServerClient
import tkinter
import matplotlib.pyplot as plt

class Cam_haadf(Cam_base):
    """Software interface for the F30 Fischione HAADF detector passing through temscript interface."""
    def __init__(self, cam_table = None, instance_gui = None):
        """Initialize camera module."""
        super().__init__()
        self.cam_table = cam_table
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
        self.table = None
        self.load_calibration_table()
        self.instance_gui = instance_gui
        #connecting the camera and setting up the release on disconnection
        self.connect()
        atexit.register(self.release_connection)
        self.client = SocketServerClient(mode='client', host=self.table["ip"][0], port=8083)
        self.client.start()
    def connect(self):
        """Connect to the camera."""
        try:
            print("connecting to temscript")
            self.tem = RemoteMicroscope((self.table["ip"][0], self.table["ip"][1] + 2))

            # base is using 8082 which is the port for the user beam shift
            self.haadf_name = self.initialize_detector()

        except Exception as err:
            print('Error in the initialization of the camera', err)

    def initialize_detector(self):
        self.haadf = self.tem.get_stem_detectors()
        self.haadf_name = list(self.haadf.keys())[0]
        self.haadf_rotation = self.tem.get_stem_rotation()
        return self.haadf_name
    def release_connection(self):
        pass
    def get_binning(self):
        """ in temscript binning control the Max frame parameter in the STEM Imaging Expert flap,
        so no magnification scaling is applied, like a conventional CCD camera"""
        haadf_param = self.tem.get_stem_acquisition_param()
        return haadf_param["binning"]
    def set_binning(self, binning: int):
        """ in temscript binning control the Max frame parameter in the STEM Imaging Expert flap,
                so no magnification scaling is applied, like a conventional CCD camera"""
        haadf_param = self.tem.get_stem_acquisition_param()
        haadf_param["binning"] = binning
        self.tem.set_stem_acquisition_param(haadf_param)
    def get_image_size(self):
        """ in temscript image_size control the frame size parameter in the STEM Imaging Expert flap,
                so a magnification scaling is applied because a smaller area is scanned.
                from temscript is possible from a 2k**2 haadf to reach a 512**2 haadf maximum.
                    image_size == "FULL" (2k), "HALF" (1k), "QUARTER" (512) """
        haadf_param = self.tem.get_stem_acquisition_param()
        return haadf_param["image_size"]
    def set_image_size(self, image_size: str):
        """image_size == "FULL"(2k), "HALF"(1k), "QUARTER"(512)"""
        haadf_param = self.tem.get_stem_acquisition_param()
        haadf_param["image_size"] = image_size
        self.tem.set_stem_acquisition_param(haadf_param)
    def set_processing(self, processing: str):
        print("not available for this camera")

    def get_processing(self):
        print("not available for this camera")

    def acquire_image(self, binning: int, exposure_time: int, image_size: str):
        self.set_binning(binning)
        self.set_exposure(exposure_time)
        self.set_image_size(image_size)

        img = self.tem.acquire(self.haadf_name)[self.haadf_name]
        min_value = np.min(img)
        max_value = np.max(img)
        img = ((img - min_value) / (max_value - min_value) * 65535).astype(np.uint16)
        img = self.rotate_img(img)
        return img

    def acquire_image_and_show(self, binning: int, exposure_time: int, image_size: str):
        self.set_binning(binning)
        self.set_exposure(exposure_time)
        self.set_image_size(image_size)

        img = self.tem.acquire(self.haadf_name)[self.haadf_name]
        min_value = np.min(img)
        max_value = np.max(img)
        img = ((img - min_value) / (max_value - min_value) * 65535).astype(np.uint16)
        img = self.rotate_img(img)
        plt.imshow(img)
        plt.show()
        return img

    def acquire_image_fast(self):
        img = self.tem.acquire(self.haadf_name)[self.haadf_name]
        min_value = np.min(img)
        max_value = np.max(img)
        img = ((img - min_value) / (max_value - min_value) * 65535).astype(np.uint16)
        img = self.rotate_img(img)
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
        print('not available')
        pass

    def start_liveview(self, delay: float = 3.0) -> None:
        print('not available')
        pass

    def set_exposure(self, exposure_time: int) -> None:
        """Set exposure time in us."""
        haadf_param = self.tem.get_stem_acquisition_param()
        haadf_param["dwell_time(s)"] = exposure_time*0.000001
        self.tem.set_stem_acquisition_param(haadf_param)

    def get_exposure(self) -> int:
        """Return exposure time in ms."""
        haadf_param = self.tem.get_stem_acquisition_param()
        return haadf_param["dwell_time(s)"]
    def acquire_series_images(self, exposure_time: int, binning: int, processing: str, buffer_size: int, stop_signal, display = False):
        print("not implemented yet")
        pass

    def prepare_acquisition_cRED_data(self, camera, binning, exposure, image_size, buffer_size, FPS_devider = 1):  # before there was also camera
        """"    camera = spirit_haadf,
                binning = one of the available binning for the choosen camera,
                exposure = choosen exposure,
                buffer_size = dimension of the stack where saving the output images"""
        # to change ####################################################################################
        self.exposure = exposure
        self.buffer_size = buffer_size
        self.binning = int(binning)
        self.image_size = image_size
        if self.image_size == "FULL":
            self.dimension = int(self.get_camera_characteristic()[1]/self.binning)
        elif self.image_size == "HALF":
            self.dimension = int(self.get_camera_characteristic()[1]/(self.binning*2))
        elif self.image_size == "QUARTER":
            self.dimension = int(self.get_camera_characteristic()[1]/(self.binning*4))
        ################################################################################################################

        self.buffer = np.zeros((self.buffer_size, self.dimension, self.dimension), dtype=np.uint16)
        self.buffer_1 = np.zeros((self.buffer_size, self.dimension, self.dimension), dtype=np.uint16)
        self.zero_buffer = np.zeros((self.dimension, self.dimension), dtype=np.uint16)

        # response of the haadf
        if self.image_size == "FULL":
            exposure_dwell = (self.exposure) * (2048 / self.binning) ** 2
        elif self.image_size == "HALF":
            exposure_dwell = (self.exposure) * (1024 / self.binning) ** 2
        elif self.image_size == "QUARTER":
            exposure_dwell = (self.exposure) * (516 / self.binning) ** 2
        if self.binning == 1:
            if self.image_size == "FULL":
                self.FPS_exp = (0.1791 * (exposure_dwell ** -0.425))/FPS_devider
            elif self.image_size == "HALF":
                self.FPS_exp = (0.3836 * (exposure_dwell ** -0.462))/FPS_devider
            elif self.image_size == "QUARTER":
                self.FPS_exp = (0.7672 * (exposure_dwell ** -0.462))/FPS_devider #### to do
        elif self.binning == 2:
            if self.image_size == "FULL":
                self.FPS_exp = (0.3391 * (exposure_dwell ** -0.437))/FPS_devider #### to do
            elif self.image_size == "HALF":
                self.FPS_exp = (0.6782 * (exposure_dwell ** -0.437))/FPS_devider
            elif self.image_size == "QUARTER":
                self.FPS_exp = (1.3564 * (exposure_dwell ** -0.437))/FPS_devider #### to do
        elif self.binning == 4:
            if self.image_size == "FULL":
                self.FPS_exp = (0.6409 * (exposure_dwell ** -0.189))/FPS_devider
            elif self.image_size == "HALF":
                self.FPS_exp = (1.3651 * (exposure_dwell ** -0.189))/FPS_devider
            elif self.image_size == "QUARTER":
                self.FPS_exp = (2.7302 * (exposure_dwell ** -0.189))/FPS_devider #### to do
        elif self.binning == 8:
            if self.image_size == "FULL":
                self.FPS_exp = (678.62 * (exposure_dwell ** -0.943))/FPS_devider #### to do
            elif self.image_size == "HALF":
                self.FPS_exp = (678.62 * (exposure_dwell ** -0.943))/FPS_devider #### to do
            elif self.image_size == "QUARTER":
                self.FPS_exp = (678.62 * (exposure_dwell ** -0.943))/FPS_devider #### to do
        #self.FPS_exp = (678.62 * (self.exposure ** -0.943))/FPS_devider

        self.sleeping = ((FPS_devider-1) *(exposure_dwell/1000)) #this is 0.0 if FPS_devider = 1
        print("ready to acquire cRED data sleep: ", self.sleeping)
        print("FPS devider: ", FPS_devider)
        print('Live mode on, Ready for the collection...')


    def acquisition_cRED_data(self, stage_thread=None, timer = None, event = None, stop_event = None):
        # to change ####################################################################################
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
                #self.stopper = True  # stop camera
                return "aborted"
            event.set()
            while self.stopper == False:
                # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.tx = (time.monotonic_ns() - self.t1) / 1000000000
                self.img = self.acquire_image_fast()
                print("haadf grab")
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
                self.img = self.acquire_image_fast()
                print("haadf grab")
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


    def save_cRED_data(self, savingpath):
        # to change ####################################################################################
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
        pixelsize = None
        max_image_pixels = 2048
        print("camera:", self.name)
        print("physical camera pixelsize in um", pixelsize)
        print("max image pixels^2", max_image_pixels)
        return pixelsize, max_image_pixels

    def load_calibration_table(self):
        cwd = os.getcwd()
        if self.cam_table != None:
            path = cwd+os.sep+r"adaptor/camera/lookup_table/"+self.cam_table["stem_detector"]
        else:
            path = cwd+os.sep+r"adaptor/camera/lookup_table/spirit_haadf_table.txt"
            #path = cwd + os.sep + r"adaptor/camera/lookup_table/f30_haadf_table.txt"
        # table = {"IMAGING", {},
        #          "DIFFRACTION", {}}
        with open(path, 'r') as file:
            self.table = json.load(file)
        self.table["bottom_mounted"] = eval(self.table["bottom_mounted"])
        self.table["cam_flip_h"] = eval(self.table["cam_flip_h"])
        self.table["cam_flip_v"] = eval(self.table["cam_flip_v"])
        self.table["cam_flip_diag"] = eval(self.table["cam_flip_diag"])
        self.table["streamable"] = eval(self.table["streamable"])
        print(self.table)
        return self.table


    def is_cam_streaming(self):
        return False
    def is_cam_bottom_mounted(self):
        return self.table["bottom_mounted"]

    def get_stem_beamshift(self):
        pos = self.client.client_send_action({"get_stem_beam": ""})
        return pos["get_stem_beam"]

    def set_stem_beamshift(self, beamshift):
        """beamshift need to be a tuple of x,y in meters"""
        pos = self.client.client_send_action({"set_stem_beam": beamshift})
        return pos["set_stem_beam"]

if __name__ == '__main__':
    cam = Cam_haadf()


