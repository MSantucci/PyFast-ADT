# serval toolkit adaptor to communicate with the asi cheeta detector (medipix3 hpd)
# adapted originally from instamatic
# Start servers in serval_toolkit:
# 1. `java -jar .\emu\tpx3_emu.jar`
# 2. `java -jar .\server\serv-2.1.3.jar`
# 3. launch `pyFast-ADT`

try:
    from .adaptor_cam import Cam_base
except:
    from adaptor_cam import Cam_base
import json
import imageio
import ctypes
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

try:
    from serval_toolkit.camera import Camera as ServalCamera
except:
    from .serval_toolkit.camera import Camera as ServalCamera

class Cam_medipix3(Cam_base):
    """Interfaces with Serval from ASI."""

    def __init__(self, name='serval', instance_gui = None):
        """Initialize camera module."""
        super().__init__()
        self.name = name
        self.instance_gui = instance_gui
        self.load_calibration_table()
        self.url = self.table["url_serval"]
        self.bpc_file_path, self.dacs_file_path = self.table["config_files"]
        self.establish_connection()
        atexit.register(self.release_connection)

    def get_image(self, exposure=None, binsize=None, triggers = 1, **kwargs):
        """Image acquisition routine. If the exposure and binsize are not
        given, the default values are read from the config file.

        exposure:
            Exposure time in seconds.
        binsize:
            Which binning to use.
        """
        if exposure is None:
            # exposure = self.default_exposure
            exposure = 0.1
        if not binsize:
            # binsize = self.default_binsize
            binsize = 1

        destination_dict = self.conn.get_request('/server/destination').json()
        destination_dict['Image'] = [{
            # Where to place the preview files (HTTP end-point: GET localhost:8080/measurement/image)
            "Base": "http://localhost",
            # What (image) format to provide the files in.
            "Format": 'tiff',
            # What data to build a frame from
            "Mode": "count"}]
        # print(destination_dict)
        self.conn.destination = destination_dict

        # # Upload exposure settings (Note: will do nothing if no change in settings)
        self.conn.set_detector_config(ExposureTime=exposure/1000, TriggerPeriod=exposure/1000 + 0.00050001, nTriggers = triggers)

        # Check if measurement is running. If not: start
        db = self.conn.dashboard
        if db['Measurement'] is None or db['Measurement']['Status'] != 'DA_RECORDING':
            self.conn.measurement_start()

        # Request a frame. Will be streamed *after* the exposure finishes
        img = self.conn.get_image_stream(nTriggers = triggers, disable_tqdm = True)

        arr = np.array(img)
        try:
            self.conn.measurement_stop()
        except Exception as err:
            print(err)
        return arr

    def get_movie(self, n_frames, exposure=None, binsize=None, **kwargs):
        """Movie acquisition routine. If the exposure and binsize are not
        given, the default values are read from the config file.

        n_frames:
            Number of frames to collect
        exposure:
            Exposure time in seconds.
        binsize:
            Which binning to use.
        """
        if exposure is None:
            # exposure = self.default_exposure
            exposure = 0.1
        if not binsize:
            # binsize = self.default_binsize
            binsize = 1

        self.conn.set_detector_config(TriggerMode = 'CONTINUOUS')

        arr = self.conn.get_images(
            nTriggers=n_frames,
            ExposureTime=exposure,
            TriggerPeriod=exposure,
        )

        return arr

    def get_image_dimensions(self) -> (int, int):
        """Get the binned dimensions reported by the camera."""
        binning = self.get_binning()
        dim_x, dim_y = self.get_camera_dimensions()

        dim_x = int(dim_x / binning)
        dim_y = int(dim_y / binning)

        return dim_x, dim_y

    def get_camera_dimensions(self) -> (int, int):
        """Get the dimensions reported by the camera."""
        dim = self.get_camera_characteristic()[1]
        return (dim, dim)

    def get_name(self) -> str:
        """Get the name reported by the camera."""
        return self.name

    def connect(self):
        pass

    def establish_connection(self) -> None: # to add parameters to work here
        """Establish connection to the camera. by adding a new destination called Image to the detector,
        this is made to preserve previous exsisting destinations necessary for other operations"""

        self.conn = ServalCamera()
        self.conn.connect(self.url)
        # self.conn.set_chip_config_files(bpc_file_path=self.bpc_file_path, dacs_file_path=self.dacs_file_path)
        # to enable again

        # self.conn.set_detector_config(**self.detector_config)
        # Check pixel depth. If 24 bit mode is used, the pgm format does not work
        # (has support up to 16 bits) so use tiff in that case. In other cases (1, 6, 12 bits)
        # use pgm since it is more efficient
        self.pixel_depth = self.conn.detector_config['PixelDepth']
        # detector_config = self.conn.get_request('/detector/config').json()
        destination_dict = self.conn.get_request('/server/destination').json() # debug line
        print(self.conn.destination) # debug line
        if self.pixel_depth == 24:
            file_format = 'tiff'
        else:
            file_format = 'pgm'
        # self.conn.destination = {
        #         "Image":
        #             [{
        #             # Where to place the preview files (HTTP end-point: GET localhost:8080/measurement/image)
        #             "Base": "http://localhost",
        #             # What (image) format to provide the files in.
        #             "Format": file_format,
        #             # What data to build a frame from
        #             "Mode": "count"
        #     }]
        # }

        destination_dict['Image'] = [{
                    # Where to place the preview files (HTTP end-point: GET localhost:8080/measurement/image)
                    "Base": "http://localhost",
                    # What (image) format to provide the files in.
                    "Format": file_format,
                    # What data to build a frame from
                    "Mode": "count"
                }]
        print(destination_dict)
        self.conn.destination = destination_dict
        print(self.conn.destination) # debug line

    def release_connection(self) -> None:
        """Release the connection to the camera."""
        self.conn.measurement_stop()
        name = self.get_name()
        msg = "Connection to camera %s released" %str(name)
        print(msg)

    ##########################################################################################
    ############################## adding my functions here ##################################
    ##########################################################################################
    def set_exposure(self, exposure_time: int):
        '''' set the exposure time in ms for the camera '''
        self.exposure = exposure_time

    def get_exposure(self):
        '''' get the exposure time in ms for the camera '''
        return self.exposure

    def start_liveview(self, delay: float):
        '''' start the live view of the camera '''
        pass

    def stop_liveview(self):
        '''' stop the live view of the camera '''
        pass

    def set_binning(self, binning: int):
        '''' set the binning of the camera, common parameters are 1, 2, 4, 8 '''
        print("binning not available for medipix3")
        self.binning = 1

    def get_binning(self):
        '''' get the binning of the camera '''
        return self.binning

    def acquire_image(self, exposure_time: int, binning: int, processing="Unprocessed"):
        """Acquire image through its adaptor and return it, exposure in ms and binning is usually 1"""

        # destination_dict = self.conn.get_request('/server/destination').json()  # debug line

        # print("before get_image method", json.dumps(destination_dict, indent = 4))  # debug line
        img = self.get_image(exposure=exposure_time, binning = binning)[0]
        img = self.rotate_img(img)
        return img

    def acquire_image_and_show(self, exposure_time: int, binning: int, processing="Unprocessed"):
        """Acquire image through its adaptor and return it, exposure in ms and binning is usually 1"""

        img = self.get_image(exposure=exposure_time, binning = binning)[0]
        img = self.rotate_img(img)
        # self.correctCross(img)
        try:
            plt.imshow(img, vmax=np.max(img) / 200)
            plt.show()
        except Exception as err:
            print(err)
            plt.imshow(img)
            plt.show()
        return img

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
        try:
            # self.processing = str(processing)
            self.processing = "Unprocessed"
        except Exception as err:
            print(err)

    def get_processing(self):
        '''' get the processing typeof the camera '''
        # xf416r are ('Uncorrected', 'Dark subtracted', None, 'Gain corrected')[cfg.FlatMode]  # str, 2 undefined
        return self.processing

    def acquire_series_images(self, exposure_time: int, binning: int, processing: str, buffer_size: int,
                              stop_signal, display=False):
        i = 0
        # stop_signal = False
        binning = 1
        processing = None
        # self.buffer = np.zeros((buffer_size, 512, 512), dtype=np.uint16)
        start_time = time.time()
        # while i < buffer_size:
        img = self.get_image(exposure=exposure_time, binning=binning, triggers = buffer_size)


            # self.img = self.acquire_image(exposure_time=exposure_time, binning=binning)
            # self.buffer[i, :, :] = self.img

            # plt.imshow(img)
            # plt.show()
            # i += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.5f} seconds")

        self.buffer = img

        return self.buffer

    def prepare_acquisition_cRED_data(self, camera: str, binning: int, exposure: int, buffer_size, FPS_devider=1):

        self.exposure = exposure
        self.buffer_size = buffer_size
        self.binning = int(binning)
        ######### timepix is 512, 512 not necessary bin it ##############

        self.buffer = np.zeros((self.buffer_size, 512, 512), dtype=np.uint16)
        self.buffer_1 = np.zeros((self.buffer_size, 512, 512), dtype=np.uint16)
        self.zero_buffer = np.zeros((512, 512), dtype=np.uint16)

        self.FPS_exp = (1 / self.exposure) / FPS_devider
        self.sleeping = ((FPS_devider - 1) * (self.exposure / 1000))  # this is 0.0 if FPS_devider = 1

        print("ready to acquire cRED data sleep: ", self.sleeping)
        print("FPS devider: ", FPS_devider)
        self.set_exposure(self.exposure)



    def acquisition_cRED_data(self, stage_thread=None, timer=None, event=None, stop_event=None):
        ''' Acquire images into the buffer up to the thread is alive, usually the stage thread is passed for cRED experiments '''
        print("starting acquisition")
        i = 0
        self.stopper = False
        if timer == None:
            self.t1 = time.monotonic_ns()
        else:
            self.ref_timings = {}
            self.t1 = time.monotonic_ns()
            self.ref_time = timer
            self.ref_timings["start_t1_time"] = self.t1
            self.ref_timings["start_ref_time"] = self.ref_time

        # camera_delay = self.table["Camera_delay"]
        self.stage_thread = stage_thread
        if timer != None:
            self.ref_timings["start_stage_thread_time"] = time.monotonic_ns()
        self.stage_thread.start()

        # self.img = self.acquire_image(exposure_time=self.exposure, binning=self.binning)

        # time.sleep(camera_delay)
        self.timings = []
        # acquisition loop in self.img passing from the camera buffer
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
            # this start the collection of the dataset
            self.prepare_collection_cred_images(exposure=self.exposure, binsize=1, triggers=10000)
            while self.stopper == False:
                # # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                # self.tx = (time.monotonic_ns() - self.t1) / 1000000000
                # self.img = self.collect_image_cred()
                # # print("cam grab")
                # # 7.01 ms ± 159 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                # self.buffer[self.i_, :, :] = self.img
                #
                # self.timings.append(self.tx)
                # time.sleep(self.sleeping)
                # self.i_ += 1
                if self.stage_thread.is_alive() == False:
                    self.stopper = True
                    self.conn.measurement_stop()
        except Exception as err:

            print(err, "triggered an error e\here debug line marco")
            # print("using buffer_1")
            # self.buffer_1_used = True
            # while self.stopper == False:
            #     # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
            #     self.tx = (time.monotonic_ns() - self.t1) / 1000000000
            #     self.img = self.collect_image_cred()
            #     # print("cam grab")
            #     # 7.01 ms ± 159 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
            #     self.buffer_1[self.i_, :, :] = self.img
            #
            #     self.timings.append(self.tx)
            #     time.sleep(self.sleeping)
            #     self.i_ += 1
            #     if self.stage_thread.is_alive() == False:
            #         self.stopper = True

        self.t2 = (time.monotonic_ns() - self.t1) / 1000000000
        if timer != None:
            self.ref_timings["end_acq_cred_time"] = time.monotonic_ns()
        self.support = []
        self.support_1 = []

        # end the data acquisition loop
        # self.conn.measurement_stop()
        number_frames_collected = self.conn.dashboard['Measurement']['FrameCount']
        print("debug line here, number_frames_collected:", number_frames_collected)

        for i_ in range(self.buffer.shape[0]):
            try:
                self.buffer[i_, :, :] = self.collect_image_cred(triggers = 1)[0]
            except:
                break
        if i_+ 1 != number_frames_collected:
            for i_2 in range(self.buffer_1.shape[0]):
                try:
                    self.buffer_1[i_2, :, :] = self.collect_image_cred(triggers=1)[0]
                except:
                    break
        print("debug line here, i_, i_2, number_frames_collected:", i_, i_2, number_frames_collected)

        i = 0
        for images, images_1 in zip(self.buffer, self.buffer_1):
            if np.array_equal(images, self.zero_buffer) == True:
                self.support.append(i)
            if np.array_equal(images_1, self.zero_buffer) == True:
                self.support_1.append(i)
            i += 1
        # print("support list: ", len(self.support),self.support)
        if len(self.support) > 0 or len(self.support_1) > 0:
            print("shape buffer before: ", np.shape(self.buffer), "shape buffer_1 before: ",
                  np.shape(self.buffer_1))
            if len(self.support) > 0:
                try:
                    print("buffer resize parameters: ", self.support[0], np.shape(self.buffer)[1],
                          np.shape(self.buffer)[2])
                    self.buffer = np.resize(self.buffer,
                                            (self.support[0], np.shape(self.buffer)[1], np.shape(self.buffer)[2]))
                except Exception as err:
                    pass
            if len(self.support_1) > 0:
                try:
                    print("buffer_1 resize parameters: ", self.support_1[0], np.shape(self.buffer)[1],
                          np.shape(self.buffer)[2])
                    self.buffer_1 = np.resize(self.buffer_1, (
                        self.support_1[0], np.shape(self.buffer_1)[1], np.shape(self.buffer_1)[2]))
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
        self.ii = 1
        self.saving_dir = str(savingpath)
        self.current_dir = os.getcwd()
        print("saving datacollection in: ", self.saving_dir)
        if len(str(self.buffer.shape[0] + self.buffer_1.shape[0])) <= 4:
            self.name_zeros = 4
        else:
            self.name_zeros = len(str(self.buffer.shape[0] + self.buffer_1.shape[0]))

        for image in self.buffer:
            if not self.current_dir == self.saving_dir:
                os.chdir(self.saving_dir)
            else:
                pass

            # format tiff uncompressed data
            self.image_name = str('%s.tif' % (format(self.ii, '.0f').rjust(self.name_zeros, '0')))
            # print("saving: ", self.image_name)
            # image = self.correctCross(image)
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
                # image = self.correctCross(image)
                imageio.imwrite(self.image_name, image)
                self.ii += 1

        os.chdir(self.current_dir)

    def correctCross(self, raw, factor=2):
        ''' correct the cross of the collected image from (512,512) to (514,514).
        the intensity from the neighboring pixels is divided by the factor value and assigned to the "cross" pixels.
        return the corrected image as a np.array of shape (514,514)
        the intensity of the reflections inside the cross are splitted in 2 consequent pixels'''
        # if factor == 3:
        # self.img_data = np.empty((516, 516), dtype=raw.dtype)
        # self.img_data[0:256, 0:256] = raw[0:256, 0:256]
        # self.img_data[0:256, 260:516] = raw[0:256, 256:512]
        # self.img_data[260:516, 0:256] = raw[256:512, 0:256]
        # self.img_data[260:516, 260:516] = raw[256:512, 256:512]
        #
        # self.img_data[255:258, :] = self.img_data[255] / factor
        # self.img_data[:, 255:258] = self.img_data[:, 255:256] / factor
        #
        # self.img_data[258:261, :] = self.img_data[260] / factor
        # self.img_data[:, 258:261] = self.img_data[:, 260:261] / factor
        #
        # self.corrected_image = self.img_data.copy()

        # elif factor == 2:
        self.img_data = np.empty((514, 514), dtype=raw.dtype)
        self.img_data[0:256, 0:256] = raw[0:256, 0:256]
        self.img_data[0:256, 258:514] = raw[0:256, 256:512]
        self.img_data[258:514, 0:256] = raw[256:512, 0:256]
        self.img_data[258:514, 258:514] = raw[256:512, 256:512]

        self.img_data[255:257, :] = self.img_data[255] / factor
        self.img_data[:, 255:257] = self.img_data[:, 255:256] / factor

        self.img_data[257:259, :] = self.img_data[260] / factor
        self.img_data[:, 257:259] = self.img_data[:, 260:261] / factor

        self.corrected_image = self.img_data.copy()

        return self.corrected_image

    def get_camera_characteristic(self):
        pixelsize = 55
        max_image_pixels = 514
        print("camera:", self.name)
        print("physical camera pixelsize in um", pixelsize)
        print("max image pixels^2", max_image_pixels)
        return pixelsize, max_image_pixels

    def load_calibration_table(self):
        cwd = os.getcwd()
        # path = cwd + os.sep + r"adaptor/camera/lookup_table/medipix3_ASI_table.txt"
        path = r"C:\\pyfast_adt\\main\\adaptor\\camera/lookup_table/medipix3_ASI_table.txt"
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

    def collect_image_cred(self, triggers = 1, **kwargs):
        """specific function for ASI cheetah data collection. this function retrieve an image from the buffer of the camera
        the problem is that resolve too fast and fill the buffer of the datacollection in pyfast adt too fast.
        or we introduce a sleeper or we find another way."""
        # Request a frame. Will be streamed *after* the exposure finishes
        img = self.conn.get_image_stream(nTriggers=triggers, disable_tqdm=True)
        arr = np.array(img, dtype=np.uint16) # for now forced to be 16 bit to discuss with lukas
        return arr

    def prepare_collection_cred_images(self, exposure=None, binsize=1, triggers = 10000, **kwargs):
        if exposure is None:
            # exposure = self.default_exposure
            exposure = 0.1

        binsize = 1

        destination_dict = self.conn.get_request('/server/destination').json()
        destination_dict['Image'] = [{
            # Where to place the preview files (HTTP end-point: GET localhost:8080/measurement/image)
            "Base": "http://localhost",
            # What (image) format to provide the files in.
            "Format": 'tiff',
            # What data to build a frame from
            "Mode": "count",
            "QueueSize": 1024}]
        # print(destination_dict)
        self.conn.destination = destination_dict

        # # Upload exposure settings (Note: will do nothing if no change in settings)
        self.conn.set_detector_config(ExposureTime=exposure/1000, TriggerPeriod=exposure/1000 + 0.00050001, nTriggers = triggers)
        print(self.conn.get_request('/detector/config').text)
        # Check if measurement is running. If not: start
        db = self.conn.dashboard
        if db['Measurement'] is None or db['Measurement']['Status'] != 'DA_RECORDING':
            self.conn.measurement_start()

    def shuffle_chip_order(self, raw, order = [1, 2, 3, 4]):
        """Shuffle the chips in the image based on the given order.

            Parameters:
            - raw: np.ndarray, Input 512x512 image.
            - order: list of int, Desired order of chips."""

        self.img_data = np.empty((512, 512), dtype=raw.dtype)

        dict_order = {1: raw[0:256, 0:256],
                      2: raw[0:256, 256:512],
                      3: raw[256:512, 0:256],
                      4: raw[256:512, 256:512]}

        self.img_data[0:256, 0:256] = dict_order[order[0]]
        self.img_data[0:256, 256:512] = dict_order[order[1]]
        self.img_data[256:512, 0:256] = dict_order[order[2]]
        self.img_data[256:512, 256:512] = dict_order[order[3]]


        self.corrected_image = self.img_data.copy()

        return self.corrected_image

if __name__ == '__main__':
    cam = Cam_medipix3()
    from IPython import embed
    embed()
