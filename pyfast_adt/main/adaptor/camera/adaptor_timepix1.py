# Cinterface to control timepix1 device with relaxd board provided from Moussa Diame Faye (IIT, Pontedera).
# modded by Marco Santucci (JGU, Mainz) to be used with Fast-ADT.
#from .adaptor_cam import Cam_base
from .adaptor_cam import Cam_base
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
try:
# Load the library
    path1 = r'adaptor/camera/timepix1/Libraries/Tpx_Controller_Mainz.dll'
    lib_handle = ctypes.CDLL(path1, ctypes.RTLD_GLOBAL)

    createMpxModule = lib_handle.createMpxModule
    destroyMpxModule = lib_handle.destroyMpxModule
    ping = lib_handle.ping
    setAcqParams = lib_handle.setAcqParam
    startAcquisition = lib_handle.startAcquisition
    stopAcquisition = lib_handle.stopAcquisition
    readMatrix = lib_handle.readMatrix
    getBusy = lib_handle.getBusy
    chipPosition = lib_handle.chipPosition
    chipCount = lib_handle.chipCount
    resetMatrix = lib_handle.resetMatrix
    resetChips = lib_handle.resetChips

    grab_image_from_detector = lib_handle.grab_image_from_detector
    init_module = lib_handle.init_module

    c_i16_array = np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS')
    c_bool_ptr = ctypes.POINTER(ctypes.c_bool)
except Exception as err: print(err)
def setup_imported_methods():
    # MpxModule* createMpxModule(int id)
    createMpxModule.argtypes = [ctypes.c_int]
    createMpxModule.restype = ctypes.c_void_p

    # void destroyMpxModule(MpxModule* this_ptr)
    destroyMpxModule.argtypes = [ctypes.c_void_p]
    destroyMpxModule.restype = None

    # bool ping(MpxModule* this_ptr)
    ping.argtypes = [ctypes.c_void_p]
    ping.restype = ctypes.c_bool

    # int setAcqParam(MpxModule* this_ptr, AcqParams* _acqPars)
    setAcqParams.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    setAcqParams.restype = ctypes.c_int

    # int startAcquisition(MpxModule* this_ptr)
    startAcquisition.argtypes = [ctypes.c_void_p]
    startAcquisition.restype = ctypes.c_int

    # int stopAcquisition(MpxModule* this_ptr)
    stopAcquisition.argtypes = [ctypes.c_void_p]
    stopAcquisition.restype = ctypes.c_int

    # int readMatrix(MpxModule* this_ptr, i16* data, u32 sz)
    readMatrix.argtypes = [ctypes.c_void_p, c_i16_array, ctypes.c_uint32]
    readMatrix.restype = ctypes.c_int

    # int resetMatrix(MpxModule* this_ptr)
    resetMatrix.argtypes = [ctypes.c_void_p]
    resetMatrix.restype = ctypes.c_int

    # int resetChips(MpxModule* this_ptr)
    resetChips.argtypes = [ctypes.c_void_p]
    resetChips.restype = ctypes.c_int

    # bool getBusy(MpxModule* this_ptr, bool* busy)
    getBusy.argtypes = [ctypes.c_void_p, c_bool_ptr]
    getBusy.restype = ctypes.c_int

    # int chipPosition(MpxModule* this_ptr, int chipnr)
    chipPosition.argtypes = [ctypes.c_void_p, ctypes.c_int]
    chipPosition.restype = ctypes.c_int

    # int chipCount(MpxModule* this_ptr)
    chipCount.argtypes = [ctypes.c_void_p]
    chipCount.restype = ctypes.c_int

    # void grab_image_from_detector(MpxModule* this_ptr, u32 _exposureTime, i16* data)
    grab_image_from_detector.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p]
    grab_image_from_detector.restype = None

    # bool init_module(MpxModule* this_ptr)
    init_module.argtypes = [ctypes.c_void_p]
    init_module.restype = ctypes.c_bool

class Cam_timepix1(Cam_base):
    """Software interface for the ASI timepix1 camera.

        drc_name : str
            Set the default folder to store data in
        name : str
            Name of the interface
        """
    # Constructor
    def __init__(self, _id, instance_gui = None):
        super().__init__()
        self.name = None  # name of the camera
        self._id = _id  # id of the camera
        self.exposure = 100  # ms
        self.x = None
        self.y = None
        self.processing = None
        self.delay = None  # seconds
        self.binning = 1
        self.buffer_size = None
        self.stop_signal = None
        self.buffer = None
        self.instance_gui = instance_gui
        print("CCameraInterface - Constructor")
        #connecting the camera and setting up the release on disconnection
        self.connect()
        self.table = None
        self.timings = []
        self.load_calibration_table()
        atexit.register(self.release_connection)
    def connect(self):
        try:
            # first things first, let's make sure all the imported methods are properly set
            setup_imported_methods()

            # For the timepix1, id should be 0
            self.this_ptr = createMpxModule(self._id)
            self.img_data = np.zeros(512 * 512, dtype=np.int16)
            self.img_data_orig = np.zeros(512 * 512, dtype=np.int16)
            # Initialize the detector
            self.isInitialized = init_module(self.this_ptr)
            if self.isInitialized:
                print("Timepix module has been successfully initialized!")
            else:
                raise Exception("Error @init_module -> returned false")

            self.dead_pixels_coordinates = np.array([])
            self.flatfield_img = cv2.imread("timepix1/Timepix_data/FlatField.tiff")
            if self.flatfield_img is None:
                print("Couldn't find and load image for Flatfield correction.\n")
            else:
                self.dead_pixels_coordinates = np.argwhere(self.flatfield_img == 0)

        except Exception as e:
            print("Error @connect -> ", e)

    def release_connection(self):
        '''' release the connection with the device '''
        print("CCameraInterface - Destructor")
        destroyMpxModule(self.this_ptr)

    def set_exposure(self, exposure_time: int):
        '''' set the exposure time in ms for the camera '''
        self.exposure = exposure_time

    def get_exposure(self):
        '''' get the exposure time in ms for the camera '''
        return self.exposure

    def start_liveview(self, delay: float):
        '''' start the live view of the camera '''
        #todo
        pass

    def stop_liveview(self):
        '''' stop the live view of the camera '''
        #todo
        pass

    def set_binning(self, binning: int):
        '''' set the binning of the camera, common parameters are 1, 2, 4, 8 '''
        print("binning not available for timepix1")
        self.binning = 1

    def get_binning(self):
        '''' get the binning of the camera '''
        return self.binning

    def acquire_image(self, exposure_time: int, binning: int, processing = "Unprocessed"):
        """Acquire image through its adaptor and return it """
        self.set_exposure(exposure_time)
        self.grab_image_from_detector()

        if processing == "Background subtracted":
            self.img = self.correct_deadpixels(self.img_data)
            self.img = self.img_data.reshape((516, 516))
        elif processing == "Gain normalized":
            self.img = self.correct_deadpixels(self.img_data)
            self.img = self.apply_flatfield_correction()

        # self.img = self.correctCross(self.img)
        self.img = self.rotate_img(self.img)
        if processing == "Unprocessed":
            try:
                plt.imshow(self.img, vmax=np.max(self.img)/200)
                plt.show()
            except Exception as err:
                print(err)
                plt.imshow(self.img)
                plt.show()
        return self.img

    def acquire_image_and_show(self, exposure_time: int, binning: int, processing = "Unprocessed"):
        """Acquire image through its adaptor and return it """
        self.set_exposure(exposure_time)
        self.grab_image_from_detector()

        if processing == "Background subtracted":
            self.img = self.correct_deadpixels(self.img_data)
            self.img = self.img_data.reshape((516, 516))
        elif processing == "Gain normalized":
            self.img = self.correct_deadpixels(self.img_data)
            self.img = self.apply_flatfield_correction()

        # self.img = self.correctCross(self.img)
        self.img = self.rotate_img(self.img)
        if processing == "Unprocessed":
            try:
                plt.imshow(self.img, vmax=np.max(self.img)/200)
                plt.show()
            except Exception as err:
                print(err)
                plt.imshow(self.img)
                plt.show()
        return self.img

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
            self.processing = str(processing)
        except Exception as err: print(err)

    def get_processing(self):
        '''' get the processing typeof the camera '''
        # xf416r are ('Uncorrected', 'Dark subtracted', None, 'Gain corrected')[cfg.FlatMode]  # str, 2 undefined
        return self.processing

    def acquire_series_images(self, exposure_time: int, binning: int, processing: str, buffer_size: int,
                              stop_signal, display = False):
        i = 0
        # stop_signal = False
        binning = 1
        processing = None
        self.buffer = np.zeros((buffer_size, 512, 512), dtype=np.uint16)
        start_time = time.time()
        while i < buffer_size:
            self.img = self.acquire_image(exposure_time=exposure_time, binning=binning)
            self.buffer[i, :, :] = self.img
            # plt.imshow(img)
            # plt.show()
            i += 1
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.5f} seconds")

        return self.buffer

    def prepare_acquisition_cRED_data(self, camera: str, binning: int, exposure: int, buffer_size, FPS_devider = 1):

        self.exposure = exposure
        self.buffer_size = buffer_size
        self.binning = int(binning)
        ######### timepix is 512, 512 not necessary bin it ##############

        self.buffer = np.zeros((self.buffer_size, 516, 516), dtype=np.uint16)
        self.buffer_1 = np.zeros((self.buffer_size, 516, 516), dtype=np.uint16)
        self.zero_buffer = np.zeros((516, 516), dtype=np.uint16)

        self.FPS_exp = (1/self.exposure)/FPS_devider
        self.sleeping = ((FPS_devider-1) *(self.exposure/1000)) #this is 0.0 if FPS_devider = 1
        # the fucntion is not reading the memory but its directly taking into account the dead time
        # timepix don't need a sleeping time
        # # response of the XF416R considering the exposure and the readoutime in rolling_shutter_mode
        print("ready to acquire cRED data sleep: ", self.sleeping)
        print("FPS devider: ", FPS_devider)
        self.set_exposure(self.exposure)

    def acquisition_cRED_data(self, stage_thread=None, timer = None, event = None, stop_event = None):
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

        #camera_delay = self.table["Camera_delay"]
        self.stage_thread = stage_thread
        if timer != None:
            self.ref_timings["start_stage_thread_time"] = time.monotonic_ns()
        self.stage_thread.start()

        #self.img = self.acquire_image(exposure_time=self.exposure, binning=self.binning)

        #time.sleep(camera_delay)
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
                #self.stopper = True # stop camera
                return "aborted"
            event.set()
            while self.stopper == False:
                # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.tx = (time.monotonic_ns() - self.t1) / 1000000000
                self.img = self.acquire_image(exposure_time=self.exposure, binning=self.binning)
                print("cam grab")
                # 7.01 ms ± 159 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.buffer[self.i_, :, :] = self.img

                self.timings.append(self.tx)
                time.sleep(self.sleeping)
                self.i_ += 1
                if self.stage_thread.is_alive() == False:
                    self.stopper = True
        except Exception as err:
            print(err)
            print("using buffer_1")
            self.buffer_1_used = True
            while self.stopper == False:
                # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.tx = (time.monotonic_ns() - self.t1) / 1000000000
                self.img = self.acquire_image(exposure_time=self.exposure, binning=self.binning)
                print("cam grab")
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

#up to here everything looks nice and working, need to be tested
########################################################################################################################
#here are the original fucntion made by moussa to treat the camera and the data, need to be modded
    def grab_image_from_detector(self):
        ''' collect an image from the detector and return it as as np.array corrected for the cross. '''
        self.img_data = self.img_data_orig.copy()
        ref = np.ctypeslib.as_ctypes(self.img_data)
        grab_image_from_detector(self.this_ptr, ctypes.c_uint32(self.exposure), ctypes.byref(ref))
        self.img = np.reshape(self.img_data, (512,512))
        self.img = self.correctCross(self.img)
        #self.apply_image_corrections() #######################

        return self.img  # Contains the image taken from the timepix detector.

    def grab_image_from_detector_debug(self, fileName):
        self.img_data = cv2.imread(fileName, cv2.IMREAD_ANYDEPTH)
        time.sleep(self.exposure / 1000.0)
        self.apply_image_corrections()

        return np.append(self.img_data, np.int16(0)).ravel()

    def apply_image_corrections(self):
        if self.flatfield_img is None:
            return

        # 1st -> Flatfield correctoin
        flatfield_corr_img = self.apply_flatfield_correction()

        # 2nd -> Deadpixel correction
        deadpixel_corr_img = self.correct_deadpixels(flatfield_corr_img)

        # 3rd -> cross correction with a factor of 1
        self.correctCross(deadpixel_corr_img)

    def apply_flatfield_correction(self, image):
        # Ensure both images have the same size
        image = image.reshape(512, 512)
        if self.img_data.shape[:2] != self.flatfield_img.shape[:2]:
            raise ValueError("Image and flatfield should have the same size.")

        # Convert the images to float32 for accurate calculations
        image = image.astype(np.float32)
        self.flatfield_img = self.flatfield_img.astype(np.float32)

        # Apply flatfield correction
        flatfield_corrected_img = np.divide(image, self.flatfield_img)
        flatfield_corrected_img *= np.mean(self.flatfield_img)
        flatfield_corrected_img = np.round(flatfield_corrected_img).astype(np.uint16)

        return flatfield_corrected_img

    def correct_deadpixels(self, img):
        ''' correct the dead pixels of the collected image, by using a known dead_pixels_map.
        the dead pixels are replaced by the mean of the neighboring pixels.
        return the resulting image as 1D vector'''
        d = 1
        for (i, j) in self.dead_pixels_coordinates:
            neighbours = img[i - d:i + d + 1, j - d:j + d + 1].flatten()
            img[i, j] = np.mean(neighbours)
        return img

    #this function is already checked and tested! for timepix 1 factor = 3, for medipix 3, factor = 2?
    def correctCross(self, raw, factor=3):
        ''' correct the cross of the collected image from (512,512) to (516,516).
        the intensity from the neighboring pixels is divided by the factor value and assigned to the "cross" pixels.
        return the corrected image as a np.array of shape (516,516)
        the intensity of the reflections inside the cross are splitted in 3 consequent pixels'''

        self.img_data = np.empty((516, 516), dtype=raw.dtype)
        self.img_data[0:256, 0:256] = raw[0:256, 0:256]
        self.img_data[0:256, 260:516] = raw[0:256, 256:512]
        self.img_data[260:516, 0:256] = raw[256:512, 0:256]
        self.img_data[260:516, 260:516] = raw[256:512, 256:512]

        self.img_data[255:258, :] = self.img_data[255] / factor
        self.img_data[:, 255:258] = self.img_data[:, 255:256] / factor

        self.img_data[258:261, :] = self.img_data[260] / factor
        self.img_data[:, 258:261] = self.img_data[:, 260:261] / factor

        self.corrected_image = self.img_data.copy()
        self.img_data = self.img_data.ravel()

        return self.corrected_image

    def get_camera_characteristic(self):
        pixelsize = 55
        max_image_pixels = 516
        print("camera:", self.name)
        print("physical camera pixelsize in um", pixelsize)
        print("max image pixels^2", max_image_pixels)
        return pixelsize, max_image_pixels

    def load_calibration_table(self):
        cwd = os.getcwd()
        path = cwd + os.sep + r"adaptor/camera/lookup_table/timepix1_table.txt"
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
if __name__ == '__main__':
    tpxDetectorID = 0
    cam = Cam_timepix1(tpxDetectorID)
    #for testing
    #img_series = cam.acquire_series_images(buffer_size=100, exposure_time=500, binning=1, processing="unprocessed",
    #                                       stop_signal=None)

    # #test
    # cam.exposure = 100
    # cam.grab_image_from_detector()
    # import matplotlib.pyplot as plt
    #
    # a = np.reshape(cam.img_data, (512, 512))
    #
    # cam.correctCross(a)
    # a = np.reshape(cam.img_data, (516, 516))
    # plt.imshow(a)
    # plt.show()
