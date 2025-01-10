import ctypes
import time

import numpy as np
import cv2

# Load the library
lib_handle = ctypes.CDLL("Libraries/TPX_Controller.so", ctypes.RTLD_GLOBAL)

createMpxModule = lib_handle._Z15createMpxModulei
destroyMpxModule = lib_handle._Z16destroyMpxModuleP9MpxModule
ping = lib_handle._Z4pingP9MpxModule
setAcqParams = lib_handle._Z11setAcqParamP9MpxModulej
startAcquisition = lib_handle._Z16startAcquisitionP9MpxModule
stopAcquisition = lib_handle._Z15stopAcquisitionP9MpxModule
readMatrix = lib_handle._Z10readMatrixP9MpxModulePsj
getBusy = lib_handle._Z7getBusyP9MpxModulePb
chipPosition = lib_handle._Z12chipPositionP9MpxModulei
chipCount = lib_handle._Z9chipCountP9MpxModule
resetMatrix = lib_handle._Z11resetMatrixP9MpxModule
resetChips = lib_handle._Z10resetChipsP9MpxModule

grab_image_from_detector = lib_handle._Z24grab_image_from_detectorP9MpxModulejPs
init_module = lib_handle._Z11init_moduleP9MpxModule

c_i16_array = np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS')
c_bool_ptr = ctypes.POINTER(ctypes.c_bool)


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


class CCameraInterface:

    # Constructor
    def __init__(self, _id):
        print("CCameraInterface - Constructor")

        # first things first, let's make sure all the imported methods are properly set
        setup_imported_methods()

        # For the timepix1, id should be 0
        self.this_ptr = createMpxModule(_id)
        self.img_data = np.zeros(512 * 512, dtype=np.int16)

        # Initialize the detector
        self.isInitialized = init_module(self.this_ptr)
        if self.isInitialized:
            print("Timepix module has been successfully initialized!")
        else:
            raise Exception("Error @init_module -> returned false")
        self.exposure_time = 500

        self.dead_pixels_coordinates = np.array([])
        self.flatfield_img = cv2.imread("Timepix_data/FlatField.tiff")
        if self.flatfield_img is None:
            print("Couldn't find and load image for Flatfield correction.\n")
        else:
            self.dead_pixels_coordinates = np.argwhere(self.flatfield_img == 0)

    # Destructor
    def __del__(self):
        print("CCameraInterface - Destructor")

        destroyMpxModule(self.this_ptr)

    def grab_image_from_detector(self):
        ref = np.ctypeslib.as_ctypes(self.img_data)
        grab_image_from_detector(self.this_ptr, ctypes.c_uint32(self.exposure_time), ctypes.byref(ref))

        self.apply_image_corrections()

        return np.append(self.img_data, np.int16(0))  # Contains the image taken from the timepix detector.

    def grab_image_from_detector_debug(self, fileName):
        self.img_data = cv2.imread(fileName, cv2.IMREAD_ANYDEPTH)
        time.sleep(self.exposure_time / 1000.0)
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

    def apply_flatfield_correction(self):
        # Ensure both images have the same size
        image = self.img_data.reshape(512, 512)
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
        d = 1
        for (i, j) in self.dead_pixels_coordinates:
            neighbours = img[i - d:i + d + 1, j - d:j + d + 1].flatten()
            img[i, j] = np.mean(neighbours)
        return img


    def correctCross(self, raw, factor=1):
        self.img_data = np.empty((516, 516), dtype=raw.dtype)
        self.img_data[0:256, 0:256] = raw[0:256, 0:256]
        self.img_data[0:256, 260:516] = raw[0:256, 256:512]
        self.img_data[260:516, 0:256] = raw[256:512, 0:256]
        self.img_data[260:516, 260:516] = raw[256:512, 256:512]

        self.img_data[255:258, :] = self.img_data[255] / factor
        self.img_data[:, 255:258] = self.img_data[:, 255:256] / factor

        self.img_data[258:261, :] = self.img_data[260] / factor
        self.img_data[:, 258:261] = self.img_data[:, 260:261] / factor

        self.img_data = self.img_data.ravel()
