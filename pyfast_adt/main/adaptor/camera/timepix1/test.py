import socket
import ctypes
import time
import cv2
import tifffile
import numpy as np

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
test_func = lib_handle._Z9test_dataPss

c_i16_array = np.ctypeslib.ndpointer(dtype=np.int16, ndim=1, flags='C_CONTIGUOUS')
i16_p = ctypes.POINTER(ctypes.c_int16)
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

    # void test_data(i16* data, i16 value)
    test_func.argtype = [c_i16_array, ctypes.c_int16]
    test_func.restype = None







#
# start_time = time.time()
# my_function()
# end_time = time.time()
#
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.5f} seconds")


def main():


    print("Hello World!")
    setup_imported_methods()

    this_ptr = createMpxModule(0)


    bIsInitialized = init_module(this_ptr)
    print("init_module returned {}".format(bIsInitialized))

    # data = np.zeros(512*512, dtype=np.int16)
    # data_ptr = data.ctypes.data_as(i16_p)
    # image_array = np.ctypeslib.as_array(data, shape=(512, 512))
    # cv2.imshow('Image Before readMatrix', image_array)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    data = np.empty(512 * 512, dtype=np.int16)
    ref = np.ctypeslib.as_ctypes(data)

    # display the image using OpenCV
    cv2.imshow('ImageBefore', data.reshape(512, 512))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    for i in range(10):
        start_time = time.time()
        grab_image_from_detector(this_ptr, 500, ctypes.byref(ref))
        name = "output" + str(i) + ".tiff"
        tifffile.imwrite(name, data.reshape(512, 512, 1), dtype=np.int16)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.5f} seconds")
    cv2.imshow("ImageAfter", data.reshape(512, 512))
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # pingRet = ping(this_ptr)
    # acq1 = startAcquisition(this_ptr)
    # acq2 = stopAcquisition(this_ptr)
    # res1 = resetMatrix(this_ptr)
    # res2 = resetChips(this_ptr)
    # chipC = chipCount(this_ptr)
    # acq = setAcqParams(this_ptr, 500)
    # test = ctypes.c_bool(False)
    # busy_ptr = ctypes.pointer(test)
    #
    # print("Test Before {}".format(test))
    # busy = getBusy(this_ptr, busy_ptr)
    # print("Test After {}".format(test))
    # print("Ping {}\nStartAcquisition {}\nStopAcquisition {}\nresetMatrix {}\nresetChips {}\nchipCount {}\nsetAcqParams {}\n".format(pingRet, acq1, acq2, res1, res2, chipC, acq))
    #
    # data = np.empty(10, dtype=np.int16)
    # print(data)
    # test_func(np.ctypeslib.as_ctypes(data), 69)
    # print("Simulating readMatrix")
    # data = np.empty(10, dtype=np.int16)
    # data_ptr = data.ctypes.data_as(i16_p)
    #
    # value = ctypes.c_int16(69)
    # print(data)
    # test_func(data_ptr, value)
    # print(data)

    destroyMpxModule(this_ptr)




if __name__ == "__main__":
    main()
