# interface to control merlin camera device through the API of QD.
# modded by Marco Santucci (JGU, Mainz) and rushabh bharadva (TU Darmstadt) to be used with Fast-ADT.
#from .adaptor_cam import Cam_base
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
try:
    from .merlin_io import load_mib # to add .merlin_io in the folder
except ImportError:
    from merlin_io import load_mib

def MPX_CMD(type_cmd: str = 'GET', cmd: str = 'DETECTORSTATUS') -> bytes:
    """Generate TCP command bytes for Merlin software.

    Default value 'GET,DETECTORSTATUS' probes for the current status of the detector.

    Parameters
    ----------
    type_cmd : str, optional
        Type of the command
    cmd : str, optional
        Command to execute

    Returns
    -------
    bytes
        Command code in bytes format
    """
    length = len(cmd)
    # tmp = 'MPX,00000000' + str(length+5) + ',' + type_cmd + ',' + cmd
    tmp = f'MPX,00000000{length+5},{type_cmd},{cmd}'
    return tmp.encode()


class Cam_merlin(Cam_base):
    ###### these are properties from instamatic/src/instamatic/config/camera/merlin.yaml #########
    # camera_rotation_vs_stage_xy: 0.0
    # default_binsize: 1 xxx
    # default_exposure: 0.02
    # dimensions: [512, 512]
    # dynamic_range: 11800
    # interface: merlin
    # physical_pixelsize: 0.055
    # possible_binsizes: [1]
    # stretch_amplitude: 0.0
    # stretch_azimuth: 0.0
    # detector_config:
    #   CONTINUOUSRW: 1
    #   COUNTERDEPTH: 12
    #   FILEENABLE: 0
    #   RUNHEADLESS: 1
    # # server hostname or IP address and port used to send commands
    # host: '127.0.0.1' xxx
    # commandport: 6341 xxx
    # dataport: 6342 xxx

    """Camera interface for the Quantum Detectors Merlin camera."""
    socket.setdefaulttimeout(5)  # seconds

    def __init__(self, instance_gui = None):
        """Initialize camera module."""
        super().__init__()
        self.START_SIZE = 14
        self.MAX_NUMFRAMESTOACQUIRE = 42_949_672_950
        self.name = None  # name of the camera
        self._state = {}

        self._soft_trigger_mode = False
        self._soft_trigger_exposure = None
        self.default_exposure = 0.5 # s?
        self.host, self.commandport, self.dataport = '127.0.0.1', 6341, 6342
        self.detector_config = {'CONTINUOUSRW': 1, 'COUNTERDEPTH': 12, 'FILEENABLE': 0, 'RUNHEADLESS': 0}
        msg = f'Camera {self.get_name()} initialized'
        self.s_data = None
        self.s_cmd = None
        self.exposure = 100  # ms
        self.default_exposure = self.exposure
        self.x = None
        self.y = None
        self.processing = "Unprocessed"
        self.delay = None  # seconds
        self.binning = 1
        self.default_binsize = self.binning
        self.buffer_size = None
        self.stop_signal = None
        self.buffer = None
        self.instance_gui = instance_gui
        self.connect() #connecting the camera and setting up the release on disconnection
        self.table = None
        self.timings = []
        self.load_calibration_table()
        self._frame_number = 0

    def receive_data(self, continuous = False, *, nbytes: int) -> bytearray:
        """Safely receive from the socket until `n_bytes` of data are received."""
        data = bytearray()
        n = 0
        t0 = time.perf_counter()
        if continuous == False:
            while len(data) != nbytes:
                # print("checker: ", len(data), nbytes, len(data) == nbytes)
                data.extend(self.s_data.recv(nbytes - len(data)))
                n += 1

                # if n in [1, 5, 10, 20, 30, 100]:
                #     print(str(data))

            t1 = time.perf_counter()
            print('Received %d bytes in %d steps (%f s)', len(data), n, t1 - t0)

        elif continuous == True:
            while len(data) != nbytes:
                data.extend(self.s_data.recv(nbytes - len(data)))
                n += 1

            t1 = time.perf_counter()
        return data

    def merlin_set(self, key: str, value: Any):
        """Set state on Merlin parameter through command socket. """

        if self._state.get(key) == value:
            return

        self.s_cmd.sendall(MPX_CMD('SET', f'{key},{value}'))
        response = self.s_cmd.recv(1024).decode()

        *_, status = response.rsplit(',', 1)
        if status == '2':
            print('Merlin did not understand: %s' % response)
        else:
            self._state[key] = value
            print('Remembering state for %s value %s', key, value)

    def merlin_get(self, key: str) -> str:
        """Get state of Merlin parameter through command socket."""

        self.s_cmd.sendall(MPX_CMD('GET', key))
        response = self.s_cmd.recv(1024).decode()

        _, value, status = response.rsplit(',', 2)
        if status == '2':
            print('Merlin did not understand: %s' % response)
        return value

    def merlin_cmd(self, key: str):
        """Send Merlin command through command socket. """
        self.s_cmd.sendall(MPX_CMD('CMD', key))
        response = self.s_cmd.recv(1024).decode()

        _, status = response.rsplit(',', 1)
        if status == '2':
            raise ValueError('Merlin did not understand: {response}')

    def setup_soft_trigger(self, exposure = None):
        """Set up for repeated acquisition using soft trigger, and start acquisition.
            exposure in ms"""
        if exposure is None:
            exposure = self.default_exposure

        # convert s to ms
        # exposure_ms = exposure * 1000
        exposure_ms = exposure

        if self._soft_trigger_mode:
            self.teardown_soft_trigger()

        self._soft_trigger_mode = True
        self._soft_trigger_exposure = exposure

        self.merlin_set('ACQUISITIONTIME', exposure_ms)
        self.merlin_set('ACQUISITIONPERIOD', exposure_ms)

        self._frame_number = 0

        # Set NUMFRAMESTOACQUIRE to maximum
        # Merlin collects up to this number of frames with a single SOFTTRIGGER acquisition
        self.merlin_set('NUMFRAMESTOACQUIRE', self.MAX_NUMFRAMESTOACQUIRE)

        self.merlin_set('TRIGGERSTART', 5)
        self.merlin_set('NUMFRAMESPERTRIGGER', 1)
        self.merlin_cmd('STARTACQUISITION') ##### start cmd
        # self.merlin_cmd('STOPACQUISITION') ##### stop cmd

        start = self.receive_data(nbytes=self.START_SIZE)

        header_size = int(start[4:])
        header = self.receive_data(nbytes=header_size)

        self._frame_length = None

    def teardown_soft_trigger(self):
        """Stop soft trigger acquisition. i.e. continuous acquisition."""
        self.merlin_cmd('STOPACQUISITION')
        self._soft_trigger_mode = False
        self._soft_trigger_exposure = None

    def get_image(self, exposure = None, **kwargs) -> np.ndarray:
        """Image acquisition routine. If the exposure is not given, the default
        value is read from the config file. the exposure in ms.  """
        if not exposure:
            exposure = self.default_exposure

        if not self._soft_trigger_mode:
            print('Set up soft trigger with exposure %s ms', exposure)
            self.setup_soft_trigger(exposure=exposure)
        elif exposure != self._soft_trigger_exposure:
            print('Change exposure to %s ms', exposure)
            self.setup_soft_trigger(exposure=exposure)
        elif self._frame_number == self.MAX_NUMFRAMESTOACQUIRE:
            print('Maximum frame %s number reached for this acquisition, resetting soft trigger.' % str(self.MAX_NUMFRAMESTOACQUIRE))

            self.setup_soft_trigger(exposure=exposure)

        self.merlin_cmd('SOFTTRIGGER')

        if not self._frame_length:
            mpx_header = self.receive_data(nbytes=self.START_SIZE)
            size = int(mpx_header[4:])

            print('Received header: %s (%s)', size, mpx_header)

            framedata = self.receive_data(nbytes=size)
            skip = 0

            self._frame_length = self.START_SIZE + size
        else:
            framedata = self.receive_data(nbytes=self._frame_length)
            skip = self.START_SIZE

        self._frame_number += 1

        # Must skip first byte when loading data to avoid off-by-one error
        data = load_mib(framedata, skip=1 + skip).squeeze()

        # data[self._frame_number % 512] = 10000
        print("shape of the image:", np.shape(data))
        self.teardown_soft_trigger()
        return data

    def get_movie(self, n_frames: int, exposure: float = None, **kwargs):
        """Gapless movie acquisition routine. If the exposure is not given, the
        default value is read from the config file.   """
        pass

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
        try:
            atexit.register(self.release_connection)
            # add cmd connection
            self.establish_connection()
            # add data connection
            self.establish_data_connection()
        except Exception as e:
            print("Error connecting to merlin detector -> ", e)

    def establish_connection(self) -> None:
        """Establish connection to command port of the merlin software."""
        # Create command socket
        self.s_cmd = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect sockets and probe for the detector status
        print('Connecting to Merlin on %s:%s', self.host, self.commandport)
        try:
            self.s_cmd.connect((self.host, self.commandport))

        except ConnectionRefusedError as e:
            raise ConnectionRefusedError(f'Could not establish command connection to {self.name}. '
                                         f'The Merlin command port is not responding.') from e
        except OSError as e:
            raise OSError(f'Could not establish command connection to {self.name} ({e.args[0]}). '
                          f'Did you start the Merlin software? Is the IP correct? Is the Merlin command port already connected?).') from e

        version = self.merlin_get(key='SOFTWAREVERSION')
        print('Merlin version: %s', version)

        status = self.merlin_get(key='DETECTORSTATUS')
        print('Merlin status: %s', status)

        for key, value in self.detector_config.items():
            self.merlin_set(key, value)

    def establish_data_connection(self) -> None:
        """Establish connection to the dataport of the merlin software."""
        # Create command socket
        s_data = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect sockets and probe for the detector status
        try:
            s_data.connect((self.host, self.dataport))
        except ConnectionRefusedError as e:
            raise ConnectionRefusedError(f'Could not establish data connection to {self.name} ({e.args[0]}). '
                'The Merlin data port is not responding.') from e

        self.s_data = s_data

    def release_connection(self) -> None:
        """Release the connection to the camera."""
        if self._soft_trigger_mode:
            print('Stopping acquisition')
            self.teardown_soft_trigger()

        self.s_cmd.close() # sock.shutdown(socket.SHUT_RDWR) alternative way

        self.s_data.close()

        name = self.get_name()
        msg = "Connection to camera %s released" %str(name)
        print(msg)


    def set_exposure(self, exposure_time: int):
        '''' set the exposure time in ms for the camera '''
        self.merlin_set('ACQUISITIONTIME', exposure_time)
        self.merlin_set('ACQUISITIONPERIOD', exposure_time)

    def get_exposure(self):
        '''' get the exposure time in ms for the camera '''
        return self.merlin_get(key = 'ACQUISITIONTIME')

    def start_liveview(self, delay: float):
        '''' start the live view of the camera '''
        # todo
        pass

    def stop_liveview(self):
        '''' stop the live view of the camera '''
        # todo
        pass

    def set_binning(self, binning: int):
        '''' set the binning of the camera, common parameters are 1, 2, 4, 8 '''
        print("binning not available for timepix1")
        self.binning = 1


    def get_binning(self):
        '''' get the binning of the camera '''
        return self.binning


    def acquire_image(self, exposure_time: int, binning: int, processing="Unprocessed"):
        """Acquire image through its adaptor and return it, exposure in ms and binning is usually 1"""

        img = self.get_image(exposure=exposure_time, binning = binning)
        img = self.rotate_img(img)
        return img

    def acquire_image_and_show(self, exposure_time: int, binning: int, processing="Unprocessed"):
        """Acquire image through its adaptor and return it, exposure in ms and binning is usually 1"""

        img = self.get_image(exposure=exposure_time, binning = binning)
        img = self.rotate_img(img)
        self.correctCross(img)
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
        '''' set the processing of the camera, processing = "Unprocessed" is the only available for this camera'''
        self.processing = "Unprocessed"

    def get_processing(self):
        '''' get the processing type of the camera '''
        return self.processing

    def acquire_series_images(self, exposure_time: int, binning = 1, processing = "unprocessed", buffers = None, stop_signal = None, display=False):
        """Image acquisition routine using multiple softtriggers. the exposure in ms. you need to pass the buffers
        where to save the images and a stop signal to say that the acquisition is finished.

        the acquisition loop can finish in 3 ways:
        self.stopper = True, usually in case of multithreading, someone else can set it to True,
        stop_signal.is_alive() == False, the thread associated with stop_signal is no more alive,
        the buffers are full the acquisition is finished. """

        self.stopper = False
        #setup softrigger
        exposure = exposure_time
        if not self._soft_trigger_mode:
            print('Set up soft trigger with exposure %s ms', exposure)
            self.setup_soft_trigger(exposure=exposure)
        elif exposure != self._soft_trigger_exposure:
            print('Change exposure to %s ms', exposure)
            self.setup_soft_trigger(exposure=exposure)
        elif self._frame_number == self.MAX_NUMFRAMESTOACQUIRE:
            print('Maximum frame %s number reached for this acquisition, resetting soft trigger.' % str(self.MAX_NUMFRAMESTOACQUIRE))
            self.setup_soft_trigger(exposure=exposure)

        n_buffers = len(buffers)
        total_size_buffers = 0
        for buffer in buffers:
            total_size_buffers += buffer.shape[0]  # evaluate max number of images
        # Initialize buffers variables
        buffers_used = [False] * n_buffers  # Tracks if each buffer has been used
        buffer_index = 0  # Start with the first buffer
        images_in_buffer = 0  # Track number of images in the current buffer

        # here start the acq.loop calling multiple softriggers and saving the images in the specific buffer
        for _ in range(total_size_buffers): # max images to acquire in a single iteration
            self.merlin_cmd('SOFTTRIGGER') #this acquire an image

            if not self._frame_length:
                mpx_header = self.receive_data(nbytes=self.START_SIZE)
                size = int(mpx_header[4:])
                print('Received header: %s (%s)', size, mpx_header)

                framedata = self.receive_data(nbytes=size)
                skip = 0

                self._frame_length = self.START_SIZE + size
            else:
                framedata = self.receive_data(nbytes=self._frame_length)
                skip = self.START_SIZE

            self._frame_number += 1

            # decode the data into a numpy array (maybe this can be done after the acquisition in postprocessing)
            # Must skip first byte when loading data to avoid off-by-one error
            data = load_mib(framedata, skip=1 + skip).squeeze()
            data = self.rotate_img(data)
            #transfer it into it's buffer
            buffers[buffer_index][images_in_buffer, :, :] = data
            images_in_buffer += 1

            # Mark the current buffer as used
            if not buffers_used[buffer_index]:
                buffers_used[buffer_index] = True
                print("Buffer %s used for the first time." %str(buffer_index))
                buffer_shape = buffers[buffer_index].shape[0]

            # If the current buffer is full, move to the next buffer
            if images_in_buffer >= buffer_shape:
                buffer_index = (buffer_index + 1) % n_buffers
                images_in_buffer = 0
                print("Switched to buffer %s" %str(buffer_index))

            # Stop acquisition if received the stopper = True in case of threading,
            # so another thread write the variable self.stopper = True
            if self.stopper:
                break
            # Stop acquisition if the thread associated with stop_signal is no more alive
            if stop_signal.is_alive() == False:
                break


        #finish acq.loop and closing the acquisition protocol calling teardown_soft_trigger
        self.teardown_soft_trigger()
        # return the data stored in the buffers and the list of which buffer was used
        return buffers, buffers_used

    def prepare_acquisition_cRED_data(self, camera: str, binning: int, exposure: int, buffer_size, FPS_devider=1):
        """this need to be modified usign the new acquire_series_images function"""
        self.setup_soft_trigger(exposure=exposure)
        self.buffer_size = buffer_size
        self.binning = 1

        self.buffer = np.zeros((self.buffer_size, 512, 512), dtype=np.uint16)
        self.buffer_1 = np.zeros((self.buffer_size, 512, 512), dtype=np.uint16)
        self.zero_buffer = np.zeros((512, 512), dtype=np.uint16)

        self.FPS_exp = (1 / self.exposure) / FPS_devider
        self.sleeping = ((FPS_devider - 1) * (self.exposure / 1000))  # this is 0.0 if FPS_devider = 1
        # the fucntion is not reading the memory but its directly taking into account the dead time
        # timepix don't need a sleeping time
        # # response of the XF416R considering the exposure and the readoutime in rolling_shutter_mode
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
            answer = tkinter.messagebox.askyesno("continuous rotation data acquisition",
                                                 "ready! press Yes when ready")
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

                if not self._frame_length: #receive the frame_length from the detector at the first iteration
                    mpx_header = self.receive_data(continuous = True, nbytes=self.START_SIZE)
                    size = int(mpx_header[4:])
                    framedata = self.receive_data(continuous = True, nbytes=size)
                    skip = 0
                    self._frame_length = self.START_SIZE + size
                else:
                    framedata = self.receive_data(continuous = True, nbytes=self._frame_length)
                    skip = self.START_SIZE

                # Must skip first byte when loading data to avoid off-by-one error
                self.img = load_mib(framedata, skip=1 + skip).squeeze()


                self.buffer[self.i_, :, :] = self.img

                self.timings.append(self.tx)
                time.sleep(self.sleeping)
                self.i_ += 1
                if self.stage_thread.is_alive() == False:
                    self.stopper = True
                    self.teardown_soft_trigger()
        except Exception as err:
            print(err)
            print("using buffer_1")
            self.buffer_1_used = True
            while self.stopper == False:
                # 5.64 ms ± 175 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
                self.tx = (time.monotonic_ns() - self.t1) / 1000000000

                if not self._frame_length:  # receive the frame_length from the detector at the first iteration
                    mpx_header = self.receive_data(continuous=True, nbytes=self.START_SIZE)
                    size = int(mpx_header[4:])
                    framedata = self.receive_data(continuous=True, nbytes=size)
                    skip = 0
                    self._frame_length = self.START_SIZE + size
                else:
                    framedata = self.receive_data(continuous=True, nbytes=self._frame_length)
                    skip = self.START_SIZE

                # Must skip first byte when loading data to avoid off-by-one error
                self.img = load_mib(framedata, skip=1 + skip).squeeze()

                self.buffer_1[self.i_, :, :] = self.img

                self.timings.append(self.tx)
                time.sleep(self.sleeping)
                self.i_ += 1
                if self.stage_thread.is_alive() == False:
                    self.stopper = True
                    self.teardown_soft_trigger()

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

    def save_cRED_data(self, savingpath):  # RB what is cRED?
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
            image = self.correctCross(image)
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
                image = self.correctCross(image)
                imageio.imwrite(self.image_name, image)
                self.ii += 1

        os.chdir(self.current_dir)

    ########################################################################################################################
    # this function is already checked and tested! for timepix 1 factor = 3, for medipix 3, factor = 2?
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
        max_image_pixels = 512
        print("camera:", self.name)
        print("physical camera pixelsize in um", pixelsize)
        print("max image pixels^2", max_image_pixels)
        return pixelsize, max_image_pixels

    def load_calibration_table(self):
        cwd = os.getcwd()
        path = cwd + os.sep + r"adaptor/camera/lookup_table/merlin_table.txt"
        # print("path here:", path)
        # hardcoded path for jem2100f
        # path = r"C:\PyFast-ADT\pyfast_ADT_master\pyfast_adt\main\adaptor\camera\lookup_table/merlin_table.txt"
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

def test_movie(cam):
    print('\n\nMovie acquisition\n---\n')

    n_frames = 50
    exposure = 0.01

    t0 = time.perf_counter()

    frames = cam.get_movie(n_frames, exposure=exposure)

    t1 = time.perf_counter()

    avg_frametime = (t1 - t0) / n_frames
    overhead = avg_frametime - exposure

    print(f'\nExposure: {exposure}, frames: {n_frames}')
    print(f'\nTotal time: {t1 - t0:.3f} s - acq. time: {avg_frametime:.3f} s - overhead: {overhead:.3f}')

    for frame in frames:
        assert frame.shape == (512, 512)

def test_single_frame(cam):
    print('\n\nSingle frame acquisition\n---\n')

    n_frames = 100
    exposure = 0.01

    t0 = time.perf_counter()

    for i in range(n_frames):
        frame = cam.get_image()
        assert frame.shape == (512, 512)

    t1 = time.perf_counter()

    avg_frametime = (t1 - t0) / n_frames
    overhead = avg_frametime - exposure

    print(f'\nExposure: {exposure}, frames: {n_frames}')
    print(f'Total time: {t1 - t0:.3f} s - acq. time: {avg_frametime:.3f} s - overhead: {overhead:.3f}')

def test_plot_single_image(cam):
    arr = cam.get_image(exposure=0.1)

    import numpy as np

    arr = arr.squeeze()
    arr = np.flipud(arr)

    import matplotlib.pyplot as plt

    plt.imshow(arr.squeeze())
    plt.show()

if __name__ == '__main__':
    cam = Cam_merlin()
    img = test_plot_single_image(cam)
    for i in range(10):
        img = test_plot_single_image(cam)

