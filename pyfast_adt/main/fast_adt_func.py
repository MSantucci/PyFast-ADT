# core functions of fast-adt GUI
# mainly here are present the functiosn that the buttons of the GUI runs when clicked
import math
import os, cv2
import tkinter.filedialog
import pandas as pd
import numpy as np
import time
import tkinter as tk
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
from functools import partial
from tracking import Tomography_tracker
import datetime
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import threading
import imageio
from ast import literal_eval
from PIL import Image, ImageTk  # Required for displaying images with Tkinter
from tracking import InSituTracker
import csv



def fake(self):
    print("placeholder")

def go_to(self, velocity = 1):
    self.tem.last_angle = self.tem.get_stage()["a"] #deg
    angle = self.go_to_value()
    velocity = velocity

    self.tem.set_alpha(angle, velocity) #deg?
    print("go to %s" %str(angle))

def undo(self, velocity = 1):
    self.tem.set_alpha(self.tem.last_angle, velocity)
    print("undo to %s" % str(round(float(self.tem.last_angle), 2)))

def acquire_image(self, exposure, binning, processing):
    print("taking an image")
    self.tem.beam_blank(False)
    if self.tem.get_screen_position() != 'UP':
        if self.cam.is_cam_bottom_mounted():
            self.tem.move_screen(True)
    img = self.cam.acquire_image(exposure_time=exposure, binning=binning, processing=processing)
    if self.stream == True:
        pass
    return img

def acquire_image_and_show(self, exposure, binning, processing):
    print("taking an image")
    self.tem.beam_blank(False)
    if self.tem.get_screen_position() != 'UP':
        if self.cam.is_cam_bottom_mounted():
            self.tem.move_screen(True)
    img = self.cam.acquire_image_and_show(exposure_time=exposure, binning=binning, processing=processing)
    return img
# the next 3 function right now are working only moving C2, to control the beam.
# so they will work only in TEM mode for FEI
def get_beam_intensity(self):
    if self.stem_value() == True:
        if self.var11 == None:
            self.var11 = True
            print("saved beam characteristics for imaging")
            self.tem.get_intensity(slot=1)
            print("slot1 = ", self.tem.beam_intensity_1)
            self.change_beam_settings_image(slot=1)
        if self.var12 == None:
            self.var12 = False
        elif self.var12 == False:
            print("saved beam characteristic for diffraction")
            self.tem.get_intensity(slot=2)
            print("slot2 = ", self.tem.beam_intensity_2)
            self.var12 = None
            self.var11 = None
            self.change_beam_settings_image(slot=2)
        else: print("something wrong in fast_adt_func.py/function get_beam_intensity")

    else:
        if self.var11 == None:
            self.var11 = True
            print("saved beam characteristics for imaging")
            self.tem.get_intensity(slot=1)
            print("slot1 = ", self.tem.beam_intensity_1)
            self.change_beam_settings_image(slot=1)
        if self.var12 == None:
            self.var12 = False
        elif self.var12 == False:
            print("saved beam characteristic for diffraction")
            self.tem.get_intensity(slot=2)
            print("slot2 = ", self.tem.beam_intensity_2)
            self.var12 = None
            self.var11 = None
            self.change_beam_settings_image(slot=2)
        else: print("something wrong in fast_adt_func.py/function get_beam_intensity")

def set_image_setting(self):
    if self.tem.beam_intensity_1:
        print("beam in image setting")
        self.tem.set_intensity(self.tem.beam_intensity_1)
    else: print("no value stored, use get_beam_intensity first!")

def set_diff_setting(self):
    if self.tem.beam_intensity_2:
        print("beam in diffraction setting")
        self.tem.set_intensity(self.tem.beam_intensity_2)
    else: print("no value stored, use get_beam_intensity first!")
# need to be applied also for STEM mode

def test_beam_shift(self):
    if self.stem_value() == True:
        print("not available in stem mode")
        #####
        self.tem.test_tracking_stem(ub_class = self)
    else:
        raster_scanning_userbeamshift(self)

def calibrate_beam_shift(self):
    # to change here for stem in this case we need to find another procedure to get the beam shift
    if self.stem_value() == True:
        # beam_coord = beam_shift_vs_image_calibration(self)
        # if len(beam_coord) == 4:
        #     self.ub.beam_coordinates = beam_coord
        #     _, cam_size = self.haadf.get_camera_characteristic()
        #     if self.get_stem_image_size_value() == "FULL":
        #         cam_size = int(cam_size / (self.get_stem_binning_value()))
        #     elif self.get_stem_image_size_value() == "HALF":
        #         cam_size = int(cam_size / (2*self.get_stem_binning_value()))
        #     elif self.get_stem_image_size_value() == "QUARTER":
        #         cam_size = int(cam_size / (4*self.get_stem_binning_value()))
        #
        #     print("haadf size = ", cam_size)
        #     cam_size = 512
        #     angle_x, scaling_x, angle_y, scaling_y, _, list_target = self.ub.run_from_list(beam_coord, cam_size)
        #     self.ub.list_calibration = list_target
        #
        #     proj = self.tem.get_projection_mode()
        #     illumination_mode = self.tem.get_illumination_mode()
        #     mag = str(self.tem.get_magnification())
        #     kl = (self.tem.get_KL())
        #
        #     self.haadf_table[proj][illumination_mode][mag][2] = angle_x
        #     self.haadf_table[proj][illumination_mode][mag][3] = scaling_x
        #     self.haadf_table[proj][illumination_mode][mag][4] = angle_y
        #     self.haadf_table[proj][illumination_mode][mag][5] = scaling_y
        #
        #     write_cam_table(self)
        #     # print("camera_table updated correctly")
        print("not available in stem")
        return
    else:
        #print(self.cam_table)
        beam_coord = beam_shift_vs_image_calibration(self)
        if len(beam_coord) == 4:
            self.ub.beam_coordinates = beam_coord
            _, cam_size = self.cam.get_camera_characteristic()
            cam_size = int(cam_size / (self.binning_value()))
            print("camera size = ", cam_size)
            angle_x, scaling_x, angle_y, scaling_y, _, list_target = self.ub.run_from_list(beam_coord, cam_size)
            self.ub.list_calibration = list_target

            proj = self.tem.get_projection_mode()
            mag = str(self.tem.get_magnification())
            self.cam_table[proj][mag][2] = angle_x
            self.cam_table[proj][mag][3] = scaling_x
            self.cam_table[proj][mag][4] = angle_y
            self.cam_table[proj][mag][5] = scaling_y

            write_cam_table(self)
            #print("camera_table updated correctly")

def acquire_tracking_images(self, tracking_path = None, custom_param = None):
    # parameters to save for stepwise
    # self.tracking_images, self.track_angles
    # from the tracking processing self.tracking_positions, self.track_result self.tomo_tracker.orig_template
    #
    """this function for continuous tracking need both the tilt step and the tracking step, because it calculate the
    ratio between them to grab the images. the tilt step of the final acquisition determine the velocity of the stage.
    for the stepwise is not important, because the velocity parameter doesn't exist. due to the fact that tilt step and
    exposure determine the velocity of the stage, to be consistent it's better to use the same exposure and tilt step for
     the final data acquisition, moreover don't have a lot of mining expose for a long time because the image will be shitty!"""
    self.tracking_images_done = False
    self.tracking_done = False
    reset_tracking_images(self)
    self.tracking_images = None

    if self.get_tracking_method() == "debug":
        if tracking_path == None:
            tracking_path = r"L:\Marco\hardware_microscopes\TecnaiF30\sergi_track\Tracking\Tomography\Sequential\18\clean"
        #used for debugging to pass already acquired images
        series1 = os.listdir(tracking_path)
        series1.sort()
        series2 = []
        series2 = [tracking_path + os.sep + name for name in series1]
        print("#images loaded", len(series2), "\n", series2)
        self.tracking_images = series2
        #fake tracking angles as func of the len of the series
        track_angles = list(np.round(np.arange(0, len(series2)+5, 5, dtype=np.float32), 4))
        self.track_angles = track_angles
        method = "KF"
        visualization = True
        self.process_tracking_button.configure(text="continue debug (1st)", command=lambda: process_tracking_images(self,
                                            tracking_images=self.tracking_images, track_angles=track_angles, method=method, visualization=visualization))
        print("loaded images from %s for debugging, ready to process" % tracking_path)


    else:
        if custom_param == None:
            param = retrieve_parameters_for_acquisition(self, mode = "tracking")
        else:
            param = custom_param

        tem = self.tem
        cam = self.cam
        exp_type = param["experiment_type"]
        optics_mode = param["optics_mode"]
        stem_dwell_time = param["stem_pixeltime"]
        tem_image_time = param["tem_imagetime"]
        start_angle = param["start_angle"]
        final_angle = param["target_angle"]
        tilt_step = param["tilt_step"]
        exposure = param["exposure"]
        binning = param["binning"]
        processing = param["processing"]
        tracking_step = param["tracking_step"]
        rotation_speed = param["rotation_speed"]
        buffer_size = param["buffer_size"]

        if final_angle < start_angle: tracking_step = -tracking_step
        #track_angles = list(np.round(np.arange(start_angle, final_angle, tracking_step, dtype=np.float32), 4))
        track_angles = list(np.round(np.arange(start_angle, final_angle + tracking_step, tracking_step, dtype=np.float32), 4))
        if abs(track_angles[-1]) > abs(final_angle):
            track_angles.pop()
        print("line 131 - track_angles:", track_angles)
        self.track_angles = track_angles


        if optics_mode == "tem":
            self.cam.stop_liveview()
            if self.camera == "timepix1":
                # not necessary flag we can directly overwrite it in the timepix adaptor in acquire_image method
                #for timepix the image will be already corrected for the cross to 516pix**2
                img_buffer = np.zeros((len(track_angles), 516, 516), dtype=np.uint16)
                acquire_for_tracking = partial(self.cam.acquire_image, exposure_time=tem_image_time, binning=binning, processing="Unprocessed")

            else:
                cam_width = int(self.cam.get_camera_characteristic()[1]/binning)
                img_buffer = np.zeros((len(track_angles), cam_width, cam_width), dtype=np.uint16)
                acquire_for_tracking = partial(self.cam.acquire_image, exposure_time=tem_image_time, binning=binning, processing=processing)

        if optics_mode == "tem":
            if self.tem.get_screen_position() != 'UP':
                if self.cam.is_cam_bottom_mounted():
                    self.tem.move_screen(True)
        else:
            if self.tem.get_screen_position() != 'DOWN':
                if self.cam.is_cam_bottom_mounted():
                    self.tem.move_screen(False)
        #################################################################################################################
        if optics_mode == "stem" and self.haadf != None and self.brand in ["fei", "fei_temspy"]:
            camera_param = self.tem.tem.get_stem_acquisition_param()
            camera_param["binning"] = self.get_stem_binning_value()
            camera_param["image_size"] = self.get_stem_image_size_value()
            camera_param["dwell_time(s)"] = stem_dwell_time * 10 ** -6

            if camera_param["image_size"] == "FULL":
                cam_width = int(self.haadf.get_camera_characteristic()[1] / camera_param["binning"])
            elif camera_param["image_size"] == "HALF":
                cam_width = int(self.haadf.get_camera_characteristic()[1] / (camera_param["binning"] * 2))
            elif camera_param["image_size"] == "QUARTER":
                cam_width = int(self.haadf.get_camera_characteristic()[1] / (camera_param["binning"] * 4))

            self.haadf_cam_size = cam_width
            img_buffer = np.zeros((len(track_angles), cam_width, cam_width), dtype=np.uint16)
            self.haadf.tem.set_stem_acquisition_param(camera_param)
            # self.haadf.set_binning(camera_param["binning"])
            # self.haadf.set_image_size(camera_param["image_size"])
            # self.haadf.set_exposure(camera_param["dwell_time(s)"])
            print("dwell time U.I= ", camera_param["dwell_time(s)"])
            print("bin", self.haadf.get_binning(), "size:", self.haadf.get_image_size(), "dwell from temscript",
                  self.haadf.get_exposure())
            acquire_for_tracking = partial(self.haadf.acquire_image_fast)

        elif optics_mode == "stem" and self.brand == "jeol" and self.haadf != None:
            print("stem mode not implemented for jeol line 143 fast_adt_func.py")
            return
        #################################################################################################################
        # backlash correction:
        backlash_correction_alpha(self, exp_type, start_angle, final_angle, rotation_speed=self.speed_tracking, rotation_speed_cred=rotation_speed)
        if start_angle != round(self.tem.get_stage()["a"], 2) and exp_type == "stepwise":
            self.tem.set_alpha(start_angle, velocity = self.speed_tracking)
            time.sleep(0.5)

        # if self.tracking_precision_running != True:
        if self.get_init_position_value() == True and self.tracking_precision_running != True:
            # save the initial position of the stage
            self.init_position_stage_tracking = self.tem.get_stage()

        # if self.init_position_stage_tracking != None:
        if self.get_init_position_value() == True:
            print("correcting init_position tracking")
            backlash_correction_single_axis(self, tracking_initial_pos = {"x": self.init_position_stage_tracking["x"], "y":self.init_position_stage_tracking["y"], "z":self.init_position_stage_tracking["z"]}, speed = self.speed_tracking)             ####changed these stuff
        else:
            print("not correcting init_position tracking")
            backlash_correction_single_axis(self, speed=self.speed_tracking)  ####changed these stuff

        if self.tracking_precision_running == True:
            self.track_prec_init_pos = self.tem.get_stage()
        self.tem.beam_blank(False)


        #time starting here
        start_event = threading.Event()
        stop_event = threading.Event()
        self.t1_track = time.monotonic_ns()
        if exp_type == "stepwise":
            self.track_exp_type = "stepwise"
            for i, angl in enumerate(track_angles):
                if i == 0:
                    pass
                else: self.tem.set_alpha(angl, velocity = self.speed_tracking)
                time.sleep(0.5)
                img = acquire_for_tracking()
                img_buffer[i, :, :] = img
            ##time stopping here
            self.t2_track = (time.monotonic_ns() - self.t1_track) / 1000000000
        elif exp_type == "continuous" and optics_mode == "tem":
            self.track_exp_type = "continuous"
            tracking_dict = {}
            tracking_dict["start_angle"] = start_angle
            tracking_dict["target_angle"] = final_angle
            tracking_dict["rotation_speed"] = rotation_speed
            tracking_dict["experiment_type"] = exp_type
            tracking_dict["tracking_step"] = tracking_step
            tracking_dict["tracking_positions"] = []
            tracking_dict["mag"] = self.get_mag_value()
            tracking_dict["kl"] = self.get_KL_value()
            tracking_dict["stem_mode"] = False
            tracking_dict["tracking_method"] = self.get_tracking_method()


            thread_stage, thread_beam = self.tem.microscope_thread_setup(tracking_file = None, tracking_dict = tracking_dict, timer = self.t1_track, event = start_event, stop_event = stop_event)

            self.cam.prepare_acquisition_cRED_data(camera= self.camera, binning= binning, exposure= tem_image_time, buffer_size = buffer_size, FPS_devider=abs(tracking_step/tilt_step))
            self.tem.set_alpha(start_angle, velocity = self.speed_tracking)
            time.sleep(0.5)
            #tracking_data = [effective_time, effective_FPS, #collected_images]
            ###
            self.beam_thread_time = time.monotonic_ns()
            thread_beam.start()
            tracking_data = self.cam.acquisition_cRED_data(stage_thread=thread_stage, timer = self.t1_track, event = start_event, stop_event = stop_event)
            ###
            if tracking_data == "aborted":
                return
            self.cam.ref_timings["start_beam_thread_time"] = self.beam_thread_time
            ##time stopping here
            self.t2_track = (time.monotonic_ns() - self.t1_track) / 1000000000

            #images are saved in the buffer of the camera ## news: now there are 2 buffer and self.buffer_1_used = True to check it
            if self.cam.buffer_1_used == True:
                img_buffer = np.copy(np.concatenate((self.cam.buffer, self.cam.buffer_1), axis=0)) ###############################
            else:
                img_buffer = np.copy(self.cam.buffer)
            eff_track_step = round((abs(final_angle - start_angle)) / (tracking_data[2]-1), 6)
            #angle associated with each image
            if final_angle < start_angle: eff_track_step = -eff_track_step
            #track_angles = list(np.round(np.arange(start_angle, final_angle, eff_track_step, dtype=np.float32), 4))
            track_angles = list(np.round(np.arange(start_angle, final_angle + eff_track_step, eff_track_step, dtype=np.float32),4))
            if abs(track_angles[-1]) > abs(final_angle):
                track_angles.pop()
            print("track_angles = ", track_angles, "line 206")
            self.track_angles = track_angles


        elif exp_type == "continuous" and optics_mode == "stem" and self.get_tracking_method() == "prague_cred_method":
            self.track_exp_type = "continuous"
            #############################################################################
            tracking_dict = {}
            tracking_dict["start_angle"] = start_angle
            tracking_dict["target_angle"] = final_angle
            tracking_dict["rotation_speed"] = rotation_speed
            tracking_dict["experiment_type"] = exp_type
            tracking_dict["tracking_step"] = tracking_step
            tracking_dict["tracking_positions"] = []
            tracking_dict["mag"] = self.get_mag_value()
            tracking_dict["kl"] = self.get_KL_value()
            tracking_dict["illumination_mode"] = self.tem.get_illumination_mode()
            tracking_dict["projection_mode"] = self.tem.get_projection_mode()
            tracking_dict["experimental_mag"] = str(round(self.tem.get_magnification()))
            tracking_dict["stem_mode"] = True
            tracking_dict["experimental_kl"] = (self.tem.get_KL())
            tracking_dict["stem_binning_value"] = self.get_stem_binning_value()
            tracking_dict["ub_class"] = self
            tracking_dict["tracking_method"] = self.get_tracking_method()

            thread_stage, thread_beam = self.tem.microscope_thread_setup(tracking_file=None,
                                                                         tracking_dict=tracking_dict,
                                                                         timer=self.t1_track, event=start_event, stop_event = stop_event)
            step = int(((abs(start_angle) + abs(final_angle)) // 5))
            support_timings = []
            # main loop
            i = 0
            for (worker_beam, start_event, result_list), (worker_stage, _) in zip(thread_beam, thread_stage):
                worker_beam.start()
                #worker_stage.start()

                self.haadf.prepare_acquisition_cRED_data(camera=self.haadf, binning=camera_param["binning"],
                                                         exposure=camera_param["dwell_time(s)"],
                                                         image_size=camera_param["image_size"],
                                                         buffer_size=int(buffer_size//step),
                                                         FPS_devider=abs(tracking_step / tilt_step))

                # backlash correction:
                backlash_correction_alpha(self, exp_type, start_angle, final_angle, rotation_speed=self.speed_tracking, rotation_speed_cred=rotation_speed)
                self.tem.beam_blank(False)

                self.tem.set_alpha(start_angle, velocity=tracking_dict["rotation_speed"])
                backlash_correction_single_axis(self)
                # tracking_data = [effective_time, effective_FPS, #collected_images]
                ###
                self.beam_thread_time = time.monotonic_ns()

                tracking_data = self.haadf.acquisition_cRED_data(stage_thread=worker_stage, timer=self.t1_track, event=start_event, stop_event = stop_event)
                ####
                if tracking_data == "aborted":
                    return
                self.haadf.ref_timings["start_beam_thread_time"] = self.beam_thread_time

                ##time stopping here
                self.t2_track = (time.monotonic_ns() - self.t1_track) / 1000000000

                # images are saved in the buffer of the camera ## news: now there are 2 buffer and self.buffer_1_used = True to check it
                if self.haadf.buffer_1_used == True:
                    img_buffer = np.copy(np.concatenate((self.haadf.buffer, self.haadf.buffer_1), axis=0))  ###############################
                else:
                    img_buffer = np.copy(self.haadf.buffer)
                eff_track_step = round((abs(final_angle - start_angle)) / (tracking_data[2] - 1), 6)
                # angle associated with each image
                if final_angle < start_angle: eff_track_step = -eff_track_step
                # track_angles = list(np.round(np.arange(start_angle, final_angle, eff_track_step, dtype=np.float32), 4))
                track_angles = list(
                    np.round(np.arange(start_angle, final_angle + eff_track_step, eff_track_step, dtype=np.float32), 4))
                if abs(track_angles[-1]) > abs(final_angle):
                    track_angles.pop()
                print("track_angles = ", track_angles, "line 206")
                if i == 0:
                    support_img_buffer = np.copy(img_buffer[:-1])
                    support_tracking_angles = track_angles[:-1]
                    support_timings.append(self.haadf.timings[:-1])
                else:
                    support_img_buffer = np.concatenate((support_img_buffer, img_buffer[:-1]), axis = 0)
                    support_tracking_angles += track_angles[:-1]
                    support_timings.append(self.haadf.timings[:-1]) # these are not concatenated because every thread has its own timing starting from 0 always

                print("line 437 check iteration:", i, support_img_buffer.shape, support_tracking_angles, support_timings)
                i += 1
            self.track_angles = support_tracking_angles
            img_buffer = support_img_buffer
            self.haadf.timings = support_timings

        elif exp_type == "continuous" and optics_mode == "stem":
            self.track_exp_type = "continuous"
            #############################################################################
            tracking_dict = {}
            tracking_dict["start_angle"] = start_angle
            tracking_dict["target_angle"] = final_angle
            tracking_dict["rotation_speed"] = rotation_speed
            tracking_dict["experiment_type"] = exp_type
            tracking_dict["tracking_step"] = tracking_step
            tracking_dict["tracking_positions"] = []
            tracking_dict["mag"] = self.get_mag_value()
            tracking_dict["kl"] = self.get_KL_value()
            tracking_dict["illumination_mode"] = self.tem.get_illumination_mode()
            tracking_dict["projection_mode"] = self.tem.get_projection_mode()
            tracking_dict["experimental_mag"] = str(round(self.tem.get_magnification()))
            tracking_dict["stem_mode"] = True
            tracking_dict["experimental_kl"] = (self.tem.get_KL())
            tracking_dict["stem_binning_value"] = self.get_stem_binning_value()
            tracking_dict["ub_class"] = self
            tracking_dict["tracking_method"] = self.get_tracking_method()

            thread_stage, thread_beam = self.tem.microscope_thread_setup(tracking_file=None, tracking_dict=tracking_dict, timer = self.t1_track, event = start_event, stop_event = stop_event)

            self.haadf.prepare_acquisition_cRED_data(camera=self.haadf, binning=camera_param["binning"], exposure=camera_param["dwell_time(s)"], image_size = camera_param["image_size"],
                                                   buffer_size=buffer_size, FPS_devider=abs(tracking_step / tilt_step))
            self.tem.set_alpha(start_angle, velocity=self.speed_tracking)
            time.sleep(0.5)
            # tracking_data = [effective_time, effective_FPS, #collected_images]
            ###
            self.beam_thread_time = time.monotonic_ns()
            thread_beam.start()
            tracking_data = self.haadf.acquisition_cRED_data(stage_thread=thread_stage, timer = self.t1_track, event = start_event, stop_event = stop_event)
            ####
            if tracking_data == "aborted":
                return
            self.haadf.ref_timings["start_beam_thread_time"] = self.beam_thread_time

            ##time stopping here
            self.t2_track = (time.monotonic_ns() - self.t1_track) / 1000000000

            # images are saved in the buffer of the camera ## news: now there are 2 buffer and self.buffer_1_used = True to check it
            if self.haadf.buffer_1_used == True:
                img_buffer = np.copy(
                    np.concatenate((self.haadf.buffer, self.haadf.buffer_1), axis=0))  ###############################
            else:
                img_buffer = np.copy(self.haadf.buffer)
            eff_track_step = round((abs(final_angle - start_angle)) / (tracking_data[2] - 1), 6)
            # angle associated with each image
            if final_angle < start_angle: eff_track_step = -eff_track_step
            # track_angles = list(np.round(np.arange(start_angle, final_angle, eff_track_step, dtype=np.float32), 4))
            track_angles = list(
                np.round(np.arange(start_angle, final_angle + eff_track_step, eff_track_step, dtype=np.float32), 4))
            if abs(track_angles[-1]) > abs(final_angle):
                track_angles.pop()
            print("track_angles = ", track_angles, "line 206")
            self.track_angles = track_angles

        #############################################################################
        # this is taking the interpolation of the tracking to be used in the acquisition
        if exp_type == "continuous":
            # self.gonio_fit_tracking = self.tem.result[4]
            # range_exp = list(range(int(np.round(start_angle, 0)), int(np.round(final_angle, 0))))
            # print("slope:", self.tem.result[1])
            # print("timers:", self.tem.result[2])
            # print("angles:", self.tem.result[3])
            # print("fit tracking \nx = ", range_exp, " \ny = ", self.gonio_fit_tracking(np.deg2rad(range_exp)))
            pass


        if self.cam.is_cam_bottom_mounted():
            self.tem.move_screen(False)
        self.tem.beam_blank(True)

        # if self.camera == "timepix1":
        #     if exp_type == "stepwise":
        #         self.cam.buffer = img_buffer.copy()
        #     corrected_buffer = np.zeros((img_buffer.shape[0], 516, 516), dtype=np.uint16)
        #     for count, image in enumerate(self.cam.buffer):
        #         image = self.cam.correctCross(image)
        #         corrected_buffer[count, :, :] = image
        #     img_buffer = corrected_buffer

        self.tracking_images = np.copy(img_buffer)

    self.tracking_images_done = True


    return self.tracking_images, self.track_angles

def process_tracking_images(self, tracking_images, track_angles, method, visualization = False):
    if self.tracking_images_done == False:
        print("acquire tracking images first")
        return
    else:
        if method == "manual":
            if self.tomo_tracker is None:
                self.tomo_tracker = Tomography_tracker(images=tracking_images, visualization=visualization, dt = None, exp_type = self.track_exp_type)
            manual_res = self.tomo_tracker.manual_tracking(images = tracking_images, visualization = False)
            manual_res = [(x, y) for ((x, y), _) in manual_res]
            positions = manual_res
            self.process_tracking_button.configure(text="Process (1st)",
                                                   command=lambda: process_tracking_images(self,
                                                   tracking_images = self.tracking_images, track_angles =self.track_angles, method = self.get_tracking_method(), visualization = visualization))
            self.plot_result = self.tomo_tracker.plot_tracking()
            if len(self.track_result["CC"]) != 0:
                self.track_result["manual"] = manual_res
            else:
                self.track_result = {"CC": [], "KF": [], "pureKF": [], "manual": manual_res}

        elif not method == "manual":
            # workflow
            # init_the_class track = Tomography_tracker(series = list(paths), visualization=False)
            # or track = Tomography_tracker(images = np.array(3D), visualization=False) and run it by track.main()
            # plot result = track.plot_tracking() just last plot
            if self.cont_value():
                self.dt = 1/self.FPS
            else:
                self.dt = 0.1
            self.tomo_tracker = Tomography_tracker(images=tracking_images, visualization=visualization, dt = self.dt)
            automatic_res = self.tomo_tracker.main()
            self.plot_result = self.tomo_tracker.plot_tracking()
            patchworkCC = []
            CC = []
            KF = []
            pureKF = []
            manual = []
            for res in automatic_res:
                pureKF.append(res[0])
                patchworkCC.append(res[1])
                KF.append(res[2])
                CC.append(res[3])

            # self.support1.append((tuple(self.predicted_position), tuple(self.template_matching_result), tuple(self.filtered_position), self.CC_positions))
            self.track_result = {"CC":CC, "patchworkCC":patchworkCC, "pureKF":pureKF, "KF":KF, "manual": manual}

            if method == "tracking_precision":
                method = "KF"
            # here is decided only the type of output from the previous dictionary
            positions = self.track_result[method]

            self.process_tracking_button.configure(text = "continue manual (2nd)", command=lambda: process_tracking_images(self,
                                                                tracking_images = self.tracking_images, track_angles = self.track_angles, method = "manual", visualization = visualization))

        self.tracking_positions = []
        for (i, angle), pos in zip(enumerate(track_angles), positions):
            self.tracking_positions.append((angle, pos[0], pos[1]))

        # to add here, in cred increase by linearization the number of tracking_positions
        #
        #

        self.tracking_images_done = True
        self.tracking_done = True

        return self.tracking_positions, self.track_result

def reset_tracking_images(self):
    #self.tracking_images_done = False ### in this way we can re-process the images multiple times if you don't like it
    self.tracking_done = False
    self.tracking_positions = []
    self.tomo_tracker = None
    self.second_iteration = False
    self.track_result = None
    self.track_exp_type = None
    #self.haadf_cam_size = None
    self.process_tracking_button.configure(text="Process (1st)", command=lambda: process_tracking_images(self,
                                                                                   tracking_images=self.tracking_images,
                                                                                   track_angles=self.track_angles,
                                                                                   method=self.get_tracking_method()))

def display_crystal_tracking(self, method):
    #self.track_result = {"CC": CC, "KF": KF, "pureKF": pureKF, "manual": manual}
    if self.tomo_tracker is None:
        print("Please process tracking images first")
        return
    elif method != None:
        self.tomo_tracker.display_tracking(images = self.tracking_images, tracking_dict = self.track_result, method = method, beam_size_diff=None)

def generate_tracking_file(self, method, text = "tracking.txt", custom_param = None):
    if custom_param == None:
        param = retrieve_parameters_for_acquisition(self)
    else:
        param = custom_param

    if self.stem_value() != True:
        _, cam_size = self.cam.get_camera_characteristic()
        cam_size = int(cam_size / (self.binning_value()))
    else:
        cam_size = self.haadf_cam_size

    self.tracking_positions = []
    for (i, angle), pos in zip(enumerate(self.track_angles), self.track_result[method]):
        self.tracking_positions.append((angle, pos[0]-round(cam_size/2, 0), -1*(pos[1]-round(cam_size/2, 0))))
        #now (0,0) is in the center of the camera and the space is oriented as you see it in the fluorescent screen
        # and as a normal cartesian space (top right is +,+: bottom right is +,-: top left is -,+: bottom left is -,-)

    param["tracking_positions"] = self.tracking_positions

    write_tracking_file(self, text, param["start_angle"], param["target_angle"], param["tilt_step"], param["rotation_speed"], param["experiment_type"], param["tracking_step"], param["tracking_positions"])

    print("tracking.txt generated in: %s" %os.getcwd(), "\n",len(self.tracking_positions),"tracking_positions saved in the tracking file")

def load_tracking_file(self):
    try:
        text = askopenfilename()
        read_tracking_file(self, text)
    except FileNotFoundError:
        print("File not found, please generate a tracking file first and try again")

def initialize_beam_position(self):
    evaluate_timings(self)
    return
    """ probably to drop as method"""
    if self.get_tracking_method() == "debug":
        real_space = np.zeros((512,512,3), dtype = np.uint8)
    else:
        self.parameters_experiment = retrieve_parameters_for_acquisition(self)
        param = self.parameters_experiment
        exp_type = param["experiment_type"]
        optics_mode = param["optics_mode"]
        stem_dwell_time = param["stem_pixeltime"]
        tem_image_time = param["tem_imagetime"]
        start_angle = param["start_angle"]
        final_angle = param["target_angle"]
        tilt_step = param["tilt_step"]
        exposure = param["exposure"]
        binning = param["binning"]
        processing = param["processing"]
        tracking_step = param["tracking_step"]
        rotation_speed = param["rotation_speed"]
        buffer_size = param["buffer_size"]

        if param["start_angle"] != round(self.tem.get_stage()["a"],2):
            self.tem.set_alpha(param["start_angle"], velocity = self.speed_tracking)
            time.sleep(0.5)

        if optics_mode == "stem":
            if self.tem.get_screen_position() != 'DOWN':
                if self.cam.is_cam_bottom_mounted():
                    self.tem.move_screen(False)
                self.tem.beam_blank(False)
        else:
            if self.tem.get_screen_position() != 'UP':
                if self.cam.is_cam_bottom_mounted():
                    self.tem.move_screen(True)
                self.tem.beam_blank(False)

        if param["optics_mode"] == "stem" and self.brand in ["fei", "fei_temspy"]:
            camera_param = self.tem.tem.get_stem_acquisition_param()
            camera_param["image_size"] = "FULL"
            if binning == 1 or binning == 2 or binning == 4:
                camera_param["binning"] = binning
            else:
                camera_param["binning"] = 4
            camera_param["dwell_time(s)"] = stem_dwell_time * 10 ** -6
            self.tem.tem.set_stem_acquisition_parameters(camera_param)
            real_space = self.tem.tem.acquire("HAADF")
        else:
            real_space = self.cam.acquire_image(exposure_time=tem_image_time, binning=binning, processing=processing)

    if self.tem.get_screen_position() != 'UP':
        if self.cam.is_cam_bottom_mounted():
            self.tem.move_screen(True)
        time.sleep(0.5)

    drawer = Tomography_tracker()
    drawer.img = np.copy(real_space)

    cv2.namedWindow("Select object")
    cv2.namedWindow("Diffraction")
    cv2.putText(drawer.img, 'right click: select beam position, Enter: save and continue', (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("Select object", drawer.img)  # thsi is outside otherwise the image is not showing the circle because is overwritten everytime
    pos_img1 = cv2.getWindowImageRect("Select object")

    cv2.moveWindow("Diffraction", pos_img1[0]+pos_img1[2], pos_img1[1])
    cv2.setMouseCallback("Select object", drawer.draw_circle, param = True)
    stop = False
    beam = None

    if self.tem.get_projection_mode() == 'IMAGING':
        self.tem.diffraction(checked_diff_value=True, kl = self.get_KL_value)
    # else: if you are in diffraction you should be already at the good one!
    #     kl = self.kl_index_table[self.get_KL_value]
    #     self.set_KL(kl)
    if self.cam.is_streamable() == True:
        self.cam.start_live_view()
        while stop == False:
            update_beam = len(drawer.support_manual)

            cv2.putText(drawer.img, 'right click: select beam position, Enter: save and continue', (0, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            key = cv2.waitKey(10)
            if key == 13 and beam != None:  # 13 is the enter key
                stop = True
            elif key == 32:
                pass  # 32 is the space bar

            if len(drawer.support_manual) > update_beam:
                print("x,y:", drawer.support_manual[-1])
                # self.tem.get_beam_shift()
                # self.tem.move_beam_to..... function to apply beam shift in the reference of the image ##################################
                # this need to be calibrated for the beam shift in the reference of the image
    else:
        if self.camera == "timepix1":
            acquire_for_live = partial(self.cam.acquire_image, binning=binning, processing="Unprocessed")
        else:
            acquire_for_live = partial(self.cam.acquire_image, binning=binning, processing=processing)

        while stop == False:

            update_beam = len(drawer.support_manual)
            try:
                diff_space = acquire_for_live(exposure_time = self.exposure_value()) # this is dynamic by the GUI
            except:
                pass #this because if the exposure is changed can become 0 and the camera will not acquire
            diff_space = cv2.resize(diff_space, (diff_space.shape[1] // 2, diff_space.shape[0] // 2))

            cv2.putText(drawer.img, 'right click: select beam position, Enter: save and continue', (0,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)
            cv2.imshow("Diffraction", diff_space)

            key = cv2.waitKey(10)
            if key == 13 and beam != None: #13 is the enter key
                stop = True
            elif key == 32: pass #32 is the space bar

            if len(drawer.support_manual) > update_beam:
                print("x,y:", drawer.support_manual[-1])
                #self.tem.get_beam_shift()
                #self.tem.move_beam_to..... function to apply beam shift in the reference of the image ##################################
                # this need to be calibrated for the beam shift in the reference of the image

    cv2.destroyAllWindows()
    beam_pos = drawer.support_manual[-1][0]
    self.tem.beam_blank(True)

    print("beam position initialized")

def start_experiment(self):
    ###### initialization of the parameters #######################################################################
    self.stop_signal = False
    self.update_experiment_number() # update the experiment counter
    saving_path = self.exp_name(full_path=True)
    saving_path_images = saving_path + os.sep + "tiff"
    try:
        param = self.parameters_experiment
    except AttributeError:
        param = retrieve_parameters_for_acquisition(self)

    exp_type = param["experiment_type"]
    optics_mode = param["optics_mode"]
    stem_dwell_time = param["stem_pixeltime"]
    tem_image_time = param["tem_imagetime"]
    start_angle = param["start_angle"]
    final_angle = param["target_angle"]
    tilt_step = param["tilt_step"]
    exposure = param["exposure"]
    binning = param["binning"]
    processing = param["processing"]
    tracking_step = param["tracking_step"]
    rotation_speed = param["rotation_speed"]
    buffer_size = param["buffer_size"]
    tracking_method = param["tracking_method"]
    if final_angle < start_angle:
        tilt_step = -tilt_step

    tracking_dict = {}
    # these are all retrieve from the GUI
    tracking_dict["start_angle"] = start_angle
    tracking_dict["target_angle"] = final_angle
    tracking_dict["rotation_speed"] = rotation_speed
    tracking_dict["experiment_type"] = exp_type
    tracking_dict["tracking_step"] = tracking_step
    tracking_dict["mag"] = self.get_mag_value()
    tracking_dict["kl"] = self.get_KL_value()
    # experimental parameters from microscope
    tracking_dict["illumination_mode"] = self.tem.get_illumination_mode()
    tracking_dict["projection_mode"] = self.tem.get_projection_mode()
    tracking_dict["experimental_mag"] = str(round(self.tem.get_magnification()))
    tracking_dict["tracking_method"] = tracking_method

    # prepare the stuff for the semi-manual stepwise acquisition ######################################################
    if self.get_tracking_method() == "semi-manual stepwise":

        now = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        print("starting experiment")
        stage_start_position = self.tem.get_stage()
        stage_start_position_std = self.tem.get_stage(standard=True)
        exp_type = "stepwise"
        exp_angle = list(np.round(np.arange(start_angle, final_angle + tilt_step, tilt_step, dtype=np.float32), 4))
        # exp_angle = list(np.round(np.linspace(start_angle, final_angle, (abs(final_angle-start_angle)/tilt_step)+1, dtype=np.float32), 4))
        if abs(exp_angle[-1]) > abs(final_angle):
            exp_angle.pop()
        print("to check: exp_angle = ", exp_angle)
        _, img_shape = self.cam.get_camera_characteristic()

        img_buffer = np.zeros((len(exp_angle), round(img_shape / binning), round(img_shape / binning)), dtype=np.uint16)
        self.tem.beam_blank(False)

        # prepare the camera to acquire the data # acq. loop here, take an image and tilt ##########################
        t1 = time.monotonic_ns()
        print("\nstart semi-manual stepwise acquisition here\n")
        semi_manual_stepwise(self, exp_angle, exposure, binning, processing, img_buffer, self.separator1)
        print("\nfinished semi-manual stepwise acquisition here\n")
        ############################################################################################################
        t2 = (time.monotonic_ns() - t1) / 1000000000
        self.cam.buffer = self.manual_img_buffer

        frames = len(exp_angle)
        final_angle = exp_angle[-1]

        self.gonio_fit = [None, None]

        if self.stem_value() == True:
            self.exp_pixelsize = image_pixelsize(self)[1]
        else:
            self.exp_pixelsize = image_pixelsize(self)
        if self.tem.get_screen_position() != 'DOWN':
            if self.cam.is_cam_bottom_mounted():
                self.tem.move_screen(False)

        self.tem.beam_blank(True)

        ## prepare the destination dir # here we are saving the data #
        os.makedirs(saving_path_images, exist_ok=True)

        if self.camera in ["us4000", "us2000"]:
            self.cam.save_cRED_data(savingpath=saving_path_images, processing=processing)
        else:
            self.cam.save_cRED_data(savingpath=saving_path_images)

        rotation_speed = 1


        # prepare the output log file
        self.resume_experiment = {"date": now,
                                  "microscope": self.brand,
                                  "voltage": self.tem.get_voltage(),
                                  "wavelength": calculate_wavelength(self, self.tem.get_voltage()),
                                  "optics_mode": optics_mode,
                                  "spotsize": self.tem.get_spotsize(),
                                  "exp_type": exp_type,
                                  "KL": self.tem.get_KL(),
                                  "C2": round(self.tem.get_intensity(), 4),
                                  "camera": self.camera,
                                  "saturation_limit": 65535,
                                  "exp_time": t2,
                                  "exposure": exposure,
                                  "aperpixel": self.exp_pixelsize,
                                  "binning": binning,
                                  "image_size": tuple(self.cam.buffer.shape[1:]),
                                  "frames": frames,
                                  "start_angle": start_angle,
                                  "final_angle": final_angle,
                                  "angular_range": abs(final_angle - start_angle),
                                  "rotation_speed": rotation_speed,
                                  "rotation_speed_fitted": self.gonio_fit[1],
                                  "eff_rotation_speed": abs(final_angle - start_angle) / t2,
                                  "tilt_step": tilt_step,
                                  "eff_tilt_step": np.sign(tilt_step) * (
                                      round(abs(final_angle - start_angle) / (frames - 1), 5)),
                                  "stage_start_position": stage_start_position,
                                  "stage_end_position": self.tem.get_stage(),
                                  "tracked_positions": tracking_dict["tracking_positions"],
                                  "userbeam_position_start": None,
                                  "userbeam_positions": None}

        self.resume_experiment_with_units = {"date": now,
                                             "Microscope": str(self.cam_table["microscope"]),
                                             "Microscope_tag": str(self.brand),
                                             "Voltage": str(self.tem.get_voltage()) + " kV",
                                             "Wavelength": str(calculate_wavelength(self, self.tem.get_voltage())) + " A",
                                             "Optics_mode": str(optics_mode),
                                             "spotsize": str(self.tem.get_spotsize()) + " a.u.",
                                             "Exp_type": str(exp_type),
                                             "KL": str(self.tem.get_KL()) + " mm",
                                             "C2": str(round(self.tem.get_intensity(), 4)) + " %",
                                             "Camera": str(self.cam_table["camera"]),
                                             "Camera_tag": str(self.camera),
                                             "Saturation_limit": 65535,
                                             "Experiment_time": str(t2) + " s",
                                             "Exposure": str(exposure) + " ms",
                                             "Aperpixel": str(self.exp_pixelsize) + " A-1",
                                             "Binning": str(binning),
                                             "Image_size": str(tuple(self.cam.buffer.shape[1:])) + " pixels",
                                             "Frames": str(frames),
                                             "Start_angle": str(start_angle) + " deg",
                                             "Final_angle": str(final_angle) + " deg",
                                             "Angular_range": str(abs(final_angle - start_angle)) + " deg",
                                             "Rotation_speed": str(round(rotation_speed, 5)) + " deg/s",
                                             "rotation_speed_fitted": str(self.gonio_fit[1]) + "deg/s",
                                             "Eff_rotation_speed": str(
                                                 round(abs(final_angle - start_angle) / t2, 5)) + " deg/s",
                                             "Tilt_step": str(tilt_step) + " deg/img",
                                             "Eff_Tilt_step": str(np.sign(tilt_step) * (
                                                 round(abs(final_angle - start_angle) / (frames - 1), 5))) + " deg/img",
                                             "Stage_start_position": str(stage_start_position_std),
                                             "Stage_end_position": str(self.tem.get_stage(standard=True)),
                                             "Tracked_positions (angle, x, y)": str(tracking_dict["tracking_positions"]),
                                             "Userbeam_position_start": None,
                                             "Userbeam_positions": None}

        self.tracking_dictionary = tracking_dict
        # write tracking images and file here
        val = write_pets_file(self, path=saving_path, pets_default_values="pets_default_values.txt")
        write_report_experiment(self, path=saving_path, add_val=val)
        return
    #### prepare the tracking positions if present ##################################################################
    elif self.get_tracking_method() != "no tracking":
        tracking_dict["tracking_positions"] = self.tracking_positions
        # calculation of the timings for the tracking in cred
        if exp_type == "continuous":
            supp = []
            for angle, x, y in tracking_dict["tracking_positions"]:
                supp.append(angle)

            #track_times = list(self.gonio_fit_tracking(supp))

            # now this is relative to 0
            if self.stem_value() != True:
                track_times = self.cam.timings
                tot_time = (self.cam.ref_timings["end_acq_cred_time"] - self.cam.ref_timings[
                    "start_stage_thread_time"]) / 10 ** 9
                track_times = [x - track_times[0] for x in track_times]
            else:
                if self.get_tracking_method() != "prague_cred_method":
                    track_times = self.haadf.timings
                    tot_time = (self.haadf.ref_timings["end_acq_cred_time"] - self.haadf.ref_timings[
                        "start_stage_thread_time"]) / 10 ** 9
                    #track_times = [(x - track_times[0]) for x in track_times]
                    #track_times = [(x / track_times[-1]) *tot_time for x in track_times]
                    # track_times = [x - track_times[0] for x in track_times]
                    track_times = [x - track_times[0] for x in track_times]
                    print("track times", track_times)
                else: # prague method
                    support_times = []
                    for track_times in self.haadf.timings:
                        #track_times = self.haadf.timings
                        tot_time = (self.haadf.ref_timings["end_acq_cred_time"] - self.haadf.ref_timings[
                            "start_stage_thread_time"]) / 10 ** 9
                        track_times = [x - track_times[0] for x in track_times]
                        #print("track times", track_times)
                        support_times.append(track_times)
                    track_times = support_times
            tracking_dict["tracking_times"] = track_times

    else:
        ##### else if no position are provided just don't use tracking ################################################
        tracking_dict["tracking_positions"] = []

    # this is the time of the start of the acquisition ################################################################
    # starter signal here #
    start_event = threading.Event()
    stop_event = threading.Event()
    self.t1_acq = time.monotonic_ns()
    if self.stem_value() != True:
        tracking_dict["ub_class"] = self.ub
        tracking_dict["stem_mode"] = False
        userbeam_position_start = self.tem.get_beam_shift()
        self.thread_stage, self.thread_beam = self.tem.microscope_thread_setup(tracking_file=None,
                                                                               tracking_dict=tracking_dict, timer = self.t1_acq, event = start_event, stop_event = stop_event)
                                                                               #,ub_class=self.ub)
    else:
        tracking_dict["ub_class"] = self
        tracking_dict["stem_mode"] = True
        tracking_dict["experimental_kl"] = (self.tem.get_KL())
        tracking_dict["stem_binning_value"] = self.get_stem_binning_value()

        userbeam_position_start = self.haadf.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        self.thread_stage, self.thread_beam = self.tem.microscope_thread_setup(tracking_file=None,
                                                                               tracking_dict=tracking_dict, timer = self.t1_acq, event = start_event, stop_event = stop_event)
                                                                               #,ub_class= self)
    now = datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S')
    print("starting experiment")
    stage_start_position = self.tem.get_stage()
    stage_start_position_std = self.tem.get_stage(standard = True)

    if self.tem.get_screen_position() != 'UP':
        if self.cam.is_cam_bottom_mounted():
            self.tem.move_screen(True)

    # prague_cred_method start here #############################################################################
    #insert here the for loop if cred prague
    if exp_type == "continuous" and optics_mode == "stem" and self.get_tracking_method() == "prague_cred_method":
        step = int((abs(start_angle) + abs(final_angle)) // 5)
        tracking_dict["tracking_method"] = "prague_cred_method"
        # main loop
        i = 0
        for (worker_beam, start_event, result_list), (worker_stage, _) in zip(self.thread_beam, self.thread_stage):
            worker_beam.start()
            #worker_stage.start()

            # backlash correction:
            backlash_correction_alpha(self, exp_type, start_angle, final_angle, rotation_speed=0.7, rotation_speed_cred=rotation_speed)
            if start_angle != round(self.tem.get_stage()["a"], 2) and exp_type == "stepwise":
                self.tem.set_alpha(start_angle, velocity=self.speed_tracking)
                time.sleep(0.5)
            backlash_correction_single_axis(self)

            if exp_type == "continuous":
                self.cam.prepare_acquisition_cRED_data(camera=self.camera, binning=binning, exposure=exposure,
                                                       buffer_size=int(buffer_size//step))
                self.tem.beam_blank(False)
                time.sleep(0.33)
                ###
                self.beam_thread_time = time.monotonic_ns()
                self.thread_beam.start()
                self.result_acquisition = self.cam.acquisition_cRED_data(stage_thread=self.thread_stage, timer=self.t1_acq,
                                                                         event=start_event, stop_event = stop_event)
                ####
                if self.result_acquisition == "aborted":
                    return
                self.cam.ref_timings["start_beam_thread_time"] = self.beam_thread_time
                self.tem.beam_blank(True)
                t2 = self.result_acquisition[0]
                frames = (self.result_acquisition[2])
                final_angle = self.tem.get_stage(standard=True)["a"][0]

            if i == 0:
                support_img_buffer = np.copy(self.cam.buffer[:-1])
                #support_tracking_angles = track_angles[:-1]
                #support_timings = self.haadf.timings[:-1]
            else:
                support_img_buffer = np.concatenate((support_img_buffer, self.cam.buffer[:-1]), axis=0)
                #support_tracking_angles += track_angles[:-1]
                #support_timings.append(self.haadf.timings[:-1])

            print("line 437 check iteration:", i, support_img_buffer.shape)
            i += 1
        self.cam.buffer = support_img_buffer
        #img_buffer = support_img_buffer
        #self.haadf.timings = support_timings
        self.gonio_fit = [None, None] # just for simplicity now

    ### prague method finish here #############################################################################
    ### here we can do the evaluation of the tracking precision method!
    elif self.get_tracking_method() == "tracking_precision":
        tracking_precision_run(self, tracking_dict)
        return

    ###### starting here standard acquisition with or without "a-priori "tracking #############################
    else:
        tracking_dict["tracking_method"] = None
        # backlash correction:
        backlash_correction_alpha(self, exp_type, start_angle, final_angle, rotation_speed=self.speed_tracking, rotation_speed_cred=rotation_speed)
        if start_angle != round(self.tem.get_stage()["a"], 2) and exp_type == "stepwise":
            self.tem.set_alpha(start_angle, velocity = self.speed_tracking)
            time.sleep(0.5)
        if self.get_tracking_method != "no tracking":
            backlash_correction_single_axis(self, tracking_initial_pos={"x": self.init_position_stage_tracking["x"],
                                                                        "y": self.init_position_stage_tracking["y"],
                                                                        "z": self.init_position_stage_tracking["z"]},
                                            speed=self.speed_tracking)  ####changed these stuff

        else:
            backlash_correction_single_axis(self)

        if exp_type == "continuous":
            self.cam.prepare_acquisition_cRED_data(camera = self.camera, binning = binning, exposure = exposure, buffer_size = buffer_size)
            self.tem.beam_blank(False)
            time.sleep(0.33)
            ###
            self.beam_thread_time = time.monotonic_ns()
            self.thread_beam.start()
            self.result_acquisition = self.cam.acquisition_cRED_data(stage_thread=self.thread_stage, timer = self.t1_acq, event = start_event, stop_event = stop_event)
            ####
            if self.result_acquisition == "aborted":
                return
            self.cam.ref_timings["start_beam_thread_time"] = self.beam_thread_time
            t2 = self.result_acquisition[0]
            frames = (self.result_acquisition[2])
            final_angle = self.tem.get_stage(standard = True)["a"][0]

        elif exp_type == "stepwise":
            self.cam.buffer = np.array([])
            self.cam.buffer_1 = np.array([])
            self.cam.buffer_1_used = False
            self.tem.beam_blank(False)
            self.thread_beam.start()

            if tracking_dict["tracking_positions"] != []:
                tracking_positions = tracking_dict["tracking_positions"]
                experiment_type = tracking_dict["experiment_type"]
                tracking_step = tracking_dict["tracking_step"]
                tracking_positions = tracking_dict["tracking_positions"]
                ub_class = tracking_dict["ub_class"]
                _, x0_p, y0_p = tracking_positions[0]
                mode = tracking_dict["projection_mode"]
                if self.stem_value() != True:
                    answer = tk.messagebox.askyesno("stepwise data acquisition", "ready! press Yes when ready")
                    if answer == False:
                        print("data acquisition aborted by user")
                        self.abort_data_acquisition()
                        return
                    beam_pos = self.tem.get_beam_shift()

                    if mode == "DIFFRACTION":
                        mode = "IMAGING"
                        mag = tracking_dict["mag"]
                    else:
                        mag = tracking_dict["experimental_mag"]

                    ub_class.angle_x = self.cam_table[mode][str(mag)][2]
                    ub_class.scaling_factor_x = self.cam_table[mode][str(mag)][3]
                    ub_class.angle_y = self.cam_table[mode][str(mag)][4]
                    ub_class.scaling_factor_y = self.cam_table[mode][str(mag)][5]
                    beam_p = ub_class.beamshift_to_pix(beam_pos, ub_class.angle_x, ub_class.scaling_factor_x, 180 - ub_class.angle_y, ub_class.scaling_factor_y)

                elif self.stem_value() == True:
                    answer = tk.messagebox.askyesno("stepwise data acquisition", "ready! press Yes when ready")
                    if answer == False:
                        print("data acquisition aborted by user")
                        self.abort_data_acquisition()
                        return
                    beam_pos = self.tem.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
                    illumination_mode = tracking_dict["illumination_mode"]
                    mag = tracking_dict["experimental_mag"]
                    haadf_table = ub_class.haadf.load_calibration_table()
                    haadf_size = ub_class.haadf_cam_size
                    calibration = haadf_table[mode][illumination_mode][mag][1] * tracking_dict["stem_binning_value"] / 1000000000
                    beam_p_x = round(beam_pos[0] / calibration, 0)
                    beam_p_y = round(beam_pos[1] / calibration, 0)
                    beam_p = (beam_p_x, beam_p_y)

                orig_beam_p = np.copy(beam_p)
            else:
                orig_beam_p = np.copy(userbeam_position_start)

            #exp_angle = list(np.round(np.arange(start_angle, final_angle, tilt_step, dtype=np.float32), 4))
            exp_angle = list(np.round(np.arange(start_angle, final_angle+tilt_step, tilt_step, dtype=np.float32), 4))
            #exp_angle = list(np.round(np.linspace(start_angle, final_angle, (abs(final_angle-start_angle)/tilt_step)+1, dtype=np.float32), 4))
            if abs(exp_angle[-1]) > abs(final_angle):
                exp_angle.pop()
            print("exp_angle = ", exp_angle, "line 504")
            _, img_shape = self.cam.get_camera_characteristic()
            img_buffer = np.zeros((len(exp_angle), round(img_shape/binning), round(img_shape/binning)), dtype=np.uint16)
            self.tem.beam_blank(False)

            if tracking_dict["tracking_positions"] == []:
                answer = tk.messagebox.askyesno("stepwise data acquisition", "ready! press Yes when ready")
                if answer == False:
                    print("data acquisition aborted by user")
                    self.abort_data_acquisition()
                    return
            #prepare the camera to acquire the data
            t1 = time.monotonic_ns()

            ### set up in-situ tracking here ####
            if self.get_tracking_method() == "a priori + in situ":
                insitu_tilt_step = 5 # this is forced but we could change it from the GUI later
                insitu_angles = list(np.round(np.arange(start_angle, final_angle + insitu_tilt_step, insitu_tilt_step, dtype=np.float32), 4))
                if abs(insitu_angles[-1]) > abs(final_angle):
                    insitu_angles.pop()

                insitu_tracker = InSituTracker(tilt_step=insitu_tilt_step, tomo_tracker=self.tomo_tracker) # initialize the in-situ tracker
                insitu_tracker.track_result = self.track_result # pipe the trackign images and the result of patchworkCC
            else: insitu_angles = [0, 0, 0, 0]

            ### acq loop here, collect image and move the beam if necessary #############
            for i, angl in enumerate(exp_angle):

                if angl in insitu_angles and self.get_tracking_method() == "a priori + in situ": # this guard resolves every 5 deg only
                    beam_p = list(np.copy(orig_beam_p))
                    print("collecting image for insitu_tracker here")
                    # add procedure to collect the image
                    # probably this at the first iteration will not be optimal because we have a shift in the beam
                    # position when we change the beam size to take the image!! we need to try!
                    set_image_setting(self)
                    self.tem.client.client_send_action({"set_stem_beam": beam_pos})
                    self.tem.beam_blank(False)
                    time.sleep(0.33)
                    img_insitu = self.haadf.acquire_image_fast()

                    time.sleep(0.33)
                    set_diff_setting(self)
                    ### process the image
                    shift_vect = insitu_tracker.in_situ_run(image = img_insitu, frame_number= i)
                    beam_p = [beam_p[0] + shift_vect[0], beam_p[1] + shift_vect[1]]


                print("image: ", i+1, "/", len(exp_angle))
                if i != 0:
                    self.tem.set_alpha(angl, velocity=self.speed_tracking)
                self.tem.beam_blank(False)

                if tracking_dict["tracking_positions"] != []:
                    angles = [angle for angle, _, _ in tracking_dict["tracking_positions"]]
                    if angl in angles:
                        index = angles.index(angl)
                        angle, track_x, track_y = tracking_dict["tracking_positions"][index]

                    if self.stem_value() != True:
                        track_beam = (beam_p[0] + (track_x - x0_p), beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
                        track_beam = ub_class.pix_to_beamshift(track_beam, ub_class.angle_x, ub_class.scaling_factor_x, 180 - ub_class.angle_y, ub_class.scaling_factor_y)
                        self.tem.set_beam_shift(track_beam)
                    elif self.stem_value() == True:
                        track_beam = (beam_p[0] + (track_x - x0_p), beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
                        track_beam = ((track_beam[0] * calibration), (track_beam[1] * calibration))
                        self.tem.client.client_send_action({"set_stem_beam": track_beam})

                time.sleep(0.33)
                img = self.cam.acquire_image(exposure_time = exposure, binning = binning, processing = processing)
                self.tem.beam_blank(True)
                img_buffer[i, :, :] = img

            t2 = (time.monotonic_ns() - t1) / 1000000000
            self.cam.buffer = img_buffer

            frames = len(exp_angle)
            final_angle = exp_angle[-1]

            if self.get_tracking_method() == "a priori + in situ":
                insitu_tracker.plot_insitu_result()



        if self.thread_beam.is_alive():
            self.tem.set_alpha(final_angle+0.2)
            time.sleep(0.5)
            self.tem.set_alpha(final_angle)

        # get the data from the thread_beam to fit the goniometer behaviour and the timings from the camera
        if exp_type == "continuous":
            self.gonio_fit = [None, None]
            #self.gonio_fit = self.tem.result
            self.timings = self.cam.timings
            #self.fit_angles = list(self.gonio_fit[0](self.timings))
            # supp = []
            # for angle in self.fit_angles:
            #     supp.append(np.round(np.rad2deg(angle), 4))
            # self.fit_angles = supp
            # print("fitted_angles: %s\n"%str(len(self.fit_angles)), self.fit_angles)
        else:
            self.gonio_fit = [None, None]

    ## finish here data standard acquisition #######################################################################
    ## out from acquisition section, entering in saving data and create reports ####################################

    if self.stem_value() == True:
        self.exp_pixelsize = image_pixelsize(self)[1]
    else:
        self.exp_pixelsize = image_pixelsize(self)
    if self.tem.get_screen_position() != 'DOWN':
        if self.cam.is_cam_bottom_mounted():
            self.tem.move_screen(False)
    # if self.tem.get_projection_mode() == 'DIFFRACTION':
    #     self.tem.diffraction(checked_diff_value=False)

    # at the end of the exp put back the beamshift in the initial position
    if tracking_dict["tracking_positions"] != []:
        if exp_type == "stepwise":
            if self.stem_value() != True:
                self.tem.set_beam_shift(beam_pos)
            elif self.stem_value() == True:
                self.tem.client.client_send_action({"set_stem_beam": beam_pos})

    self.tem.beam_blank(True)
    ##### prepare the destination dir
    os.makedirs(saving_path_images, exist_ok=True)
    #if self.cam.name in ["BM-UltraScan", "CCD"]:
    if self.camera in ["us4000", "us2000"]:
        self.cam.save_cRED_data(savingpath=saving_path_images, processing = processing)
    else:
        self.cam.save_cRED_data(savingpath=saving_path_images)

    try:
        if self.tem.calibrated_speed != None:
            rotation_speed = self.tem.calibrated_speed["m"]
    except:
        if self.tem.calibrated_speed.empty == False:
            rotation_speed = self.tem.calibrated_speed["m"]

    # prepare the output log file
    self.resume_experiment = {"date": now,
                              "microscope": self.brand,
                              "voltage": self.tem.get_voltage(),
                              "wavelength": calculate_wavelength(self, self.tem.get_voltage()),
                              "optics_mode": optics_mode,
                              "spotsize": self.tem.get_spotsize(),
                              "exp_type": exp_type,
                              "KL": self.tem.get_KL(),
                              "C2": round(self.tem.get_intensity(),4),
                              "camera": self.camera,
                              "saturation_limit": 65535,
                              "exp_time": t2,
                              "exposure": exposure,
                              "aperpixel": self.exp_pixelsize,
                              "binning": binning,
                              "image_size": tuple(self.cam.buffer.shape[1:]),
                              "frames": frames,
                              "start_angle": start_angle,
                              "final_angle": final_angle,
                              "angular_range": abs(final_angle-start_angle),
                              "rotation_speed": rotation_speed,
                              "rotation_speed_fitted": self.gonio_fit[1],
                              "eff_rotation_speed": abs(final_angle-start_angle)/t2,
                              "tilt_step": tilt_step,
                              "eff_tilt_step": np.sign(tilt_step)*(round(abs(final_angle-start_angle)/(frames-1), 5)),
                              "stage_start_position": stage_start_position,
                              "stage_end_position": self.tem.get_stage(),
                              "tracked_positions": tracking_dict["tracking_positions"],
                              "userbeam_position_start": userbeam_position_start,
                              "userbeam_positions": None}

    self.resume_experiment_with_units =  {"date": now,
                                          "Microscope": str(self.cam_table["microscope"]),
                                          "Microscope_tag": str(self.brand),
                                          "Voltage": str(self.tem.get_voltage())+ " kV",
                                          "Wavelength": str(calculate_wavelength(self, self.tem.get_voltage()))+ " A",
                                          "Optics_mode": str(optics_mode),
                                          "spotsize": str(self.tem.get_spotsize()) +" a.u.",
                                          "Exp_type": str(exp_type),
                                          "KL": str(self.tem.get_KL())+ " mm",
                                          "C2": str(round(self.tem.get_intensity(), 4))+ " %",
                                          "Camera": str(self.cam_table["camera"]),
                                          "Camera_tag": str(self.camera),
                                          "Saturation_limit": 65535,
                                          "Experiment_time": str(t2)+ " s",
                                          "Exposure": str(exposure)+ " ms",
                                          "Aperpixel": str(self.exp_pixelsize)+ " A-1",
                                          "Binning": str(binning),
                                          "Image_size": str(tuple(self.cam.buffer.shape[1:]))+ " pixels",
                                          "Frames": str(frames),
                                          "Start_angle": str(start_angle)+ " deg",
                                          "Final_angle": str(final_angle)+ " deg",
                                          "Angular_range": str(abs(final_angle - start_angle))+ " deg",
                                          "Rotation_speed": str(round(rotation_speed, 5))+ " deg/s",
                                          "rotation_speed_fitted": str(self.gonio_fit[1]) + "deg/s",
                                          "Eff_rotation_speed": str(round(abs(final_angle - start_angle) / t2, 5))+ " deg/s",
                                          "Tilt_step": str(tilt_step)+ " deg/img",
                                          "Eff_Tilt_step": str(np.sign(tilt_step) * (
                                              round(abs(final_angle - start_angle) / (frames - 1), 5)))+ " deg/img",
                                          "Stage_start_position": str(stage_start_position_std),
                                          "Stage_end_position": str(self.tem.get_stage(standard = True)),
                                          "Tracked_positions (angle, x, y)": str(tracking_dict["tracking_positions"]),
                                          "Userbeam_position_start": str(userbeam_position_start)+ " a.u.",
                                          "Userbeam_positions": None}

    if self.get_tracking_method() != "no tracking":
        self.resume_experiment_with_units["tracking_initial_position"] = str(self.init_position_stage_tracking)

    self.tracking_dictionary = tracking_dict

    #write tracking images and file here
    val = write_pets_file(self, path = saving_path, pets_default_values = "pets_default_values.txt")
    write_report_experiment(self, path=saving_path, add_val = val)

    ### saveing tracking images
    if self.get_tracking_method() != "no tracking":
        save_tracking_images(self, buffer = self.tracking_images, saving_dir = saving_path)
def save_tracking_images(self, buffer, saving_dir):
    ii = 0
    current_dir = os.getcwd()
    saving_dir = saving_dir + "/tracking_images/"
    os.makedirs(saving_dir, exist_ok=True)
    os.chdir(saving_dir)
    for image in buffer:
        # format tiff uncompressed data
        image_name = str('tracking_img_%s.tif' % (format(ii, '.0f').rjust(3, '0')))
        # print("saving: ", self.image_name)
        imageio.imwrite(image_name, image)
        ii += 1
    os.chdir(current_dir)
def stop_experiment(self):
    self.stop_signal = True
    if self.cam.is_cam_bottom_mounted():
        self.tem.move_screen(False)
    self.tem.beam_blank(True)
    if self.tem.get_projection_mode() == 'DIFFRACTION':
        self.tem.diffraction(checked_diff_value=False)

def write_report_experiment(self, path, add_val = None):
    print("writing report experiment")
    # formatting the dictionary with measuring units
    report = self.resume_experiment_with_units.items()
    with open(path+os.sep+"exp_report_pyFast_ADT.txt", 'w') as file:
        file.write('3DED log file generated from FAST-ADT\n')
        for key, value in report:
            file.write((key+" = "+str(value)+"\n"))
        file.write("Noise_parameters"+" = "+add_val + "\n")

def write_pets_file(self, path, pets_default_values):
    """pets2 file writer to run it after the acquisition"""
    with open(pets_default_values, 'r') as file:
        file_contents = file.read()
    par = {}
    lines = file_contents.split('\n')

    for i, line in enumerate(lines):
        if 'noiseparameters' in line:
            par['noise'] = line.split('=')[-1].strip()
        elif 'background' in line:
            par['background'] = line.split('=')[-1].strip()
        elif 'dstarminps' in line:
            par['dstarminps'] = line.split('=')[-1].strip()
        elif 'dstarmaxps' in line:
            par['dstarmaxps'] = line.split('=')[-1].strip()
        elif 'dstarmin' in line:
            par['dstarmin'] = line.split('=')[-1].strip()
        elif 'dstarmax' in line:
            par['dstarmax'] = line.split('=')[-1].strip()
        elif 'omega' in line:
            par['omega'] = line.split('=')[-1].strip()
        elif 'reflectionsize' in line:
            par['reflectionsize'] = line.split('=')[-1].strip()
        elif 'I/sigma' in line:
            par['I/sigma'] = line.split('=')[-1].strip()
        elif 'phi' in line:
            par['phi'] = line.split('=')[-1].strip()


    pets_file_name = path+os.sep+self.exp_name()+".pts2"
    print("writing pets file: %s" % pets_file_name)
    with open(pets_file_name, "w") as file_1:
        file_1.write("avoidicerings yes\n")
        file_1.write("autotask\npeak search\ntilt axis\npeak analysis\nfind cell\nendautotask\n")
        file_1.write("noiseparameters"+"\t"+par["noise"] + "\n")
        file_1.write("detector default"+"\n")
        f_size = int(self.resume_experiment["image_size"][0])
        if f_size < 600: # due to time\medipix size not defined due to the cross
            file_1.write("bin"+"\t"+"1"+"\n")
        elif f_size == 1024:
            file_1.write("bin"+"\t"+"2"+"\n")
        elif f_size == 2048:
            file_1.write("bin" + "\t" + "4" + "\n")
        elif f_size == 4096:
            file_1.write("bin" + "\t" + "8" + "\n")
        file_1.write("saturationlimit"+"\t"+"64000"+"\n")
        file_1.write("background"+"\t" +par['background']+"\n")
        file_1.write("lambda"+"\t"+ str(self.resume_experiment["wavelength"]) + "\n")
        file_1.write("Aperpixel"+"\t"+ str(self.resume_experiment["aperpixel"]) + "\n")
        if self.resume_experiment["exp_type"] == "continuous":
            exp_type = "continuous"
            phi = str(round(float(abs(self.resume_experiment["eff_tilt_step"]))/2, 5))
        else:
            exp_type = "precession"
            phi = par["phi"]
        file_1.write("geometry"+"\t"+exp_type+"\n")
        file_1.write("dstarminps"+"\t"+ par["dstarminps"] + "\n")
        file_1.write("dstarmaxps"+"\t" + par["dstarmaxps"] + "\n")
        file_1.write("dstarmin"+"\t" + par["dstarmin"] + "\n")
        file_1.write("dstarmax"+"\t" + par["dstarmax"] + "\n")
        file_1.write("omega"+"\t" + par["omega"] + "\t"+"0" +"\t"+"1"+"\n")
        file_1.write("phi"+"\t" + phi + "\n")
        file_1.write("center"+"\t"+"auto"+ "\n")
        file_1.write("centermode centralbeam 1"+"\n")
        file_1.write("reflectionsize"+"\t"+ par["reflectionsize"] + "\n")
        file_1.write("I/sigma"+"\t"+ par["I/sigma"] + "\n")
        file_1.write("beamstop"+"\t" + "no"+ "\n")
        file_1.write("imagelist"+"\t"+"\n")

        frames_name = os.listdir(path+os.sep+"tiff")
        frames_name.sort()
        # angle_list = list((np.round(np.arange(self.resume_experiment["start_angle"],
        #                                       self.resume_experiment["final_angle"],
        #                                       self.resume_experiment["eff_tilt_step"], dtype=np.float32), 4)))
        if self.resume_experiment["exp_type"] == "stepwise":
            angle_list = list((np.round(np.arange(self.resume_experiment["start_angle"],
                                     self.resume_experiment["final_angle"] + self.resume_experiment["eff_tilt_step"],
                                     self.resume_experiment["eff_tilt_step"], dtype=np.float32), 4)))
            if abs(angle_list[-1]) > abs(self.resume_experiment["final_angle"]):
                angle_list.pop()
            print("angle_list = ", angle_list, "line 665")
        elif self.resume_experiment["exp_type"] == "continuous":
            #angle_list = self.fit_angles
            angle_list = list((np.round(np.arange(self.resume_experiment["start_angle"],
                                                  self.resume_experiment["final_angle"] + self.resume_experiment["eff_tilt_step"],
                                                  self.resume_experiment["eff_tilt_step"], dtype=np.float32), 4)))
        for frame,angle in zip(frames_name, angle_list):

            file_1.write("tiff"+os.sep+frame+"\t"+str(angle)+"\t"+"0.00"+"\n")
        file_1.write("endimagelist" + "\t")

        if self.pets_checkbox_var.get() == True:
            os.startfile(pets_file_name)

        return par['noise']

def write_eadt_file(self, path, eadt_default_values):
    """eadt file writer to run it after the acquisition, to finish"""
    with open(eadt_default_values, 'r') as file:
        file_contents = file.read()
    par = {}
    lines = file_contents.split('\n')

    for i, line in enumerate(lines):
        if 'noiseparameters' in line:
            par['noise'] = line.split('=')[-1].strip()
        elif 'background' in line:
            par['background'] = line.split('=')[-1].strip()
        elif 'dstarminps' in line:
            par['dstarminps'] = line.split('=')[-1].strip()
        elif 'dstarmaxps' in line:
            par['dstarmaxps'] = line.split('=')[-1].strip()
        elif 'dstarmin' in line:
            par['dstarmin'] = line.split('=')[-1].strip()
        elif 'dstarmax' in line:
            par['dstarmax'] = line.split('=')[-1].strip()
        elif 'omega' in line:
            par['omega'] = line.split('=')[-1].strip()
        elif 'reflectionsize' in line:
            par['reflectionsize'] = line.split('=')[-1].strip()
        elif 'I/sigma' in line:
            par['I/sigma'] = line.split('=')[-1].strip()
        elif 'phi' in line:
            par['phi'] = line.split('=')[-1].strip()


    eadt_file_name = path+os.sep+self.exp_name()+".eadt"
    print("writing pets file: %s" % eadt_file_name)
    with open(eadt_file_name, "w") as file_1:
        file_1.write("autotask\npeak search\ntilt axis\npeak analysis\nfind cell\nendautotask\n")
        file_1.write("noiseparameters"+"\t"+par["noise"] + "\n")
        file_1.write("detector default"+"\n")
        f_size = int(self.resume_experiment["image_size"][0])
        if f_size < 600: # due to time\medipix size not defined due to the cross
            file_1.write("bin"+"\t"+"1"+"\n")
        elif f_size == 1024:
            file_1.write("bin"+"\t"+"2"+"\n")
        elif f_size == 2048:
            file_1.write("bin" + "\t" + "4" + "\n")
        elif f_size == 4096:
            file_1.write("bin" + "\t" + "8" + "\n")
        file_1.write("saturationlimit"+"\t"+"64000"+"\n")
        file_1.write("background"+"\t" +par['background']+"\n")
        file_1.write("lambda"+"\t"+ str(self.resume_experiment["wavelength"]) + "\n")
        file_1.write("Aperpixel"+"\t"+ str(self.resume_experiment["aperpixel"]) + "\n")
        if self.resume_experiment["exp_type"] == "continuous":
            exp_type = "continuous"
            phi = str(round(float(abs(self.resume_experiment["eff_tilt_step"]))/2, 5))
        else:
            exp_type = "precession"
            phi = par["phi"]
        file_1.write("geometry"+"\t"+exp_type+"\n")
        file_1.write("dstarminps"+"\t"+ par["dstarminps"] + "\n")
        file_1.write("dstarmaxps"+"\t" + par["dstarmaxps"] + "\n")
        file_1.write("dstarmin"+"\t" + par["dstarmin"] + "\n")
        file_1.write("dstarmax"+"\t" + par["dstarmax"] + "\n")
        file_1.write("omega"+"\t" + par["omega"] + "\t"+"0" +"\t"+"1"+"\n")
        file_1.write("phi"+"\t" + phi + "\n")
        file_1.write("center"+"\t"+"auto"+ "\n")
        file_1.write("centermode centralbeam 1"+"\n")
        file_1.write("reflectionsize"+"\t"+ par["reflectionsize"] + "\n")
        file_1.write("I/sigma"+"\t"+ par["I/sigma"] + "\n")
        file_1.write("beamstop"+"\t" + "no"+ "\n")
        file_1.write("imagelist"+"\t"+"\n")

        frames_name = os.listdir(path+os.sep+"tiff")
        frames_name.sort()
        # angle_list = list((np.round(np.arange(self.resume_experiment["start_angle"],
        #                                       self.resume_experiment["final_angle"],
        #                                       self.resume_experiment["eff_tilt_step"], dtype=np.float32), 4)))
        if self.resume_experiment["exp_type"] == "stepwise":
            angle_list = list((np.round(np.arange(self.resume_experiment["start_angle"],
                                                  self.resume_experiment["final_angle"] + self.resume_experiment[
                                                      "eff_tilt_step"],
                                                  self.resume_experiment["eff_tilt_step"], dtype=np.float32), 4)))
            if abs(angle_list[-1]) > abs(self.resume_experiment["final_angle"]):
                angle_list.pop()
            print("angle_list = ", angle_list, "line 665")
        elif self.resume_experiment["exp_type"] == "continuous":
            angle_list = self.fit_angles
        for frame,angle in zip(frames_name, angle_list):

            file_1.write("tiff"+os.sep+frame+"\t"+str(angle)+"\t"+"0.00"+"\n")
        file_1.write("endimagelist" + "\t")

        if self.pets_checkbox_var.get() == True:
            os.startfile(eadt_file_name)

def raster_scanning_userbeamshift(self, list_target = None):
    """ i don't know where put it soo will stay here for now"""
    # fake scan using user beam shift
    freq = 0.05
    if self.stem_value() == True:
        """here for testing but is not necessary at all"""
        #proj = self.tem.get_projection_mode()
        proj = "DIFFRACTION"
        illumination_mode = self.tem.get_illumination_mode()
        #mag = str(self.tem.get_magnification())
        mag = "3000"
        kl = (self.tem.get_KL())

        if list_target == None:
            angle_x = self.haadf_table[proj][illumination_mode][mag][2]
            scaling_x = self.haadf_table[proj][illumination_mode][mag][3]
            angle_y = self.haadf_table[proj][illumination_mode][mag][4]
            scaling_y = self.haadf_table[proj][illumination_mode][mag][5]
            list_target, list_target_pix = self.ub.list_pix_to_beamshift(angle_x=angle_x, scaling_factor_x=scaling_x,
                                                                         angle_y=angle_y,
                                                                         scaling_factor_y=scaling_y, default=True)
            print(list_target)
            print(list_target_pix)
            one = list_target[0]
            two = list_target[1]
            three = list_target[2]
            four = list_target[3]
            five = list_target[4]
            six = list_target[5]
            seven = list_target[6]
            eight = list_target[8]
            nine = list_target[9]

        else:
            one = list_target[0]
            two = list_target[1]
            three = list_target[2]
            four = list_target[3]
            five = list_target[4]
            six = list_target[5]
            seven = list_target[6]
            eight = list_target[8]
            nine = list_target[9]

        for i in range(10):
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": two})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": three})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": four})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": five})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": one})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": six})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": seven})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": three})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": eight})
            time.sleep(freq)
            self.haadf.client.client_send_action({"set_stem_beam": nine})
            if i == 9:
                time.sleep(freq)
                self.haadf.client.client_send_action({"set_stem_beam": three})

        print("stem done")
    else:
        if list_target == None:
            proj = self.tem.get_projection_mode()
            illumination_mode = self.tem.get_illumination_mode()
            mag = str(self.tem.get_magnification())
            kl = (self.tem.get_KL())
            angle_x = self.cam_table[proj][mag][2]
            scaling_x = self.cam_table[proj][mag][3]
            angle_y = self.cam_table[proj][mag][4]
            scaling_y = self.cam_table[proj][mag][5]
            list_target, _ = self.ub.list_pix_to_beamshift(angle_x = angle_x, scaling_factor_x = scaling_x, angle_y = angle_y, scaling_factor_y = scaling_y, default = True)
            one = list_target[0]
            two = list_target[1]
            three = list_target[2]
            four = list_target[3]
            five = list_target[4]
            six = list_target[5]
            seven = list_target[6]
            eight = list_target[8]
            nine = list_target[9]

        else:
            one = list_target[0]
            two = list_target[1]
            three = list_target[2]
            four = list_target[3]
            five = list_target[4]
            six = list_target[5]
            seven = list_target[6]
            eight = list_target[8]
            nine = list_target[9]

        for i in range(10):
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(two))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(three))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(four))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(five))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(one))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(six))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(seven))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(three))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(eight))
            time.sleep(freq)
            self.tem.set_beam_shift(np.asarray(nine))
            if i == 9:
                time.sleep(freq)
                self.tem.set_beam_shift(np.asarray(three))

        print("tem done")

def read_tracking_file(self, text):
    # Read the tracking file, text is a relative path
    with open(text, 'r') as file:
        file_contents = file.read()

    values = {}
    lines = file_contents.split('\n')

    for i, line in enumerate(lines):
        if 'start_angle' in line:
            values['start_angle'] = line.split('=')[-1].strip()
        elif 'target_angle' in line:
            values['target_angle'] = line.split('=')[-1].strip()
        elif 'tilt_step' in line:
            values['tilt_step'] = line.split('=')[-1].strip()
        elif 'rotation_speed' in line:
            values['rotation_speed'] = line.split('=')[-1].strip()
        elif 'experiment_type' in line:
            values['experiment_type'] = line.split('=')[-1].strip()
        elif 'tracking_step' in line:
            values['tracking_step'] = line.split('=')[-1].strip()
        elif 'tracking_positions' in line:
            # Start reading tracking positions
            tracking_positions = []
            continue
        elif 'end_tracking_file' in line:
            # Stop reading tracking positions
            values['tracking_positions'] = tracking_positions
            break
        else:
            # Append the line as a tuple to tracking_positions
            try:
                position = tuple(map(float, line.split(',')))
            except Exception as err:
                print(err)
                position = None
            tracking_positions.append(position)

    self.tracking_positions = tracking_positions
    set_parameters_gui(self, values)
    print("tracking positions loaded correctly")

    return values

def write_tracking_file(self, text, start_angle, target_angle, tilt_step, rotation_speed,
                        experiment_type, tracking_step, tracking_positions = None):

    with open(text, 'w') as file:
        file.write('initial position stage for tracking = %s' %str(self.init_position_stage_tracking))
        file.write('start_angle (deg) = %f\n' % start_angle)
        file.write('target_angle (deg) = %f\n' % target_angle)
        file.write('tilt_step (deg/img) = %f\n' % tilt_step)
        file.write('rotation_speed (deg/s) = %f\n' % rotation_speed)
        file.write('experiment_type = %s\n' % experiment_type)
        file.write('tracking_step (deg) = %f\n' % tracking_step)
        file.write('tracking_positions (angle, x, y)\n')

        for position in tracking_positions:
            file.write('%f, %f, %f\n' % (position[0], position[1], position[2]))

        file.write('end_tracking_file\n')
    print("tracking file written in: %s" %text)
    showinfo("Tracking file", "Tracking file generated in: %s %s" %(text, os.getcwd()))

def retrieve_parameters_for_acquisition(self, mode = "acquisition"):
    # generate_tracking_file(self, text, start_angle, target_angle, rotation_speed, experiment_type, tracking_step,
    #                    tracking_positions):
    if self.seq_value() == True:
        experiment_type = "stepwise"
        rotation_speed = 0
        buffer_size = int(round(((abs(self.angle_value())+abs(self.final_angle_value())) / self.tilt_step_value()) + 1, 0))
    elif self.cont_value() == True:
        if mode == "tracking":
            rotation_speed = rotation_speed_value(self, mode)
            experiment_type = "continuous"
            if self.stem_value():
                buffer_size = int(round(((abs(self.angle_value()) + abs(self.final_angle_value())) / self.tilt_step_value()) + 1, 0))
                buffer_size = int(round((buffer_size * 1.15) / 2, 0))  # adding a 15% of buffer to split in 2 buffers
            else:
                buffer_size = int(round(((abs(self.angle_value()) + abs(self.final_angle_value())) / self.tilt_step_value()) + 1, 0))*(self.tem_imagetime_value()/self.exposure_value())
                buffer_size = int(round((buffer_size * 1.15) / 2, 0))  # adding a 15% of buffer to split in 2 buffers

        else:
            rotation_speed = rotation_speed_value(self)
            experiment_type = "continuous"
            buffer_size = int(round(((abs(self.angle_value()) + abs(self.final_angle_value())) / self.tilt_step_value()) + 1, 0))
            buffer_size = int(round((buffer_size*1.15)/2,0)) + 20 #adding a 25% of buffer to split in 2 buffers + 20 frames minimum.

    else:
        experiment_type = "nothing choosen"
        rotation_speed = 0

    if self.tem_value() == True:
        optics_mode = "tem"
    elif self.stem_value() == True:
        optics_mode = "stem"
    else:
        optics_mode = "nothing choosen"

    parameters_gui = {"experiment_type": experiment_type,             #stepwise/continuous
                      "optics_mode": optics_mode,                     #tem/stem
                      "stem_pixeltime": self.stem_pixeltime_value(),  #stem pixeltime
                      "tem_imagetime": self.tem_imagetime_value(),    #tem imagetime
                      "start_angle": self.angle_value(),              #start angle
                      "target_angle": self.final_angle_value(),       #final angle
                      "tilt_step": self.tilt_step_value(),            #tilt step
                      "exposure": self.exposure_value(),              #exposure time diffraction
                      "binning": self.binning_value(),                #binning
                      "processing": self.processing_value(),          #processing images
                      "tracking_step": self.tracking_step_value(),    #tracking tilt step?
                      "rotation_speed": rotation_speed,               #rotation speed cred
                      "buffer_size": buffer_size,                     #buffer size images
                      "kl": self.get_KL_value(),                      #KL value
                      "mag": self.get_mag_value(),                    #mag value
                      "tracking_method": self.get_tracking_method()}  #tracking method to use
    # ###  guard to add if checkbox are not pressed
    return parameters_gui

def set_parameters_gui(self, values):
    #set up the params of the UI to be compatible for the tracking previously done in case of crash work as a backup of the info
    if values['start_angle']:
        self.angle_entry.delete(0, tk.END)
        self.angle_entry.insert(0, values['start_angle'])
    if values['target_angle']:
        self.final_angle_entry.delete(0, tk.END)
        self.final_angle_entry.insert(0, values['target_angle'])
    if values['tilt_step']:
        self.tilt_step_entry.delete(0, tk.END)
        self.tilt_step_entry.insert(0, values['tilt_step'])
    if values['experiment_type'] == "stepwise":
        self.seq_var.set(True)
    else:
        self.cont_var.set(True)
    if values['tracking_step']:
        self.tracking_step_var.set(round(float(values["tracking_step"]),2))


def rotation_speed_value(self, mode = "acquisition"):
    """"FPS_function": ["678.62", "-0.943"], using this parameter of the self.cam_table,
    this calibration is an exponential fit of the FPS real vs exposure time (i.e 1/FPS theor).
    the value here come out in radians/s because compatible with fei"""
    # if mode == "tracking":
    #     if self.stem_value():
    #         self.FPS_track = self.cam_table["FPS_function"][0] * (self.stem_pixeltime_value() ** self.cam_table["FPS_function"][1])
    #     else:
    #         self.FPS_track = self.cam_table["FPS_function"][0] * (self.tem_imagetime_value() ** self.cam_table["FPS_function"][1])
    # elif mode == "acquisition":
    #     exp_time = self.exposure_value()
    exp_time = self.exposure_value()
    self.FPS = self.cam_table["FPS_function"][0] * (exp_time ** self.cam_table["FPS_function"][1])
    self.FPS = round(self.FPS, 2)
    rotation_speed = (self.FPS * float(self.tilt_step_entry.get()))
    return rotation_speed

def calculate_wavelength(self, voltage, from_list = False):
    """based on https://virtuelle-experimente.de/en/elektronenbeugung/wellenlaenge/de-broglie-relativistisch.php
    constant took from google search, results match the relativistic wavelength in the JEOL website:
    https://www.jeol.com/words/emterms/20121023.071258.php#gsc.tab=0
    input: voltage in kV, from_list = True if you want a table"""
    h = 6.62607015 * 10 ** -34  # planck constant J.s
    m = 9.1093837 * 10 ** -31  # electron rest mass kg
    e = 1.60217663 * 10 ** -19  # elementary charge C
    c = 2.99792458 * 10 ** 8  # speed of light m/s

    if from_list == True:
        voltage_list = [0.01, 0.1, 1, 10, 20, 30, 40, 60, 80, 100, 120, 160, 200, 300, 400, 500, 1000, 1250]
        for volt in voltage_list:
            voltage = volt * 1000  # voltage in V

            wl = (h * c) / ((((e * voltage) ** 2) + (2 * e * voltage * m * (c ** 2))) ** 0.5)
            print("Voltage (kV):", volt, "Lambda (A):", round(wl * 10 ** 10, 6))
    else:
        voltage = voltage * 1000  # voltage in V

        wl = (h * c) / ((((e * voltage) ** 2) + (2 * e * voltage * m * (c ** 2))) ** 0.5)

        return round(wl*10**10, 6)

def load_camera_table(self):
    if self.stem_var.get() != True:
        try:
            self.cam_table = self.cam.load_calibration_table()
        except Exception as err:
            print("error loading the camera table", err)
            self.cam_table = None
        return self.cam_table
    elif self.stem_var.get() == True:
        try:
            self.haadf_table = self.haadf.load_calibration_table()
            self.cam_table = self.cam.load_calibration_table()
        except Exception as err:
            print("error loading the haadf or cam table", err)
            self.haadf_table = None
        return self.cam_table, self.haadf_table

def image_pixelsize(self):
    """if you are in real space return pixelsize in nm, else in A-1 (Aperpixel)"""
    if self.stem_var.get() != True:
        if self.cam_table == None:
            self.cam_table = load_camera_table(self)
        if self.get_tracking_method() == "debug":
            kl = self.cam_table["DIFFRACTION"].keys()[0]
            pixelsize = self.cam_table["DIFFRACTION"][kl][1]
        else:
            if self.tem.get_projection_mode() == "IMAGING":
                mag = str(self.tem.get_magnification())
                pixelsize = self.cam_table["IMAGING"][mag][1]
            else:
                kl = str(self.tem.get_KL())
                pixelsize = self.cam_table["DIFFRACTION"][kl][1]
        return pixelsize*self.binning_value()

    elif self.stem_var.get() == True:
        illumination_mode = self.tem.get_illumination_mode()

        if self.haadf_table == None:
            self.cam_table, self.haadf_table = load_camera_table(self)

        if self.get_tracking_method() == "debug":
            kl = self.haadf_table["DIFFRACTION"].keys()[0]
            pixelsize = self.haadf_table["DIFFRACTION"][kl][1]
        else:
            if self.tem.get_projection_mode() == "IMAGING":
                mag = str(self.tem.get_magnification())
                pixelsize = self.haadf_table["IMAGING"][mag][1]
            else:
                kl = str(self.tem.get_KL())
                mag = str(self.tem.get_magnification())
                mag_pixelsize = self.haadf_table["DIFFRACTION"][illumination_mode][mag][1]
                kl_pixelsize = self.cam_table["DIFFRACTION"][kl][1]

                kl_value = kl_pixelsize * self.binning_value()

                if self.get_stem_image_size_value() == "FULL":
                    mag_value = mag_pixelsize * self.get_stem_binning_value()
                elif self.get_stem_image_size_value() == "HALF":
                    mag_value = mag_pixelsize / 2 * self.get_stem_binning_value()
                elif self.get_stem_image_size_value() == "QUARTER":
                    mag_value = mag_pixelsize / 4 * self.get_stem_binning_value()
                return mag_value, kl_value


        if self.get_stem_image_size_value() == "FULL":
            value = pixelsize*self.get_stem_binning_value()
        elif self.get_stem_image_size_value() == "HALF":
            value = pixelsize/2*self.get_stem_binning_value()
        elif self.get_stem_image_size_value() == "QUARTER":
            value = pixelsize/4*self.get_stem_binning_value()

        return value

def beam_shift_vs_image_calibration(self):
    #self.beam_coordinates_test = [user_beam(xy), image_pixel(xy)]
    self.beam_coordinates_test = []
    if self.camera == "timepix1":
        processing = "Unprocessed"
        self.binning_combobox.current(1) # set binning to 1
    else: processing = self.processing_value()

    if self.get_tracking_method() == "debug":
        path = r"L:\Marco\hardware_microscopes\TecnaiF30\sergi_track\Tracking\Tomography\Sequential\18\clean\track_18__008.tif"
        self.orig_image = cv2.imread(path)
        live = cv2.imread(path)
    else:
        img_time = self.tem_imagetime_value()
        binning = self.binning_value()

        if self.stem_value() == True:
            print("not necessary in stem mode")
            #self.orig_image = self.haadf.acquire_image(exposure_time=self.stem_pixeltime_value(), binning=self.get_stem_binning_value(), image_size=self.get_stem_image_size_value())
            #self.orig_image = cv2.normalize(self.orig_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            #self.haadf.client.client_send_action(action={"set_stem_beam": (0, 0)})
        else:
            self.orig_image = self.cam.acquire_image(exposure_time=img_time, binning=binning, processing=processing)
        live_func = partial(self.cam.acquire_image, exposure_time=img_time, binning=binning, processing=processing)

    #self.orig_image = cv2.normalize(self.orig_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #self.orig_image = cv2.merge([self.orig_image, self.orig_image, self.orig_image])

    # if self.stem_value() == True:
    #     self.beam_coordinates_stem = []
    #     self.orig_image_copy = np.copy(self.orig_image)
    #     cv2.namedWindow("original_image")

    cv2.namedWindow("live_image")
    cv2.setMouseCallback("live_image", draw_circle_beam_shift, param = (self))
    # if self.stem_value() == True:
    #     cv2.imshow("original_image", self.orig_image_copy)
    #     cv2.setMouseCallback("live_image", draw_circle_beam_shift_stem, param=(self))

    while True:
        live = cv2.normalize(live_func(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        live = cv2.merge([live, live, live])
        # put the help text
        anchor = live.shape[1]
        cv2.putText(live, "ensure on 0 you have your beam shift alignment. going from 1 to 4 \n"
                          "place the beam on top and right click on the beam center 1 time.\n"
                          "when done press enter", (int(anchor * 0.10), int(anchor * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.imshow("original_image", self.orig_image_copy)
        start_x = int(round(live.shape[0] / 2))
        x_2 = int(round(start_x / 2))
        offset = 10
        # cv2.line(self.orig_image, (start_x, 0), (start_x, int(start_x * 2)), (0, 255, 0), 1)
        # cv2.line(self.orig_image, (0, start_x), (int(start_x * 2), start_x), (0, 255, 0), 1)
        cv2.line(live, (start_x, 0), (start_x, int(start_x * 2)), (0, 0, 130), 1)
        cv2.line(live, (0, start_x), (int(start_x * 2), start_x), (0, 0, 130), 1)
        cv2.rectangle(live, (x_2, x_2), (start_x + x_2, start_x + x_2), (255, 0, 0), 1)
        cv2.putText(live, "0", (start_x, start_x), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(live, "1", (x_2-offset, x_2-offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(live, "2", (x_2+start_x+offset, x_2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(live, "3", (x_2+start_x+offset, x_2+start_x+offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(live, "4", (x_2, x_2+start_x+offset+offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("live_image", live)

        key = cv2.waitKey(1)

        if key == 13:  # if press enter
            print("pressed enter")
            break

    cv2.destroyAllWindows()

    beam_pos = []
    pixels = []

    for beam, pxl in self.beam_coordinates_test:
        beam_pos.append(beam)
        pixels.append(pxl)

    # fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    # ax[0].imshow(self.orig_image)
    # ax[0].plot(beam_pos)
    # ax[0].set_title("original_image, pixel")
    # ax[1].plot(pixels)
    # ax[1].set_title("original_image, beamshift space")
    # if self.stem_value() == True:
    #     return self.beam_coordinates_stem
    # else:
    return self.beam_coordinates_test
def load_initial_parameters_experiment(self):
    self.tem.test_tracking(tracking=self.tracking_positions, ub_class=self.ub)

def draw_circle_beam_shift(event, x, y, flags, param): #this function must have this argument to work
    self = param
    #if event == cv2.EVENT_MOUSEMOVE:
        #cv2.circle(self.orig_image_copy, (x, y), 2, (0, 0, 255), -1)
        #cv2.imshow("original_image", self.orig_image_copy)
    #if event == cv2.EVENT_RBUTTONDOWN:
        #self.orig_image_copy = self.orig_image.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        self.beam_coor = self.tem.get_beam_shift()
        self.beam_coordinates_test.append((self.beam_coor, (x, y)))
        print(self.beam_coordinates_test[-1])

# def draw_circle_beam_shift_stem(event, x, y, flags, param):  # this function must have this argument to work
#     self = param
#     if event == cv2.EVENT_LBUTTONDOWN:
#         self.beam_coor = self.haadf.client.client_send_action(action = {"get_stem_beam":0})
#         #self.beam_coordinates_stem.append((self.beam_coor, (x, y)))
#         beamx, beamy = self.beam_coor["get_stem_beam"]
#         self.beam_coordinates_stem.append(((beamx, beamy), (x, y)))
#         print(self.beam_coordinates_stem[-1])
#         #cv2.circle(self.orig_image_copy, (x, y), 2, (0, 0, 255), -1)
#         #cv2.imshow("original_image", self.orig_image_copy)
#     #if event == cv2.EVENT_RBUTTONDOWN:
#     #    self.orig_image_copy = self.orig_image.copy()

def to_json(self, o, level=0):
    """"from https://stackoverflow.com/questions/10097477/python-json-array-newlines?rq=3"""
    INDENT = 3
    SPACE = " "
    NEWLINE = "\n"
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(self, v, level + 1)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + ", ".join([to_json(self, e, level + 1) for e in o]) + "]"
    # Tuples are interpreted as lists
    elif isinstance(o, tuple):
        ret += "[" + ", ".join(to_json(self, e, level + 1) for e in o) + "]"
    elif isinstance(o, bool):
        if o == True:
            ret += '"True"'
        elif o == False:
            ret += '"False"'
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.integer):
        ret += "[" + ', '.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, np.ndarray) and np.issubdtype(o.dtype, np.inexact):
        ret += "[" + ', '.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    else:
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret

def write_cam_table(self):
    if self.stem_value() != True:
        print("before func", self.cam_table)
        print("out func to_json", to_json(self, self.cam_table))
        cwd = os.getcwd()
        path = cwd + os.sep +r"adaptor/camera/lookup_table/" + str(self.camera) +"_table.txt"

        temp_save = self.cam.load_calibration_table()
        temp_save["bottom_mounted"] = str(temp_save["bottom_mounted"])
        temp_save["cam_flip_h"] = str(temp_save["cam_flip_h"])
        temp_save["cam_flip_v"] = str(temp_save["cam_flip_v"])
        temp_save["cam_flip_diag"] = str(temp_save["cam_flip_diag"])
        if len(self.cam_table["IMAGING"]) < 1 and len(self.cam_table["DIFFRACTION"]) < 1:
            raise TypeError("possible lose of camera calibration file because the variable is not a proper dictionary, please make a backup and try again!")
        else:
            try:
                with open(path, 'w') as file:
                    file.write(to_json(self, self.cam_table))

                temp_read = json.load(open(path, "r"))

                if to_json(self, temp_read) == to_json(self, self.cam_table):
                    print("camera table updated in: %s" % path)

                else:
                    print("old", to_json(self, temp_read))
                    print("new", to_json(self, self.cam_table))
                    raise TypeError("possible lose of camera calibration file after writing the new file, please make a backup and try again!")

            except:
                print("new calibration not saved")
                with open(path, 'w') as file:
                    file.write(to_json(self, temp_save))
    else:
        print("before func", self.haadf_table)
        print("out func to_json", to_json(self, self.haadf_table))
        cwd = os.getcwd()
        # need to be adapted correctly to the name of the self.haadf, now hardforced to spirit
        #path = cwd + os.sep + r"adaptor/camera/lookup_table/" + str(self.camera) + "_table.txt"
        path = cwd + os.sep + r"adaptor/camera/lookup_table/" + "spirit_haadf" + "_table.txt"

        temp_save = self.haadf.load_calibration_table()
        temp_save["bottom_mounted"] = str(temp_save["bottom_mounted"])
        temp_save["cam_flip_h"] = str(temp_save["cam_flip_h"])
        temp_save["cam_flip_diag"] = str(temp_save["cam_flip_diag"])
        if len(self.haadf_table["IMAGING"]) < 1 and len(self.haadf_table["DIFFRACTION"]) < 1:
            raise TypeError(
                "possible lose of haadf calibration file because the variable is not a proper dictionary, please make a backup and try again!")
        else:
            try:
                with open(path, 'w') as file:
                    file.write(to_json(self, self.haadf_table))

                temp_read = json.load(open(path, "r"))

                if to_json(self, temp_read) == to_json(self, self.haadf_table):
                    print("haadf table updated in: %s" % path)

                else:
                    print("old", to_json(self, temp_read))
                    print("new", to_json(self, self.haadf_table))
                    raise TypeError(
                        "possible lose of haadf calibration file after writing the new file, please make a backup and try again!")

            except:
                print("new calibration not saved")
                with open(path, 'w') as file:
                    file.write(to_json(self, temp_save))

def list_pix_to_beamshift_stem(self, list_target_p=None):
    """the calibration in stem mode is handled differently wrt tem mode. here TIA is already providing the conversion
    between the beamshift function and the pixels in the haadf detector."""
    proj = self.tem.get_projection_mode()
    illumination_mode = self.tem.get_illumination_mode()
    mag = str(self.tem.get_magnification())
    kl = (self.tem.get_KL())

    _, cam_size = self.haadf.get_camera_characteristic()
    if self.get_stem_image_size_value() == "FULL":
        cam_size = int(cam_size / (self.get_stem_binning_value()))
    elif self.get_stem_image_size_value() == "HALF":
        cam_size = int(cam_size / (2 * self.get_stem_binning_value()))
    elif self.get_stem_image_size_value() == "QUARTER":
        cam_size = int(cam_size / (4 * self.get_stem_binning_value()))
    # this convert pix in nm, from tia
    calibration = self.haadf_table[proj][illumination_mode][mag][1]*self.get_stem_binning_value()

    print("haadf size = ", cam_size)
    print("haadf calibration (nm/pix) = ", calibration)


# to add the bot for DL value and temspy TAD to control cont rotation here.

#### stem support here ####
def stem_mode_imaging(self):
    # self.brand = "fei"
    try:
        if self.haadf == None:
            if self.brand == 'fei' or self.brand == 'fei_temspy':
                from adaptor.camera.adaptor_haadf import Cam_haadf
                self.haadf = Cam_haadf(self.cam_table, instance_gui = self)
                self.cam_table, self.haadf_table = load_camera_table(self)
                print("haadf initialized")
    except Exception as err:
        print("error initializing haadf", err)
def evaluate_timings(self):
    try:
        haadf = self.haadf.ref_timings.copy()
        ref = haadf["start_ref_time"]
        haadf["start_t1_time"] = (haadf["start_t1_time"] - ref)/10**9
        haadf["start_stage_thread_time"] = (haadf["start_stage_thread_time"] - ref)/10**9
        haadf["end_acq_cred_time"] = (haadf["end_acq_cred_time"] - ref)/10**9
        haadf["start_beam_thread_time"] = (haadf["start_beam_thread_time"] - ref)/10**9
        print(haadf)
    except: print("haadf timings not present")
    try:
        cam = self.cam.ref_timings.copy()
        ref = cam["start_ref_time"]
        cam["start_t1_time"] = (cam["start_t1_time"] - ref)/10**9
        cam["start_stage_thread_time"] = (cam["start_stage_thread_time"] - ref)/10**9
        cam["end_acq_cred_time"] = (cam["end_acq_cred_time"] - ref)/10**9
        cam["start_beam_thread_time"] = (cam["start_beam_thread_time"] - ref)/10**9
        print(cam)
    except:
        print("cam timings not present")
    try:
        tem = self.tem.result[5].copy()
        ref = tem["start_ref_time"]
        tem["start_i_time"] = (tem["start_i_time"] - ref)/10**9
        tem["end_angle_tracking"] = (tem["end_angle_tracking"] - ref)/10**9
        print(tem)
    except:
        print("tem timings not present")

def tracking_precision_run(self, tracking_dict):
    self.tracking_precision_running = True
    saving = tkinter.filedialog.askdirectory(title="Please select the folder where you want to save the data output")
    saving = saving + os.sep + "tracking_precision"
    os.makedirs(saving, exist_ok=True)
    tracking_dict["tracking_method"] = "tracking_precision"
    # save the previous parameters of the initial tracking
    # self.tomo_tracker.orig_template
    self.initial_tracking = {"tracking_images": self.tracking_images,
                             "tracking_angles": self.track_angles,
                             "tracking_positions": self.tracking_positions,
                             "tracking_result": self.track_result,
                             "tracking_plot": self.plot_result,
                             "tomo_tracker_class": self.tomo_tracker}
    #maybe here we can start the loop
    try:
        loops = int(input("select an integer number of runs to average for the evaluation:"))
    except:
        loops = int(input("must select an integer, suggested = 9 :"))

    tracking_initial_positions = [] # store the init_positions of the trackings
    for loop in range(int(loops)):
        print("starting cycle %s / %s" % (str(loop+1), str(loops)))
        acquire_tracking_images(self, tracking_path=None)  # this is to acquire the tracking the second time
        # self.process_tracking_images(self.tracking_images, self.tracking_angles, self.get_tracking_method(), visualization = False)

        if loop == 0:
            tracking_initial_positions.append(("initial_scan", self.init_position_stage_tracking))
            tracking_initial_positions.append((loop+1, self.track_prec_init_pos))
        else:
            tracking_initial_positions.append((loop+1, self.track_prec_init_pos))

        if self.cont_value():
            self.dt = 1 / self.FPS
        else:
            self.dt = 0.1
        # thsi is taking instead always the new one!
        self.tomo_tracker = Tomography_tracker(images=self.tracking_images, dt=self.dt,
                                               existing_roi=self.initial_tracking["tomo_tracker_class"].orig_template)
        automatic_res = self.tomo_tracker.main()
        self.plot_result = self.tomo_tracker.plot_tracking()
        patchworkCC = []
        CC = []
        KF = []
        pureKF = []
        manual = []
        for res in automatic_res:
            pureKF.append(res[0])
            patchworkCC.append(res[1])
            KF.append(res[2])
            CC.append(res[3])

        # self.support1.append((tuple(self.predicted_position), tuple(self.template_matching_result), tuple(self.filtered_position), self.CC_positions))
        self.track_result = {"CC": CC, "patchworkCC": patchworkCC, "pureKF": pureKF, "KF": KF, "manual": manual}

        # here is decided only the type of output from the previous dictionary
        positions = self.track_result["patchworkCC"]

        self.tracking_positions = []
        for (i, angle), pos in zip(enumerate(self.initial_tracking["tracking_angles"]), positions):
            self.tracking_positions.append((angle, pos[0], pos[1]))

        # to add here, in cred increase by linearization the number of tracking_positions
        #
        #

        self.tracking_images_done = True
        self.tracking_done = True

        self.second_tracking = {"tracking_images": self.tracking_images,
                                "tracking_angles": self.track_angles,
                                "tracking_positions": self.tracking_positions,
                                "tracking_result": self.track_result,
                                "tracking_plot": self.plot_result,
                                "tomo_tracker_class": self.tomo_tracker}

        if loop == 0:
            # saving_images
            orig_path = os.getcwd()
            os.chdir(saving)
            os.makedirs("tracking_images", exist_ok=True)
            os.chdir(orig_path)
        # this evaluate the tracking and save a plot for every iteration and a csv file with the results in tracking_images folder
        self.result_tracking_precision = evaluate_tracking_precision(self, saving, loop+1, self.initial_tracking, self.second_tracking)  # this compare the 2 tracking sequences

        if loop == 0:
            # saving_images
            orig_path = os.getcwd()

            os.chdir(saving + os.sep + "tracking_images")
            os.makedirs("initial_scan", exist_ok=True)
            os.chdir(saving + os.sep + "tracking_images" + os.sep + "initial_scan")

            for i, img in enumerate(self.initial_tracking["tracking_images"]):
                # format tiff uncompressed data
                image_name = str('%s.tif' % (format(i, '.0f').rjust(3, '0')))
                imageio.imwrite(image_name, img)

        os.chdir(saving + os.sep + "tracking_images")

        if loop == 0:
            imageio.imwrite("original_template.tif", self.initial_tracking["tomo_tracker_class"].orig_template)

        os.makedirs("%s_scan" %str(loop+1), exist_ok=True)
        os.chdir(saving + os.sep + "tracking_images" + os.sep + "%s_scan" %str(loop+1))

        for i, img in enumerate(self.second_tracking["tracking_images"]):
            # format tiff uncompressed data
            image_name = str('%s.tif' % (format(i, '.0f').rjust(3, '0')))
            imageio.imwrite(image_name, img)

        os.chdir(orig_path)
        self.gonio_fit = [None, None]

        # store the values for the next iteration so we get 1vs2, 2vs3 and so on ..
        self.initial_tracking = {"tracking_images": self.tracking_images,
                                 "tracking_angles": self.track_angles,
                                 "tracking_positions": self.tracking_positions,
                                 "tracking_result": self.track_result,
                                 "tracking_plot": self.plot_result,
                                 "tomo_tracker_class": self.tomo_tracker}
    os.chdir(saving)
    mag = self.tem.get_magnification()
    with open("initial_positions.txt", "w") as file:
        # Write the report contents
        for init_pos in tracking_initial_positions:
            file.write("\n"+str(init_pos))
        file.write("\ncamera_magnification\t")
        file.write(str(mag))
        file.write("\nmag pixelsize reported nm\t")
        file.write(str(image_pixelsize(self)))
        file.write("\nbacklash_correction used")
        file.write("\nalpha\t %s" %(str(self.get_a_backlash_correction_value())))
        file.write("\nx\t %s" %(str(self.get_x_backlash_correction_value())))
        file.write("\ny\t %s" %(str(self.get_y_backlash_correction_value())))
        file.write("\nz\t %s" %(str(self.get_z_backlash_correction_value())))
        file.write("\ninitial_position\t %s" %(str(self.get_init_position_value())))

    os.chdir(orig_path)

    overall_tracking_precision(self, saving, output_method= "patchworkCC")
    self.tracking_precision_running = False


def evaluate_tracking_precision(self, saving, iteration, initial_tracking, second_tracking, input_param = None, output = None):
    """sergi method to evaluate the tracking performances between 2 consecutive tracking acquisition.
    if iteration 1 this means we are comparing initial_scan vs 1_scan, iter 2 is 1_scan vs 2_scan and so on."""

    orig_path = os.getcwd()
    os.chdir(saving + os.sep + "tracking_images")


    result = {"KF":{"angle": [],
                    "displacement":[],
                    "initial_shift": [],
                    "tracking_x_first_scan": [],
                    "tracking_y_first_scan": [],
                    "tracking_x_second_scan": [],
                    "tracking_y_second_scan": []},
              "patchworkCC":{"angle": [],
                    "displacement":[],
                    "initial_shift": [],
                    "tracking_x_first_scan": [],
                    "tracking_y_first_scan": [],
                    "tracking_x_second_scan": [],
                    "tracking_y_second_scan": []},
              "pureKF":{"angle": [],
                    "displacement":[],
                    "initial_shift": [],
                    "tracking_x_first_scan": [],
                    "tracking_y_first_scan": [],
                    "tracking_x_second_scan": [],
                    "tracking_y_second_scan": []},
              "CC":{"angle": [],
                    "displacement":[],
                    "initial_shift": [],
                    "tracking_x_first_scan": [],
                    "tracking_y_first_scan": [],
                    "tracking_x_second_scan": [],
                    "tracking_y_second_scan": []}}


    methods = ["KF", "patchworkCC", "pureKF", "CC"]
    colors = ["b", "black", "g", "r"]

    if input_param == None:
        illumination_mode = self.tem.get_illumination_mode()
        mode = self.tem.get_projection_mode()
        mag = str(round(self.tem.get_magnification()))
        # calibration is in nm/pixels
        try:
            if self.stem_value() == True:
                calibration = self.haadf_table[mode][mag][1] * self.get_stem_binning_value()
            elif self.stem_value() != True:
                calibration = self.cam_table[mode][mag][1] * self.binning_value()
        except:
            calibration = 1
    else:
        # calibration is in nm/pixels
        calibration = input_param

    for method, color in zip(methods, colors):

        first_x0, first_y0 = initial_tracking["tracking_result"][method][0]
        second_x0, second_y0 = second_tracking["tracking_result"][method][0]
        i = 0
        for angle, (first_x, first_y), (second_x, second_y) in zip(initial_tracking["tracking_angles"] ,initial_tracking["tracking_result"][method], second_tracking["tracking_result"][method]):
            first_xi = first_x - first_x0
            first_yi = first_y - first_y0
            second_xi = second_x - second_x0
            second_yi = second_y - second_y0
            displacement = math.sqrt((second_xi-first_xi)**2 + (second_yi-first_yi)**2)*calibration
            result[method]["angle"].append(angle)
            result[method]["displacement"].append(displacement)
            result[method]["tracking_x_first_scan"].append(first_x)
            result[method]["tracking_y_first_scan"].append(first_y)
            result[method]["tracking_x_second_scan"].append(second_x)
            result[method]["tracking_y_second_scan"].append(second_y)

            if i == 0:
                first_xi = first_x - first_x0
                first_yi = first_y - first_y0
                second_xi = second_x - first_x0
                second_yi = second_y - first_y0
                initial_shift = math.sqrt((second_xi - first_xi) ** 2 + (second_yi - first_yi) ** 2)*calibration
                result[method]["initial_shift"].append(initial_shift)
            i += 1

        plt.plot(result[method]["angle"], result[method]["displacement"], label= method, marker="o", color = color)
        max_displ1 = max(result[method]["displacement"])
        angle_max1 = result[method]["angle"][result[method]["displacement"].index(max_displ1)]

        print("\n\nevaluation for : ", method, "method")
        print("initial_shift =", result[method]["initial_shift"][0])
        print("max_displacement =", max_displ1)
        print("angle_max =", angle_max1)
        print("optimum beam diameter =", "2(max_displacement - feature_size)", "\n2*(%s - feature_size)" % max_displ1)

        # writing the raw data in a csv file for patchworkCC
        if output == None:
            os.chdir(saving)
        else:
            os.chdir(output)
        if iteration == 1:
            temp_file_path = "raw_data_%s_iter_%s_initial_vs_1_scan.csv" % (str(method), str(iteration))
        else:
            temp_file_path = "raw_data_%s_iter_%s_%s_vs_%s_scan.csv" % (str(method), str(iteration), str(iteration-1), str(iteration))
        # temp_file_path = "raw_data_%s_iteration_%s.csv" %(str(method), str(iteration))
        # # Write custom text/header first
        # with open(temp_file_path, 'w') as f:
        #     f.write('result of the tracking procedure PatchworkCC calibration used: %s nm/pixels\n' % str(calibration))
        #     f.write("initial_shift = %s\n" % str(np.round(result[method]["initial_shift"][0], 3)))
        result1 = result[method].copy()
        del result1["initial_shift"]
        df1 = pd.DataFrame(result1)

        df2 = pd.DataFrame({"angle": ["average (nm)", "3 sigma", "max"],  # Keep the angle column
                            "displacement": [df1["displacement"][1:].mean(), df1["displacement"][1:].std()*3, df1["displacement"].max()]})
        df3 = pd.concat([df1, df2], ignore_index=True)
        df3.to_csv(temp_file_path, index = False)
        os.chdir(saving + os.sep + "tracking_images")

    plt.title("Tilt-scan reproducibility plot iteration: %s" %str(iteration))
    plt.xlabel('alpha tilt angle (deg)')
    plt.ylabel('maximum hysteresis shift (nm)')
    if output == None:
        plt.savefig("displacement_plot_%s.jpg" %str(iteration))
    else:
        os.chdir(output)
        plt.savefig("displacement_plot_%s.jpg" % str(iteration))

    plt.legend()
    plt.show()

    # save data
    df = pd.DataFrame(result)
    df.to_csv("displacement_data_%s.csv" %str(iteration))

    # Calculate paths
    if output == None:
        os.chdir(saving)
    else:
        os.chdir(output)
    path1 = (np.array(initial_tracking["tracking_result"]["patchworkCC"][:]) - np.array(initial_tracking["tracking_result"]["patchworkCC"][0])) * calibration
    path2 = (np.array(second_tracking["tracking_result"]["patchworkCC"][:]) - np.array(second_tracking["tracking_result"]["patchworkCC"][0])) * calibration

    # Differences in x and y between the paths
    diff_x = path1[:, 0] - path2[:, 0]
    diff_y = path1[:, 1] - path2[:, 1]

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # Two rows, one column

    # First subplot: Tracking paths
    axs[0].plot(path1[:, 0], path1[:, 1] * -1, label="path #1", marker="*", color="black", markersize=5, alpha=0.7)
    axs[0].plot(path2[:, 0], path2[:, 1] * -1, label="path #2", marker="+", color="red", markersize=5, alpha=0.7)

    # Annotate every 10th point
    for i in range(0, len(path1), 10):
        axs[0].text(path1[i, 0], path1[i, 1] * -1, "%s" % str(i), fontsize=8, ha='left', va='bottom')
        axs[0].text(path2[i, 0], path2[i, 1] * -1, "%s" % str(i), fontsize=8, ha='right', va='top')

    axs[0].set_title("Tracking Paths Comparison Iteration %s" % str(iteration))
    axs[0].set_xlabel('x position (nm)')
    axs[0].set_ylabel('y position (nm)')
    axs[0].legend()

    # Second subplot: Differences in x and y
    axs[1].plot(range(len(diff_x)), diff_x, label="Difference in x", color="blue", marker="o", markersize=3)
    axs[1].plot(range(len(diff_y)), diff_y, label="Difference in y", color="green", marker="s", markersize=3)

    # Annotate every 10th point
    for i in range(0, len(diff_x), 10):
        axs[1].text(i, diff_x[i], "%s" % str(i), fontsize=8, ha='left', va='bottom')
        axs[1].text(i, diff_y[i], "%s" % str(i), fontsize=8, ha='right', va='top')

    axs[1].set_title("Differences Between Path #1 and Path #2")
    axs[1].set_xlabel('Point Index')
    axs[1].set_ylabel('Difference (nm)')
    axs[1].legend()

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("tracking_paths_comparison_#%s.jpg" % str(iteration))
    plt.show()

    os.chdir(orig_path)
    return result

def overall_tracking_precision(self, saving = None, output = None, output_method = None):
    orig_path = os.getcwd()
    if output == None:
        csv_dir = saving + os.sep + "tracking_images"
    else:
        csv_dir = output
    i = 1
    #saving = tkinter.filedialog.askdirectory()
    names = os.listdir(csv_dir)
    substring = "displacement_data"
    names = [file for file in names if file.endswith('.csv') and substring in file and "averaged_" not in file]
    names.sort()
    target_list = []
    for name in names:
        target_number = [int(s) for s in name if s.isdigit()][0]
        target_list.append(target_number)

    kf = pd.DataFrame()
    patchworkcc = pd.DataFrame()
    cc = pd.DataFrame()
    purekf = pd.DataFrame()
    for name, number in zip(names, target_list):

        df1 = pd.read_csv(csv_dir + os.sep + name, converters={'KF': literal_eval, 'CC': literal_eval, 'patchworkCC': literal_eval, 'pureKF': literal_eval})

        methods = list(df1.columns[1:])
        colors = ["b", "black", "g", "r"]
        for method, color in zip(methods, colors):
            label_angle = "angle"
            label_displ = "displacement_%s" % str(number)
            if i == 1:
                # save all the angles in every dictionary in the iteration 1
                kf[label_angle] = df1[method].values[0]
                cc[label_angle] = df1[method].values[0]
                patchworkcc[label_angle] = df1[method].values[0]
                purekf[label_angle] = df1[method].values[0]
            if method == "KF":
                kf[label_displ] = df1[method].values[1]
                # shift = df1[method].values[2]
            elif method == "CC":
                cc[label_displ] = df1[method].values[1]
                # shift = df1[method].values[2]
            elif method == "patchworkCC":
                patchworkcc[label_displ] = df1[method].values[1]
                # shift = df1[method].values[2]
            elif method == "pureKF":
                purekf[label_displ] = df1[method].values[1]
                # shift = df1[method].values[2]

        i += 1

    print(kf)
    methods = ["KF", "CC", "patchworkCC", "pureKF"]
    colors = ["b", "r", "black", "g"]
    label_df = [kf, cc, patchworkcc, purekf]
    if output_method == None:
        for df, method, color in zip(label_df, methods, colors):
            # Drop the 'angle' column for the calculations
            df_values_only = df.drop(['angle'], axis=1)

            # Calculate mean and std deviation for each row
            row_mean = df_values_only.mean(axis=1)
            row_std = df_values_only.std(axis=1)

            # Add the calculated values back to the DataFrame if needed
            df['mean'] = row_mean
            df['std_dev'] = row_std

            plt.errorbar(df['angle'], df['mean'], yerr=df['std_dev'], fmt='o', label=method, color=color, errorevery=2)

            if output == None:
                df.to_csv("averaged_displacement_data_%s.csv" %str(method))
            else:
                os.chdir(output)
                df.to_csv("averaged_displacement_data_%s.csv" %str(method))

        plt.xlabel('Angles (deg)')
        plt.ylabel('Displacement (nm)')
        plt.title('tilt reproducibility plot')
        plt.legend()
        plt.savefig("averaged_displacement_plot.jpg")
        plt.show()

    else:
        index_ = methods.index(output_method)

        df = label_df[index_]
        method = methods[index_]
        # color = "black"
        color = "b"
        # Drop the 'angle' and the tracking points column for the calculations of the basic displacement
        df_values_only = df.drop(['angle'], axis=1)

        # Calculate mean and std deviation for each row
        row_mean = df_values_only.mean(axis=1)
        row_std = df_values_only.std(axis=1)

        # Add the calculated values back to the DataFrame if needed
        df['mean'] = row_mean
        df['std_dev'] = row_std

        plt.errorbar(df['angle'], df['mean'], yerr=df['std_dev'], fmt='o', label=method, color=color, ecolor="black", errorevery=2)
        if output == None:
            df.to_csv("averaged_displacement_data_%s.csv" % str(method))
        else:
            os.chdir(output)
            df.to_csv("averaged_displacement_data_%s.csv" % str(method))

        plt.xlabel('Angles (deg)')
        plt.ylabel('Displacement (nm)')
        plt.title('tilt reproducibility plot')
        plt.legend()
        plt.savefig("averaged_displacement_plot.jpg")
        plt.show()

        # # here instead prepare the df to calculate it in a series like 0vs1, 1vs2 and so on.. to remove the bias effect
        # # Drop the 'angle' and the tracking points column for the calculations of the basic displacement
        # df_tracking_points = kf.drop(['angle', 'displacement_1', 'displacement_2', 'displacement_3'], axis=1)
        # df_tracking_points = df_tracking_points.apply(lambda col: col - col.iloc[0])
        # # Iterate over consecutive column pairs
        # num_pairs = (len(df_tracking_points.columns) // 2) - 1  # Number of displacement calculations
        # calibration = float(input("please provide the calibration pxl to nm"))
        # for i in range(num_pairs):
        #     df_tracking_points[f"displacement_{i + 1}"] = df_tracking_points.apply(
        #         lambda row: np.sqrt(
        #             (row[f"track_x_{i + 1}"] - row[f"track_x_{i}"]) ** 2 +
        #             (row[f"track_y_{i + 1}"] - row[f"track_y_{i}"]) ** 2
        #         ) * calibration,
        #         axis=1
        #     )
        # #print(df_tracking_points)
        # df_values_only = df_tracking_points.drop(['track_x_0', 'track_y_0', 'track_x_1', 'track_y_1', 'track_x_2', 'track_y_2', 'track_x_3', 'track_y_3'], axis=1)
        # # Calculate mean and std deviation for each row
        # row_mean = df_values_only.mean(axis=1)
        # row_std = df_values_only.std(axis=1)
        #
        # # Add the calculated values back to the DataFrame if needed
        # df['mean'] = row_mean
        # df['std_dev'] = row_std
        #
        # plt.errorbar(df['angle'], df['mean'], yerr=df['std_dev'], fmt='o', label=method, color = color, ecolor="black", errorevery=2)
        # if output == None:
        #     df.to_csv("averaged_displacement_data_%s.csv" % str(method))
        # else:
        #     os.chdir(output)
        #     df.to_csv("averaged_displacement_data_%s.csv" % str(method))
        #
        # plt.xlabel('Angles (deg)')
        # plt.ylabel('Displacement (nm)')
        # plt.title('tilt reproducibility plot')
        # plt.legend()
        # plt.savefig("averaged_displacement_plot.jpg")
        # plt.show()

    os.chdir(orig_path)

def re_evaluate_tracking_precision(self):
    orig_path = os.getcwd()
    saving = tkinter.filedialog.askdirectory(title="Please select the folder where is present the folder 'tracking_images' from a previous tracking precision run")
    output_path = saving + os.sep + "re_evaluation"
    os.makedirs(output_path, exist_ok=True)
    tracking_images = saving + os.sep + "tracking_images"
    os.chdir(tracking_images)
    # List all files and directories in the folder
    all_items = os.listdir(tracking_images)
    # Filter only the directories
    folders_only = [item for item in all_items if os.path.isdir(os.path.join(tracking_images, item))]
    folders_only.remove("initial_scan")

    # load the initial scan
    # self.tracking_images = [os.path.join(tracking_images+os.sep+"initial_scan", item) for item in os.listdir(os.path.join(tracking_images, "initial_scan"))]
    first_scan_dir = tkinter.filedialog.askdirectory(title="Please select the folder where to start the re-evaluation, usually is the 'initial_scan' folder")
    target = os.path.split(first_scan_dir)
    if target[1] != ("initial_scan"):
        folders_only.remove(target[1])
        target_number = [int(s) for s in target[1] if s.isdigit()][0]
    else: target_number = 999

    self.tracking_images = [os.path.join(first_scan_dir, item) for item in os.listdir(os.path.join(first_scan_dir))]

    self.tracking_images = [img for img in self.tracking_images if img.endswith(".tif")]
    self.tracking_images.sort()
    # if self.cont_value():
    #     self.dt = 1 / self.FPS
    # else:
    #     self.dt = 0.1

    self.dt = 0.1
    self.tomo_tracker = Tomography_tracker(images=self.tracking_images, visualization=False, dt=self.dt)
    self.tomo_tracker.select_other_KF_model(KF_from_list="ukf_4D")
    automatic_res = self.tomo_tracker.main()
    self.plot_result = self.tomo_tracker.plot_tracking()
    patchworkCC = []
    CC = []
    KF = []
    pureKF = []
    manual = []
    for res in automatic_res:
        pureKF.append(res[0])
        patchworkCC.append(res[1])
        KF.append(res[2])
        CC.append(res[3])

    # self.support1.append((tuple(self.predicted_position), tuple(self.template_matching_result), tuple(self.filtered_position), self.CC_positions))
    self.track_result = {"CC": CC, "patchworkCC": patchworkCC, "pureKF": pureKF, "KF": KF, "manual": manual}

    # here is decided only the type of output from the previous dictionary
    positions = self.track_result["KF"]
    start_angle = float(input("initial angle:"))
    final_angle = float(input("final angle:"))
    tracking_step = float(input("tracking step size:"))
    input_param = float(input("calibration pxl to nm"))

    # start_angle = -60
    # final_angle = 60
    # tracking_step = 1
    # input_param = 2.2

    if final_angle < start_angle: tracking_step = -tracking_step
    # track_angles = list(np.round(np.arange(start_angle, final_angle, tracking_step, dtype=np.float32), 4))
    self.track_angles = list(np.round(np.arange(start_angle, final_angle + tracking_step, tracking_step, dtype=np.float32), 4))

    self.tracking_positions = []
    for (i, angle), pos in zip(enumerate(self.track_angles), positions):
        self.tracking_positions.append((angle, pos[0], pos[1]))

    self.initial_tracking = {"tracking_images": self.tracking_images,
                             "tracking_angles": self.track_angles,
                             "tracking_positions": self.tracking_positions,
                             "tracking_result": self.track_result,
                             "tracking_plot": self.plot_result,
                             "tomo_tracker_class": self.tomo_tracker}

    # to add here, in cred increase by linearization the number of tracking_positions
    #
    #

    self.tracking_images_done = True
    self.tracking_done = True
    if target[1] != ("initial_scan"):
        cycles_ = len(folders_only)+1
    else: cycles_ = len(folders_only)
    ####################### i iterations
    for i in range(cycles_):
        i += 1
        if i == target_number:
            continue
        print("cycle %s / %s" % (str(i), str(cycles_)))
        # load the i_scan
        self.tracking_images = [os.path.join(tracking_images + os.sep + "%s_scan" %str(i), item) for item in
                                os.listdir(os.path.join(tracking_images, "%s_scan" %str(i)))]
        self.tracking_images = [img for img in self.tracking_images if img.endswith(".tif")]
        self.tracking_images.sort()

        self.tomo_tracker = Tomography_tracker(images=self.tracking_images, visualization=False, dt=self.dt,
                                               existing_roi=self.initial_tracking["tomo_tracker_class"].orig_template)
        self.tomo_tracker.select_other_KF_model(KF_from_list="ukf_4D")
        automatic_res = self.tomo_tracker.main()
        self.plot_result = self.tomo_tracker.plot_tracking()
        patchworkCC = []
        CC = []
        KF = []
        pureKF = []
        manual = []
        for res in automatic_res:
            pureKF.append(res[0])
            patchworkCC.append(res[1])
            KF.append(res[2])
            CC.append(res[3])

        # self.support1.append((tuple(self.predicted_position), tuple(self.template_matching_result), tuple(self.filtered_position), self.CC_positions))
        self.track_result = {"CC": CC, "patchworkCC": patchworkCC, "pureKF": pureKF, "KF": KF, "manual": manual}

        # here is decided only the type of output from the previous dictionary
        positions = self.track_result["KF"]

        self.second_tracking = {"tracking_images": self.tracking_images,
                                "tracking_angles": self.track_angles,
                                "tracking_positions": self.tracking_positions,
                                "tracking_result": self.track_result,
                                "tracking_plot": self.plot_result,
                                "tomo_tracker_class": self.tomo_tracker}

        # input_param is the calibration of the images
        evaluate_tracking_precision(self, saving, i, self.initial_tracking, self.second_tracking, input_param, output_path)


        # store the values of the last as input for the next iteration 1vs2, 2vs3, and so on ..
        self.initial_tracking = {"tracking_images": self.tracking_images,
                                 "tracking_angles": self.track_angles,
                                 "tracking_positions": self.tracking_positions,
                                 "tracking_result": self.track_result,
                                 "tracking_plot": self.plot_result,
                                 "tomo_tracker_class": self.tomo_tracker}

    overall_tracking_precision(self, saving, output_path, output_method="patchworkCC")


def re_evaluate_crystal_tracking_path(self):
    orig_path = os.getcwd()
    saving = tkinter.filedialog.askdirectory()
    output_path = os.path.split(saving)[0] + os.sep + "re_evaluation_tracking_path"
    os.makedirs(output_path, exist_ok=True)
    tracking_images = saving
    os.chdir(tracking_images)
    # List all files and directories in the folder
    self.tracking_images = os.listdir(tracking_images)

    # load the tracking images
    self.tracking_images.sort()
    self.tracking_images = [img for img in self.tracking_images if img.endswith(".tif")]
    support = []
    for a in self.tracking_images:
        support.append(tracking_images + os.sep + a)
    self.tracking_images = support
    # if self.cont_value():
    #     self.dt = 1 / self.FPS
    # else:
    #     self.dt = 0.1

    self.dt = 0.1

    answer = tkinter.messagebox.askyesno("Question", "load an already existing roi?")
    if answer:
        existing_roi = tkinter.filedialog.askopenfilename()
        img_roi = cv2.imread(existing_roi)
        img_roi = cv2.normalize(img_roi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        self.tomo_tracker = Tomography_tracker(images=self.tracking_images, existing_roi = img_roi, visualization=False, dt=self.dt)
        # new model here 14112024
        self.tomo_tracker.select_other_KF_model(KF_from_list="ukf_4D")
        automatic_res = self.tomo_tracker.main()
        self.plot_result = self.tomo_tracker.plot_tracking_reevaluation()
    else:
        self.tomo_tracker = Tomography_tracker(images=self.tracking_images, visualization=False, dt=self.dt)
        # new model here 14112024
        self.tomo_tracker.select_other_KF_model(KF_from_list="ukf_4D")
        automatic_res = self.tomo_tracker.main()
        self.plot_result = self.tomo_tracker.plot_tracking_reevaluation()
    patchworkCC = []
    CC = []
    KF = []
    pureKF = []
    manual = []
    for res in automatic_res:
        pureKF.append(res[0])
        patchworkCC.append(res[1])
        KF.append(res[2])
        CC.append(res[3])

    # self.support1.append((tuple(self.predicted_position), tuple(self.template_matching_result), tuple(self.filtered_position), self.CC_positions))
    self.track_result = {"CC": CC, "patchworkCC": patchworkCC, "pureKF": pureKF, "KF": KF, "manual": manual}
    print(self.track_result)
    print("finished")
    # Save the plot as an image file
    plot_filename = 'crystal_tracking_plot.png'
    self.plot_result[0].savefig(output_path + os.sep + plot_filename)
    header = "Initial Angle: %s \nLast Angle: %s \ntracking every %s degree \nX, Y position in pixels" % ("xx", "xx", "xx")

    np.savetxt(output_path + os.sep + "tracking_datapointsCC.txt", self.track_result["CC"], header=header, comments="", delimiter=" , ", newline="\n", fmt="%.2f")
    np.savetxt(output_path + os.sep + "tracking_datapointspatchworkCC.txt", self.track_result["patchworkCC"], header=header, comments="", delimiter=" , ", newline="\n", fmt="%.2f")

    angles_ = np.linspace(0, len(support)-1, len(support))
    b_ = np.column_stack((angles_, np.array(self.track_result["patchworkCC"])))
    np.savetxt(output_path + os.sep + "test_fit.txt", b_, comments="", delimiter=" \t ", newline="\n", fmt="%.2f")
    os.chdir(orig_path)


##### here the methods for the semi manual mode, this was requested to make the manual acquisition smoother ######
def semi_manual_stepwise(self,exp_angle, exposure, binning, processing, img_buffer, root):
    # debug = True
    # if debug == True:
    #     self.manual_size = img_buffer[0].shape
    #     self.cam.acquire_image = np.zeros(size, dtype=np.uint16)
    self.manual_w = tk.Toplevel(root)
    self.manual_w.geometry("600x600")
    self.manual_w.title("Semi-Manual Stepwise Acquisition")
    self.canvas = tk.Canvas(self.manual_w, width=512, height=512)
    self.canvas.pack()

    # Create buttons
    button_frame = tk.Frame(self.manual_w)
    button_frame.pack(side=tk.BOTTOM, pady=10)

    button1 = tk.Button(button_frame, text="button1 (collect image)", command=lambda: display_image1(self, exposure, binning, processing))
    button1.pack(side=tk.LEFT, padx=10)

    button2 = tk.Button(button_frame, text="button2 (stop)", command=lambda: display_image2(self))
    button2.pack(side=tk.RIGHT, padx=10)

    button3 = tk.Button(button_frame, text="button3 (next angle)", command=lambda: display_image3(self, self.manual_i, exp_angle))
    button3.pack(side=tk.BOTTOM, padx=10)

    self.manual_i = 0
    self.manual_img_buffer = img_buffer
    self.manual_w.mainloop()

def display_image1(self, exposure, binning, processing):
    image = self.cam.acquire_image(exposure_time=exposure, binning=binning, processing=processing)
    # image = np.zeros(self.manual_size, dtype=np.uint16)
    self.manual_img = np.copy(image)
    # Convert to ImageTk format
    image = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image)

    # Display image on canvas
    self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
    self.canvas.image = image_tk  # Keep a reference to prevent garbage collection
    return image

def display_image2(self):
    print("data acquisition aborted by user")
    self.abort_data_acquisition()
    self.manual_w.quit()

    # we need to save the data here still ########################


def display_image3(self, i, exp_angle):
    if i >= len(exp_angle):
        print("image: ", i + 1, "/", len(exp_angle))
        print("last data point collected, please press stop button to end the collection and save the data")
        self.manual_img_buffer[i, :, :] = self.manual_img
        return
    angl = exp_angle[i]
    print("image: ", i + 1, "/", len(exp_angle))
    if i != 0:
        self.manual_img_buffer[i, :, :] = self.manual_img
        self.tem.set_alpha(angl, velocity=0.7)
    self.tem.beam_blank(False)
    time.sleep(0.33)
    self.manual_i += 1

## end of the semi manual acquisition ################

# def insitu_tracker(self, param = None):
#     """ first implementation only in stepwise + stem mode"""
#     print("i'm in")
#     # tem = self.tem
#     # cam = self.cam
#     exp_type = param["experiment_type"]
#     optics_mode = param["optics_mode"]
#     stem_dwell_time = param["stem_pixeltime"]
#     tem_image_time = param["tem_imagetime"]
#     start_angle = param["start_angle"]
#     final_angle = param["target_angle"]
#     tilt_step = param["tilt_step"]
#     exposure = param["exposure"]
#     binning = param["binning"]
#     processing = param["processing"]
#     tracking_step = param["tracking_step"]
#     rotation_speed = param["rotation_speed"]
#     buffer_size = param["buffer_size"]
#
#     if exp_type == "continuous":
#         return
#     else: pass
#     if optics_mode == "tem":
#         return
#     else: pass
#
#     # here the templates\ROI images of the tracking
#     # self.tomo_tracker.list_templates
#
#     # here the raw images used for the tracking
#     # self.tomo_tracker.series_support
#
#     # acquire image using the haadf
#     # perform CC using the template of the a priori, now you have a new point x,y (a new beam_p to use) where put the beam
#     # just overwrite the beam_p with the new one, or i need to overwrite x0_p?
#
#     beam_p = (beam_p_x, beam_p_y)
#     track_x = a-priori crystal tracking position
#     x0_p = a-priori crystal tracking position[0]
#
#     if angl in angles:
#         index = angles.index(angl)
#         angle, track_x, track_y = tracking_dict["tracking_positions"][index]
#
#     if self.stem_value() != True: # TEM mode
#         track_beam = (beam_p[0] + (track_x - x0_p), beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
#         track_beam = ub_class.pix_to_beamshift(track_beam, ub_class.angle_x, ub_class.scaling_factor_x, 180 - ub_class.angle_y, ub_class.scaling_factor_y)
#         self.tem.set_beam_shift(track_beam)
#
#     elif self.stem_value() == True: # STEM mode
#         track_beam = (beam_p[0] + (track_x - x0_p), beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
#         track_beam = ((track_beam[0] * calibration), (track_beam[1] * calibration))
#         self.tem.client.client_send_action({"set_stem_beam": track_beam})
#
#     return
#
#
#
def calculate_storage_capacity(data_type=np.uint16, available_memory_gb=16):
    """
    Calculate the maximum number of images that can be stored in memory for predefined resolutions and data type.

    Parameters:
    - data_type (data-type): Data type of the images, e.g., np.uint16, np.uint8.
    - available_memory_gb (float): Amount of memory available for storage in GB.

    Returns:
    - results (dict): Dictionary with pixels^2 as keys and a list [max_images storable, memory_per_image_mb] as values.
    """
    import numpy as np
    # Predefined resolutions
    resolutions = [(512, 512), (514, 514), (516, 516), (1024, 1024), (2048, 2048), (4096, 4096)]

    result = {}
    for resolution in resolutions:
        # Calculate memory required per image
        image_size = np.prod(resolution) * np.dtype(data_type).itemsize
        memory_per_image_mb = image_size * 1e-6  # Convert bytes to MB
        memory_per_image_gb = image_size * 1e-9  # Convert bytes to GB

        # Calculate maximum number of images that can fit in available memory
        max_images = int(available_memory_gb / memory_per_image_gb)

        # Store results
        result[str(resolution[0])] = [max_images, round(memory_per_image_mb, 3)]
        print(f"Resolution {resolution}: Can store {max_images} images, each requiring {memory_per_image_mb} MB.")
    return result

def get_available_memory_gb(self):
    import psutil
    # Get memory information
    mem_info = psutil.virtual_memory()
    # Total and available memory in GB
    total_memory_gb = mem_info.total / (1024 ** 3)
    available_memory_gb = mem_info.available / (1024 ** 3)
    print(f"Total memory: {total_memory_gb:.2f} GB")
    print(f"Available memory: {available_memory_gb:.2f} GB")
    return total_memory_gb, available_memory_gb

def automatic_eucentric_height(self):
    """ automatic fine eucentric height from D.N. Mastronarde / Journal of Structural Biology 152 (2005) 36–51
    collect eight images -24 to 24 deg. image shift in y is L.S. fitted to:
    y=(y0 + ys)*cos(alpha)-z0*sin(alpha)-y0
    to determine both z0, the Z-height, and y0, the offset between tilt and optical axes, where ys is the image
    shift of the specimen at zero tilt. It will work only for modest Z-height disparities (up to 10 um) and may
    restart after adjusting Z-height if image shifts become too large. """

    from scipy.optimize import curve_fit

    print("starting automatic eucentric height routine")
    # self.tem.beam_blank(False)
    if self.camera != "power_user":
        # retrieve the parameters to acquire tracking images from -24 to +24 deg and do it
        param = retrieve_parameters_for_acquisition(self, mode="tracking")

        param["start_angle"] = -24
        param["target_angle"] = 24
        param["buffer_size"] = 8
        param["tracking_step"] = ((abs(param["start_angle"]) + abs(param["target_angle"])) / (param["buffer_size"] - 1 ))
        param["tilt_step"] = param["tracking_step"] # to set FPS_devider = 1
        param["tracking_method"] = "patchworkCC"

        #acquire the images
        acquire_tracking_images(self, custom_param=param) # routine to acquire tracking images
        #process the images
        ## missing tracking_angles
        tracking_positions, track_result = process_tracking_images(self, self.tracking_images, self.track_angles, param["tracking_method"])
        # reset the routine to normal state
        reset_tracking_images(self)
        tracking_positions = np.array(tracking_positions)

    else:
        # provide the tracking data in power user mode
        path = tkinter.filedialog.askopenfilename(title="Please select the text file where the tracking data are stored")
        tracking_positions = np.loadtxt(path, delimiter="\t")

    # Extract columns andcompute the L.S. fit
    angle = tracking_positions[:, 0]  # First column
    xdata = tracking_positions[:, 1]  # Second column
    ydata = tracking_positions[:, 2]  # third column

    def eucentric_model_rigid_body(alpha, y0, ys, z0):
        return (y0 + ys) * np.cos(np.deg2rad(alpha)) - z0 * np.sin(np.deg2rad(alpha)) - y0  # mastronarde eq.

    y0 = 1  # offset from optical axis
    ys = 1  # shift 0 deg of the image
    z0 = 1  # offset optimum eucentric height

    popt, pcov, infodict, mesg, ier = curve_fit(eucentric_model_rigid_body, angle, ydata, p0=[y0, ys, z0], full_output=True)

    pcov_score = np.linalg.cond(pcov)
    perr = np.sqrt(np.diag(pcov))

    plt.plot(angle, eucentric_model_rigid_body(angle, *popt), 'r-', label='L.S. fit: y0=%5.3f, ys=%5.3f, z0=%5.3f' % tuple(popt))
    plt.plot(angle, ydata, 'b-', label='original data')
    plt.legend()
    plt.show()

    print("result of the fit pixels as (y0, ys, z0): ", popt)
    print("parametrization:", pcov_score)
    print("3 std dev error for the parameters:", perr*3)

    if self.camera != "power_user":
        # move Z-height to the found value
        calib = image_pixelsize(self)
        if self.stem_var.get() == True:
            calib = calib[0]

        coord_stage = self.tem.get_stage() # um and deg values
        self.tem.set_stage_position(z = coord_stage["z"]+(popt[2]*calib))

    else:
        pass

def backlash_data_acquisition(self):
    """script to perform the experiment of the 10/01/2025 for backlash characterization of the TEM goniometer"""
    # 1) note the initial coordinate -124.69 um in this case
    init_position = self.tem.get_stage()

    init_x = float(init_position["x"])
    init_y = float(init_position["y"])
    init_z = float(init_position["z"])
    init_a = float(init_position["a"])

    exposure = self.exposure_value()
    binning = self.binning_value()
    processing = self.processing_value()

    mag = self.tem.get_magnification()
    axis_choice = input("pick an axis to probe backlash: x, y, z, a")
    speed = 1
    sleeper = 1

    if axis_choice != "a":
        # datapoints = np.round(np.linspace(0.5, 10, 20), 1)
        datapoints = np.round(np.linspace(0.5, 5, 10), 1)
    else:
        datapoints1 = np.linspace(0.1, 5, 14)
        datapoints2 = np.linspace(8, 65, 20)
        datapoints = np.concatenate((datapoints1, datapoints2))
    print("starting the procedure...")

    if axis_choice == "x":
        # positive increment
        original_path = os.getcwd()
        initial_path = self.get_dir_value()
        os.makedirs(initial_path, exist_ok=True)
        os.makedirs(initial_path+os.sep+"moving x", exist_ok=True)
        os.chdir(initial_path+os.sep+"moving x")
        os.makedirs("positive increment", exist_ok=True)
        path_positive = initial_path + os.sep + "moving x" + os.sep + "positive increment"
        os.chdir(path_positive)


        for i, datapoint_label in enumerate(datapoints):
            print("starting datapoint + %s, %s / %s" %(str(datapoint_label), str(i+1), str(len(datapoints))))
            # 2) move up and down 4 um (-120.69 and after -128.69) and return to the initial position (identical for both + or – series!)
            self.tem.set_stage_position(x=init_x + 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(x=init_x - 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(x=init_x, speed=speed)
            time.sleep(sleeper)
            # 3) take an image (reference) (here more or less i'm always in the same spot)
            reference_datapoint = self.cam.acquire_image(exposure_time = exposure, binning = binning, processing = processing)
            reference_datapoint = cv2.normalize(reference_datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 4) move of the quantity wanted (i.e. 0.1 or 0.x) (here you decide + or - series)
            self.tem.set_stage_position(x=init_x + datapoint_label, speed=speed)
            time.sleep(sleeper)
            # 5) return to -124.69 um
            self.tem.set_stage_position(x=init_x, speed=speed)
            time.sleep(sleeper)
            # 6) take an image after
            datapoint = self.cam.acquire_image(exposure_time = exposure, binning = binning, processing = processing)
            datapoint = cv2.normalize(datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 7) compare images in step 3 and 6, or save the data and iterate

            # format tiff uncompressed data
            original_name = "original_position_"+str(datapoint_label)+".tif"
            datapoint_name = str(datapoint_label)+".tif"
            imageio.imwrite(datapoint_name, datapoint)
            imageio.imwrite(original_name, reference_datapoint)
            time.sleep(sleeper)
        # evaluate the shifts in the positive run
        process_images_in_folder(self, path_positive, path_positive+os.sep+"results")

        # repeat for the negative increment
        os.chdir(initial_path + os.sep + "moving x")
        os.makedirs("negative increment", exist_ok=True)
        path_negative = initial_path + os.sep + "moving x" + os.sep + "negative increment"
        os.chdir(path_negative)

        for i, datapoint_label in enumerate(datapoints):
            print("starting datapoint - %s, %s / %s" % (str(datapoint_label), str(i+1), str(len(datapoints))))
            # 2) move up and down 4 um (-120.69 and after -128.69) and return to the initial position (identical for both + or – series!)
            self.tem.set_stage_position(x=init_x + 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(x=init_x - 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(x=init_x, speed=speed)
            time.sleep(sleeper)
            # 3) take an image (reference) (here more or less i'm always in the same spot)
            reference_datapoint = self.cam.acquire_image(exposure_time = exposure, binning = binning, processing = processing)
            reference_datapoint = cv2.normalize(reference_datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 4) move of the quantity wanted (i.e. 0.1 or 0.x) (here you decide + or - series)
            self.tem.set_stage_position(x=init_x - datapoint_label, speed=speed)
            time.sleep(sleeper)
            # 5) return to -124.69 um
            self.tem.set_stage_position(x=init_x, speed=speed)
            time.sleep(sleeper)
            # 6) take an image after
            datapoint = self.cam.acquire_image(exposure_time = exposure, binning = binning, processing = processing)
            datapoint = cv2.normalize(datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 7) compare images in step 3 and 6, or save the data and iterate

            # format tiff uncompressed data
            original_name = "original_position_" + str(datapoint_label) + ".tif"
            datapoint_name = str(datapoint_label) + ".tif"
            imageio.imwrite(datapoint_name, datapoint)
            imageio.imwrite(original_name, reference_datapoint)
            time.sleep(sleeper)
        # evaluate the shifts in the positive run
        process_images_in_folder(self, path_negative, path_negative+os.sep+"results")

        print("experiment finished, writing report")
        os.chdir(initial_path + os.sep + "moving x")
        with open("details.txt", "w") as report_file:
            # Write the report contents
            report_file.write("Backlash Testing Report\nmoving axis %s, in positive and negative direction\n" %str(axis_choice))
            report_file.write("experiment date and time: %s\n" %str(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
            report_file.write("initial position: %s\n" %(str(init_position)))
            report_file.write("magnification: %s\n" %str(mag))
            report_file.write("experimental datapoints: \n%s\n" %str(datapoints))
            report_file.write("experimental data path: %s\n" %str(initial_path + os.sep + "moving x"+os.sep+"positive increment/negative increment"))
            report_file.write("procedure:\n")
            report_file.write("1) note the initial position of the object you want to use as probe to move\n")
            report_file.write("2) move up and down 4 um and return to the initial position\n")
            report_file.write("3) take an image (reference)\n")
            report_file.write("4) move of the incremental quantity (positive if you add it or negative if you subtract it to the initial position)\n")
            report_file.write("5) return to the initial position\n")
            report_file.write("6) take an image after\n")
            report_file.write("the report is generated by PyFast-ADT v0.1.0\n")

        os.chdir(original_path)

    elif axis_choice == "y":
        # positive increment
        original_path = os.getcwd()
        initial_path = self.get_dir_value()
        os.makedirs(initial_path, exist_ok=True)
        os.makedirs(initial_path + os.sep + "moving y", exist_ok=True)
        os.chdir(initial_path + os.sep + "moving y")
        os.makedirs("positive increment", exist_ok=True)
        path_positive = initial_path + os.sep + "moving y" + os.sep + "positive increment"
        os.chdir(path_positive)

        for i, datapoint_label in enumerate(datapoints):
            print("starting datapoint + %s, %s / %s" % (str(datapoint_label), str(i + 1), str(len(datapoints))))
            # 2) move up and down 4 um (-120.69 and after -128.69) and return to the initial position (identical for both + or – series!)
            self.tem.set_stage_position(y=init_y + 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(y=init_y - 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(y=init_y, speed=speed)
            time.sleep(sleeper)
            # 3) take an image (reference) (here more or less i'm always in the same spot)
            reference_datapoint = self.cam.acquire_image(exposure_time=exposure, binning=binning, processing=processing)
            reference_datapoint = cv2.normalize(reference_datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 4) move of the quantity wanted (i.e. 0.1 or 0.x) (here you decide + or - series)
            self.tem.set_stage_position(y=init_y + datapoint_label, speed=speed)
            time.sleep(sleeper)
            # 5) return to -124.69 um
            self.tem.set_stage_position(y=init_y, speed=speed)
            time.sleep(sleeper)
            # 6) take an image after
            datapoint = self.cam.acquire_image(exposure_time=exposure, binning=binning, processing=processing)
            datapoint = cv2.normalize(datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 7) compare images in step 3 and 6, or save the data and iterate

            # format tiff uncompressed data
            original_name = "original_position_" + str(datapoint_label) + ".tif"
            datapoint_name = str(datapoint_label) + ".tif"
            imageio.imwrite(datapoint_name, datapoint)
            imageio.imwrite(original_name, reference_datapoint)
            time.sleep(sleeper)
        # evaluate the shifts in the positive run
        process_images_in_folder(self, path_positive, path_positive + os.sep + "results")

        # repeat for the negative increment
        os.chdir(initial_path + os.sep + "moving y")
        os.makedirs("negative increment", exist_ok=True)
        path_negative = initial_path + os.sep + "moving y" + os.sep + "negative increment"
        os.chdir(path_negative)

        for i, datapoint_label in enumerate(datapoints):
            print("starting datapoint - %s, %s / %s" % (str(datapoint_label), str(i + 1), str(len(datapoints))))
            # 2) move up and down 4 um (-120.69 and after -128.69) and return to the initial position (identical for both + or – series!)
            self.tem.set_stage_position(y=init_y + 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(y=init_y - 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(y=init_y, speed=speed)
            time.sleep(sleeper)
            # 3) take an image (reference) (here more or less i'm always in the same spot)
            reference_datapoint = self.cam.acquire_image(exposure_time=exposure, binning=binning, processing=processing)
            reference_datapoint = cv2.normalize(reference_datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 4) move of the quantity wanted (i.e. 0.1 or 0.x) (here you decide + or - series)
            self.tem.set_stage_position(y=init_y - datapoint_label, speed=speed)
            time.sleep(sleeper)
            # 5) return to -124.69 um
            self.tem.set_stage_position(y=init_y, speed=speed)
            time.sleep(sleeper)
            # 6) take an image after
            datapoint = self.cam.acquire_image(exposure_time=exposure, binning=binning, processing=processing)
            datapoint = cv2.normalize(datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # 7) compare images in step 3 and 6, or save the data and iterate

            # format tiff uncompressed data
            original_name = "original_position_" + str(datapoint_label) + ".tif"
            datapoint_name = str(datapoint_label) + ".tif"
            imageio.imwrite(datapoint_name, datapoint)
            imageio.imwrite(original_name, reference_datapoint)
            time.sleep(sleeper)
        # evaluate the shifts in the positive run
        process_images_in_folder(self, path_negative, path_negative + os.sep + "results")

        print("experiment finished, writing report")
        os.chdir(initial_path + os.sep + "moving y")
        with open("details.txt", "w") as report_file:
            # Write the report contents
            report_file.write(
                "Backlash Testing Report\nmoving axis %s, in positive and negative direction\n" % str(axis_choice))
            report_file.write(
                "experiment date and time: %s\n" % str(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
            report_file.write("initial position: %s\n" % (str(init_position)))
            report_file.write("magnification: %s\n" % str(mag))
            report_file.write("experimental datapoints: \n%s\n" % str(datapoints))
            report_file.write("experimental data path: %s\n" % str(
                initial_path + os.sep + "moving x" + os.sep + "positive increment/negative increment"))
            report_file.write("procedure:\n")
            report_file.write("1) note the initial position of the object you want to use as probe to move\n")
            report_file.write("2) move up and down 4 um and return to the initial position\n")
            report_file.write("3) take an image (reference)\n")
            report_file.write(
                "4) move of the incremental quantity (positive if you add it or negative if you subtract it to the initial position)\n")
            report_file.write("5) return to the initial position\n")
            report_file.write("6) take an image after\n")
            report_file.write("the report is generated by PyFast-ADT v0.1.0\n")

        os.chdir(original_path)

    elif axis_choice == "z":
        pass

    elif axis_choice == "a":
        pass

    else:
        print("not available axis choice please pick x, y, z or a\nreturning to the main menu")


def calculate_shift_with_opencv(self, template, image, ref_center, method=cv2.TM_CCOEFF_NORMED):
    """
    Calculate the shift between a template and an image using OpenCV's matchTemplate.
    """
    # Perform cross-correlation using matchTemplate
    image = cv2.bilateralFilter(image, 9, 150, 150)
    template = cv2.bilateralFilter(template, 9, 150, 150)
    result = cv2.matchTemplate(image, template, method)

    # Find the location of the maximum correlation
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Compute the center of the template in the second image
    matched_center = (max_loc[0] + template.shape[1] // 2, max_loc[1] + template.shape[0] // 2)

    # Calculate the shift relative to the reference center
    shift_x = matched_center[0] - ref_center[0]
    shift_y = matched_center[1] - ref_center[1]

    return shift_y, shift_x, result, max_loc


def process_images_in_folder(self, image_folder, output_folder):
    # List all the files in the folder
    files = os.listdir(image_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Filter for image files and sort them by the numeric value in the filename
    image_files = sorted(
        [f for f in files if f.endswith('.tif') and not f.startswith('original_position')],
        key=lambda x: float(x.split('.tif')[0])  # Extract the numeric part before '.tif'
    )

    shifts_dx = []
    shifts_dy = []
    increments = []

    # Load the first image and select the ROI
    ref_image_path = os.path.join(image_folder, image_files[0])
    ref_image = cv2.imread(ref_image_path, cv2.IMREAD_GRAYSCALE)

    print("Select the ROI in the reference image (Image 1) and press Enter/Space to confirm.")
    roi = cv2.selectROI("Select ROI", ref_image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")

    x, y, w, h = map(int, roi)
    template = ref_image[y:y + h, x:x + w]
    ref_center = (x + w // 2, y + h // 2)

    for image_file in image_files:
        # Find the corresponding "original_position" image
        original_image_file = f"original_position_{image_file}"
        if original_image_file not in files:
            continue  # Skip if the counterpart is missing

        # Load the images
        image_path = os.path.join(image_folder, image_file)
        original_image_path = os.path.join(image_folder, original_image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

        # If we are at the second image or later, update the template
        if image_file != image_files[0]:
            shift_y, shift_x, result, max_loc = calculate_shift_with_opencv(self, template, image, ref_center)
            # Update the template based on the new position
            template = image[max_loc[1]:max_loc[1] + h, max_loc[0]:max_loc[0] + w]
            ref_center = (max_loc[0] + w // 2, max_loc[1] + h // 2)
            # calculate the real shift between the images
            shift_y, shift_x, result, max_loc = calculate_shift_with_opencv(self, template, original_image, ref_center)
        else:
            # Calculate the shift between the initial template and the first image
            shift_y, shift_x, result, max_loc = calculate_shift_with_opencv(self, template, original_image, ref_center)

        # Append the shift values and increment number
        shifts_dy.append(shift_y)
        shifts_dx.append(shift_x)
        increments.append(float(image_file.split('.tif')[0]))

        # Save the image and its counterpart subplot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Image {image_file}")
        plt.imshow(image, cmap='gray')
        plt.scatter(ref_center[0], ref_center[1], color='red', label='ROI Center')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title(f"Original Position {image_file}")
        plt.imshow(original_image, cmap='gray')
        plt.scatter(max_loc[0] + w // 2, max_loc[1] + h // 2, color='blue', label='Max Correlation')
        plt.arrow(
            ref_center[0], ref_center[1],
            shift_x, shift_y,
            color='blue', head_width=10, head_length=20, label="Shift Vector"
        )
        plt.legend()

        plt.tight_layout()
        subplot_path = os.path.join(output_folder, f"subplot_{image_file.split('.tif')[0]}.png")
        plt.savefig(subplot_path)
        plt.close()

    # Save the results to a CSV file
    csv_path = os.path.join(output_folder, 'resulting_shift.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Increment', 'Shift_dy', 'Shift_dx'])
        for inc, dy, dx in zip(increments, shifts_dy, shifts_dx):
            writer.writerow([inc, dy, dx])

    # Plot the shifts
    plt.figure(figsize=(12, 6))

    # Plot Shift in dy
    plt.subplot(1, 2, 1)
    plt.plot(increments, shifts_dy, marker='o', color='red', label="Shift in dy")
    plt.title("Shift in dy")
    plt.xlabel("Increment")
    plt.ylabel("Shift in dy")
    plt.grid(True)

    # Plot Shift in dx
    plt.subplot(1, 2, 2)
    plt.plot(increments, shifts_dx, marker='o', color='blue', label="Shift in dx")
    plt.title("Shift in dx")
    plt.xlabel("Increment")
    plt.ylabel("Shift in dx")
    plt.grid(True)

    plt.tight_layout()
    final_plot_path = os.path.join(output_folder, 'shifts_plot.png')
    plt.savefig(final_plot_path)
    plt.show()

def re_evaluate_backlash_data(self):
    data_path = tkinter.filedialog.askdirectory(title="Please select the folder where the backlash data are present")
    process_images_in_folder(self, data_path, data_path+os.sep+"results")

def backlash_correction_single_axis(self, tracking_initial_pos = None, speed = 1):
    """this function check for the 3 var for backlash correction x,y,z in the extra space.
    if one is ticked the backlash correction is performed for that axis, and iterate for the others."""
    if tracking_initial_pos != None:
        initial_pos = tracking_initial_pos
        self.tem.set_stage_position(x=initial_pos["x"], y=initial_pos["y"], z=initial_pos["z"], speed = speed)  #
        time.sleep(1)                                            #
    else:
        initial_pos = self.tem.get_stage()

    axes = []
    if self.get_x_backlash_correction_value(): axes.append("x")
    if self.get_y_backlash_correction_value(): axes.append("y")
    if self.get_z_backlash_correction_value(): axes.append("z")
    if axes == []:
        return

    print("axes choosen: %s" %str(axes))

    for axis in axes:
        choosen_pos = initial_pos[axis]

        print("starting backlash correction for %s axis" %str(axis))
        self.tem.set_stage_position(**{axis: choosen_pos + 4}, speed = speed)
        time.sleep(1)
        self.tem.set_stage_position(**{axis: choosen_pos - 4}, speed = speed)
        time.sleep(1)
        self.tem.set_stage_position(**{axis: choosen_pos}, speed = speed)
        time.sleep(1)

def backlash_correction_alpha(self, exp_type, start_angle, final_angle, rotation_speed=0.7, rotation_speed_cred=0.3):
    """backlash correction can work in different ways as a function of the input parameters.
     1) if stepwise is used, the correction is made by steps. instead in continuous the step is made in a single continuous step.
     2) type can be chosen between 'normal' and 'high precision', this flag is chosen by a checkbox in the gui.
        if not ticked, the normal mode is chosen, else high precision is selected and a fake rotation its added before
        the acquisition to increase the reproducibility of the goniometer."""
    if self.get_a_backlash_correction_value() != True:
        return
    else:
        if self.get_high_performance_value() == True: type = "high precision"
        else: type = "normal"

        if self.tem.__class__.__name__ == "Tem_fei_temspy":
            print(self.tem.__class__.__name__, "guard for backlash correction")
            rotate = self.tem.set_alpha_temspy
        else:
            print(self.tem.__class__.__name__, "guard for backlash correction")
            rotate = self.tem.set_alpha

        if exp_type == "stepwise":
            if start_angle < final_angle: sign = -1
                # add a fake step rotation before taking the track data in steps # this is for the 3rd plot
                # for angle_ in list(np.round(np.arange(start_angle, final_angle - (sign), abs(sign), dtype=np.float32), 2)):
                #     self.tem.set_alpha(angle_)
                #     time.sleep(0.33)
            else: sign = 1
                # add a fake step rotation before taking the track data in steps
                # for angle_ in list(np.round(np.arange(start_angle, final_angle - (sign), -sign, dtype=np.float32), 2)):
                #     self.tem.set_alpha(angle_)            #     time.sleep(0.33)

            if type == "high precision":
                # add a fake rotation before taking the track data
                rotate(start_angle, velocity = rotation_speed)
                rotate(final_angle, velocity = rotation_speed)

            # backlash correction
            rotate(start_angle + (sign * 3), velocity = rotation_speed)
            time.sleep(1)
            rotate(start_angle + (sign * 2), velocity = rotation_speed)
            time.sleep(1)
            rotate(start_angle + (sign * 1), velocity = rotation_speed)
            time.sleep(1)
            rotate(start_angle, velocity = rotation_speed)
            time.sleep(3)

        elif exp_type == "continuous":
            if start_angle < final_angle: sign = -1
            else: sign = 1

            if type == "high precision":
                # add a fake rotation before taking the track data
                rotate(start_angle, velocity =rotation_speed_cred)
                time.sleep(1)
                rotate(final_angle, velocity =rotation_speed_cred)

            # backlash correction
            rotate(start_angle + (sign * 3), velocity = rotation_speed)
            time.sleep(1)
            rotate(start_angle, velocity=rotation_speed_cred)
            time.sleep(3)


def backlash_stage_acquisition(self):
    """script to perform the experiment of the 29/01/2025 for backlash characterization of the TEM goniometer. the main
    difference wrt backlash_data_acquisition function is that here the coordinates of the stage are used instead of
    reference images from detectors. """
    # 1) note the initial coordinate -124.69 um in this case
    init_position = self.tem.get_stage()

    init_x = float(init_position["x"])
    init_y = float(init_position["y"])
    init_z = float(init_position["z"])
    init_a = float(init_position["a"])

    mag = self.tem.get_magnification()
    axis_choice = input("pick an axis to probe backlash: x, y, z, a")
    speed = 1
    sleeper = 1

    columns = ["datapoint_label", "x_before", "y_before", "z_before", "a_before", "x_after", "y_after", "z_after", "a_after"]
    df = pd.DataFrame(columns=columns)


    if axis_choice != "a":
        # datapoints = np.round(np.linspace(0.5, 10, 20), 1)
        datapoints = np.round(np.linspace(0.5, 5, 10), 1)
    else:
        datapoints1 = np.linspace(0.1, 5, 14)
        datapoints2 = np.linspace(8, 65, 20)
        datapoints = np.concatenate((datapoints1, datapoints2))
    print("starting the procedure...")

    if axis_choice == "x":
        # positive increment
        original_path = os.getcwd()
        initial_path = self.get_dir_value()
        os.makedirs(initial_path, exist_ok=True)
        os.makedirs(initial_path+os.sep+"moving x", exist_ok=True)
        os.chdir(initial_path+os.sep+"moving x")
        os.makedirs("positive increment", exist_ok=True)
        path_positive = initial_path + os.sep + "moving x" + os.sep + "positive increment"
        os.chdir(path_positive)


        for i, datapoint_label in enumerate(datapoints):
            print("starting datapoint + %s, %s / %s" %(str(datapoint_label), str(i+1), str(len(datapoints))))
            # 2) move up and down 4 um (-120.69 and after -128.69) and return to the initial position (identical for both + or – series!)
            self.tem.set_stage_position(x=init_x + 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(x=init_x - 4, speed=speed)
            time.sleep(sleeper)
            self.tem.set_stage_position(x=init_x, speed=speed)
            time.sleep(sleeper)
            # 3) take an image (reference) (here more or less i'm always in the same spot)
            reference_datapoint = self.tem.get_stage()
            ref_x, ref_y, ref_z, ref_a = reference_datapoint["x"], reference_datapoint["y"], reference_datapoint["z"], reference_datapoint["a"]
            time.sleep(sleeper)

            # 4) move of the quantity wanted (i.e. 0.1 or 0.x) (here you decide + or - series)
            self.tem.set_stage_position(x=init_x + datapoint_label, speed=speed)
            time.sleep(sleeper)
            # 5) return to -124.69 um
            self.tem.set_stage_position(x=init_x, speed=speed)
            time.sleep(sleeper)
            # 6) take an image after
            datapoint = self.tem.get_stage()
            after_x, after_y, after_z, after_a = datapoint["x"], datapoint["y"], datapoint["z"], datapoint["a"]

            # 7) compare images in step 3 and 6, or save the data and iterate
            # Append results to DataFrame
            df.loc[len(df)] = [datapoint_label, ref_x, ref_y, ref_z, ref_a, after_x, after_y, after_z, after_a]
            time.sleep(sleeper)

        # Save raw data to CSV
        df.to_csv("raw_data_stage_positive.csv", index=False)
        # Compute shifts
        df["Shift_dx"] = df["x_after"] - df["x_before"]
        df["Shift_dy"] = df["y_after"] - df["y_before"]
        df["Shift_dz"] = df["z_after"] - df["z_before"]
        df["Shift_dalpha"] = df["a_after"] - df["a_before"]
        # Create the resulting shift DataFrame
        shift_df = df[["datapoint_label", "Shift_dy", "Shift_dx", "Shift_dz", "Shift_dalpha"]]
        shift_df.rename(columns={"datapoint_label": "Increment"}, inplace=True)
        # Save processed shift data
        shift_df.to_csv("resulting_shift.csv", index=False)


        # # repeat for the negative increment
        # os.chdir(initial_path + os.sep + "moving x")
        # os.makedirs("negative increment", exist_ok=True)
        # path_negative = initial_path + os.sep + "moving x" + os.sep + "negative increment"
        # os.chdir(path_negative)
        #
        # for i, datapoint_label in enumerate(datapoints):
        #     print("starting datapoint - %s, %s / %s" % (str(datapoint_label), str(i+1), str(len(datapoints))))
        #     # 2) move up and down 4 um (-120.69 and after -128.69) and return to the initial position (identical for both + or – series!)
        #     self.tem.set_stage_position(x=init_x + 4, speed=speed)
        #     time.sleep(sleeper)
        #     self.tem.set_stage_position(x=init_x - 4, speed=speed)
        #     time.sleep(sleeper)
        #     self.tem.set_stage_position(x=init_x, speed=speed)
        #     time.sleep(sleeper)
        #     # 3) take an image (reference) (here more or less i'm always in the same spot)
        #     reference_datapoint = self.cam.acquire_image(exposure_time = exposure, binning = binning, processing = processing)
        #     reference_datapoint = cv2.normalize(reference_datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #     # 4) move of the quantity wanted (i.e. 0.1 or 0.x) (here you decide + or - series)
        #     self.tem.set_stage_position(x=init_x - datapoint_label, speed=speed)
        #     time.sleep(sleeper)
        #     # 5) return to -124.69 um
        #     self.tem.set_stage_position(x=init_x, speed=speed)
        #     time.sleep(sleeper)
        #     # 6) take an image after
        #     datapoint = self.cam.acquire_image(exposure_time = exposure, binning = binning, processing = processing)
        #     datapoint = cv2.normalize(datapoint, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        #     # 7) compare images in step 3 and 6, or save the data and iterate
        #
        #     # format tiff uncompressed data
        #     original_name = "original_position_" + str(datapoint_label) + ".tif"
        #     datapoint_name = str(datapoint_label) + ".tif"
        #     imageio.imwrite(datapoint_name, datapoint)
        #     imageio.imwrite(original_name, reference_datapoint)
        #     time.sleep(sleeper)
        # # evaluate the shifts in the positive run
        # process_images_in_folder(self, path_negative, path_negative+os.sep+"results")

        print("experiment finished, writing report")
        os.chdir(initial_path + os.sep + "moving x")
        with open("details.txt", "w") as report_file:
            # Write the report contents
            report_file.write("Backlash Testing Report\nmoving axis %s, in positive and negative direction\n" %str(axis_choice))
            report_file.write("experiment date and time: %s\n" %str(datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S')))
            report_file.write("initial position: %s\n" %(str(init_position)))
            report_file.write("magnification: %s\n" %str(mag))
            report_file.write("experimental datapoints: \n%s\n" %str(datapoints))
            report_file.write("experimental data path: %s\n" %str(initial_path + os.sep + "moving x"+os.sep+"positive increment/negative increment"))
            report_file.write("procedure:\n")
            report_file.write("1) note the initial position of the object you want to use as probe to move\n")
            report_file.write("2) move up and down 4 um and return to the initial position\n")
            report_file.write("3) take an image (reference)\n")
            report_file.write("4) move of the incremental quantity (positive if you add it or negative if you subtract it to the initial position)\n")
            report_file.write("5) return to the initial position\n")
            report_file.write("6) take an image after\n")
            report_file.write("the report is generated by PyFast-ADT v0.1.0\n")

        os.chdir(original_path)

    elif axis_choice == "y":
        pass

    elif axis_choice == "z":
        pass

    elif axis_choice == "a":
        pass

    else:
        print("not available axis choice please pick x, y, z or a\nreturning to the main menu")

def evaluate_average_displacement_track_precision():
    """ aaaa """
    pass











