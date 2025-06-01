# handpanel for digital micrograph in tkinter for FEI tecnai
# the script generate a tkinter window with the basic commands used in the handpanel of a tecnai microscope, necessary to handle 3ded experiments
try:
    import temscript
except Exception as err:
    print('import failed:', err)
try:
    import DigitalMicrograph as DM
    DM.ClearResults()
except Exception as err:
    print('import failed:', err)
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askdirectory
import os
import sys
import json
import numpy as np
from scipy.interpolate import interp1d
print(sys.path)
dir_path = os.getcwd()
#dir_path = r"/python_FEI_FastADT/07042023_build_up/main"
if dir_path in sys.path:
    print('dir already there')
else:
    sys.path.append(dir_path)

from fast_adt_func import *
from beam_calibration import BeamCalibration
from server_socket_stream import server_img

class FastADT(tk.Toplevel):
    def __init__(self, master=None, brand = 'fei', camera = 'timepix1'):
        super().__init__(master)
        # camera initialization
        self.camera = camera
        if self.camera == 'timepix1':
            from adaptor.camera.adaptor_timepix1 import Cam_timepix1
            self.cam = Cam_timepix1(_id=0, instance_gui = self)
        elif self.camera == 'xf416r':
            from adaptor.camera.adaptor_xf416r_GPU import Cam_xf416r
            self.cam = Cam_xf416r(instance_gui = self)
        elif self.camera == 'us4000':
            from adaptor.camera.adaptor_us4000 import Cam_us4000
            self.cam = Cam_us4000(instance_gui = self)
        elif self.camera == 'us2000':
            from adaptor.camera.adaptor_us2000 import Cam_us2000
            self.cam = Cam_us2000(instance_gui = self)
        elif self.camera == 'ceta':
            from adaptor.camera.adaptor_ceta import Cam_ceta
            self.cam = Cam_ceta(instance_gui = self)
        elif self.camera == 'medipix3':
            from adaptor.camera.adaptor_serval import Cam_medipix3
            self.cam = Cam_medipix3(instance_gui = self)
        elif self.camera == 'merlin':
            from adaptor.camera.adaptor_merlin import Cam_merlin
            self.cam = Cam_merlin(instance_gui = self)
        elif self.camera == 'power_user':
            from adaptor.camera.adaptor_simulator import Cam_simulator
            self.cam = Cam_simulator(instance_gui = self)
            # self.cam_table = None
        else:
            print('camera not supported or recognized, exit')
            return
        try:
            self.cam_table = self.cam.load_calibration_table()
            self.live_view_gui()
        except Exception as err:
            print(err)

        # interface/microscope initialization
        self.brand = brand
        if self.brand == 'fei':
            from adaptor.microscope.adaptor_fei import Tem_fei
            self.tem = Tem_fei(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table, master = self)
        elif self.brand == 'jeol':
            from adaptor.microscope.adaptor_jeol import Tem_jeol
            # self.tem = Tem_jeol(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table)
            self.tem = Tem_jeol(ip_gonio = self.cam_table["ip_goniotool"][0], port_gonio = self.cam_table["ip_goniotool"][1], cam_table=self.cam_table, master = self)

        elif self.brand == 'gatan_fei':
            from adaptor.microscope.adaptor_gatan_fei import Tem_gatan_fei
            self.tem = Tem_gatan_fei(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table, master = self)
        elif self.brand == 'gatan_jeol':
            from adaptor.microscope.adaptor_gatan_jeol import Tem_gatan_jeol
            self.tem = Tem_gatan_jeol(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table, master = self)
        elif self.brand == 'fei_temspy':
            from adaptor.microscope.adaptor_fei_temspy import Tem_fei_temspy
            self.tem = Tem_fei_temspy(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table, master = master)
        elif self.brand == 'power_user':
            self.tem = None
        else:
            print('brand not supported or recognized, exit')
            return

        self.zero_pos_row = 1
        self.zero_pos_col = 0
        self.title("<< PyFAST-ADT >>")
        self.geometry("415x980")
        #self.label = tk.Label(self, text="This is a child window")
        #self.label.grid(pady=20, row=0)
        self.create_widgets()
        self.change_colors("#444444")

        self.widget_status()
        self.var12 = None
        self.var11 = None
        self.tracking_images_done = False
        self.tracking_done = False
        self.ub = BeamCalibration(cam_table = self.cam_table)
        self.haadf = None
        self.tracking_precision_running = False

        self.init_position_stage_tracking = None

        # block for streaming of the live feed of the camera
        if self.camera != 'power_user':
            if self.cam_table.streamable == True:
                self.server_stream = server_img()
                if self.camera == 'medipix3':
                    # self.server_thread = threading.Thread(target=self.server_stream.send_image, daemon=True)
                    pass
                else:
                    self.server_thread = threading.Thread(target=self.server_stream.send_image, daemon=True)
                self.server_thread.start()
                self.server_stream.send_img = False
                self.server_stream.image = np.zeros((512,512), dtype = np.uint8)


    def change_colors(self, new_bg_color):
        # Change the color of the root window (self)
        self.configure(bg=new_bg_color)
        # Change the color of all widgets and frames
        for widget in self.winfo_children():
            try:
                widget.configure(bg=new_bg_color)
            except: pass
            if isinstance(widget, tk.Frame):
                for frame_widget in widget.winfo_children():
                    try:
                        frame_widget.configure(bg=new_bg_color)
                    except:pass

            # Change the foreground color of all labels and text widgets
            for frame_widget in widget.winfo_children():
                if isinstance(frame_widget, (tk.Label, tk.Text)):
                    try:
                        frame_widget.configure(fg='white')
                    except:pass
            # Change the background color of all buttons
            for frame_widget in widget.winfo_children():
                if isinstance(frame_widget, tk.Button):
                    try:
                        frame_widget.configure(bg="gray")
                    except: pass

                elif isinstance(frame_widget, (tk.Entry, tk.Spinbox)):
                    try:
                        frame_widget.configure(bg='white')
                    except:pass

                elif isinstance(frame_widget, tk.Checkbutton):
                    try:
                        frame_widget.configure(fg='dark gray')
                    except:pass

    def change_beam_settings_image(self, slot = 1):
        """if slot =0, reset colors to both blank. if equal 1 change the image beam setting color to green and the diff
        beam setting to blank. if equal 2 change both to green"""
        # box_img is the green image and box_img2 is the blank image
        if slot ==1:
            self.box_image_label.configure(image=self.box_img)
            self.box_image2_label.configure(image=self.box_img2)
        if slot ==2:
            self.box_image_label.configure(image=self.box_img)
            self.box_image2_label.configure(image=self.box_img)
        if slot ==0:
            self.box_image_label.configure(image=self.box_img2)
            self.box_image2_label.configure(image=self.box_img2)
    def tem_hide(self):
        if self.stem_var.get():
            self.stem_pixeltime_label.grid(row=3, column=0, padx=10, pady=5)
            self.stem_pixeltime_entry.grid(row=3, column=1, padx=10, pady=5)
            self.image_size_label.grid(row=3, column=2, padx=5, pady=5, sticky="w")
            self.image_size_combobox.grid(row=3, column=3, padx=5, pady=5, sticky="w")
            # remove the one of TEM mode
            self.tem_imagetime_label.grid_remove()
            self.tem_imagetime_entry.grid_remove()
            # switch of the camera for STEM imaging
            stem_mode_imaging(self)
            self.mag_label.grid_remove()
            self.mag_combobox.grid_remove()
            self.stem_binning_label.grid(row=4, column=2, padx=5, pady=5, sticky="w")
            self.stem_binning_combobox.grid(row=4, column=3, padx=5, pady=5, sticky="w")


        else:
            self.tem_imagetime_label.grid(row=3, column=0, padx=10, pady=5)
            self.tem_imagetime_entry.grid(row=3, column=1, padx=10, pady=5)
            # remove the one of STEM mode
            self.stem_pixeltime_label.grid_remove()
            self.stem_pixeltime_entry.grid_remove()
            self.image_size_label.grid_remove()
            self.image_size_combobox.grid_remove()
            self.stem_binning_label.grid_remove()
            self.stem_binning_combobox.grid_remove()
            # switch of the camera for TEM imaging
            self.haadf = None
            self.stem_binning_label.grid_remove()
            self.stem_binning_combobox.grid_remove()
            self.mag_label.grid(row=4, column=2, padx=5, pady=5, sticky="w")
            self.mag_combobox.grid(row=4, column=3, padx=5, pady=5, sticky="w")

    def create_widgets(self):
        ############# Parameter Setup #############
        # add separator
        self.separator1 = tk.Frame(self, height=100, width=325, bd = 10)
        self.separator1.grid(row=self.zero_pos_row, column=self.zero_pos_col, rowspan=4, columnspan=4)
        self.label1 = tk.Label(self.separator1, text = "Parameter Setup").grid(row = 0, column=0, columnspan=2, sticky="w")
        # Create buttons Stage
        self.button_width = 5
        self.button_height = 2

        self.seq_var = tk.BooleanVar()
        self.seq_check = tk.Checkbutton(self.separator1, text="Stepwise", variable=self.seq_var)
        self.seq_check.grid(row=1, column=0, columnspan = 2, padx=5, sticky="w")

        self.cont_var = tk.BooleanVar()
        self.cont_check = tk.Checkbutton(self.separator1, text="Continuous", variable=self.cont_var)
        self.cont_check.grid(row=1, column=1, columnspan = 2, padx=5, sticky="w")
        self.cont_check.select()

        self.tem_var = tk.BooleanVar()
        self.tem_check = tk.Checkbutton(self.separator1, text="TEM Imaging", variable=self.tem_var)
        self.tem_check.grid(row=2, column=0, columnspan = 2, padx=5, sticky="w")
        self.tem_check.select()

        self.stem_var = tk.BooleanVar()
        self.stem_check = tk.Checkbutton(self.separator1, text="STEM Imaging", variable=self.stem_var, command = self.tem_hide)
        self.stem_check.grid(row=2, column=1, columnspan = 2, padx=5, sticky="w")

        # create the PixelTime label and entry #########################################################################
        self.stem_pixeltime_label = tk.Label(self.separator1, text="DwellTime (us):")
        #self.stem_pixeltime_label.grid(row=3, column=0, padx=(5, 5), pady=5, sticky="e")

        self.stem_pixeltime_entry = tk.Entry(self.separator1, width=self.button_width)
        self.stem_pixeltime_entry.insert(0, "1.0")
        #self.stem_pixeltime_entry.grid(row=3, column=1, padx=(5, 5), pady=5, sticky="w")
        # create the image_size label and combobox for stem  #############################################################
        self.image_size_label = tk.Label(self.separator1, text="STEM img size:")
        #self.image_size_label.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.image_size_values = ["FULL", "HALF", "QUARTER"]

        self.image_size_combobox = ttk.Combobox(self.separator1, values=self.image_size_values, width=self.button_width)
        #self.image_size_combobox.grid(row=3, column=3, padx=5, pady=5, sticky="w")
        self.image_size_combobox.current(0)

        self.stem_binning_label = tk.Label(self.separator1, text="STEM binning:")
        # self.image_size_label.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        self.stem_binning_values = ["1", "2", "4", "8"]

        self.stem_binning_combobox = ttk.Combobox(self.separator1, values=self.stem_binning_values, width=self.button_width)
        # self.image_size_combobox.grid(row=3, column=3, padx=5, pady=5, sticky="w")
        self.stem_binning_combobox.current(1)
        ################################################################################################################
        # create the KL label and combobox
        self.KL_label = tk.Label(self.separator1, text="Cam Length (mm):").grid(row=5, column=2, padx=5, pady=5, sticky="w")
        if self.brand == "power_user":
            self.KL_values = ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
        else:
            self.tem.load_calibration_table(self.cam_table)
            self.KL_values = list(self.tem.kl_index_table.keys())

        self.KL_combobox = ttk.Combobox(self.separator1, values=self.KL_values, width=self.button_width)
        self.KL_combobox.grid(row=5, column=3, padx=5, pady=5, sticky="w")
        self.KL_combobox.current(8)

        #################################################################################################################################
        if self.brand == "power_user":
            self.mag_values = ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0"]
        else:
            self.mag_values = list(self.tem.mag_index_table.keys())
        self.mag_label = tk.Label(self.separator1, text="Magnification:")
        self.mag_combobox = ttk.Combobox(self.separator1, values=self.mag_values, width=self.button_width)
        self.mag_combobox.grid(row=4, column=3, padx=5, pady=5, sticky="w")
        #################################################################################################################################

        self.diff_lens_label = tk.Label(self.separator1, text="Diffraction Lens:").grid(row=6, column=2, padx=5, pady=5, sticky="w")
        self.diff_lens_entry = tk.Entry(self.separator1, width=self.button_width+3)
        self.diff_lens_entry.grid(row=6, column=3, padx=(5, 5), pady=5, sticky="w")
        self.diff_lens_entry.insert(0, "0.0000")

        # create the ImageTime label and entry
        self.tem_imagetime_label = tk.Label(self.separator1, text="ImageTime (s): ")
        self.tem_imagetime_label.grid(row=3, column=0, padx=10, pady=5)

        self.tem_imagetime_entry = tk.Entry(self.separator1, width=self.button_width)
        self.tem_imagetime_entry.insert(0, "0.05")
        self.tem_imagetime_entry.grid(row=3, column=1, padx=10, pady=5)


        # create the angle and tilt step labels and entries
        self.angle_label = tk.Label(self.separator1, text="Initial Angle (°): ")
        self.angle_label.grid(row=4, column=0, padx=10, pady=5)

        self.angle_entry = tk.Entry(self.separator1, width=self.button_width)
        self.angle_entry.insert(0, "0.0")
        self.angle_entry.grid(row=4, column= 1, padx=10, pady=5)

        self.final_angle_label = tk.Label(self.separator1, text="Final Angle (°): ")
        self.final_angle_label.grid(row=5, column=0, padx=10, pady=5)

        self.final_angle_entry = tk.Entry(self.separator1, width=self.button_width)
        self.final_angle_entry.insert(0, "0.0")
        self.final_angle_entry.grid(row=5, column=1, padx=10, pady=5)

        self.tilt_step_label = tk.Label(self.separator1, text="Tilt Step (°/img): ")
        self.tilt_step_label.grid(row=6, column=0, padx=10, pady=5)

        self.tilt_step_entry = tk.Entry(self.separator1, width=self.button_width)
        self.tilt_step_entry.insert(0, "1.0")
        self.tilt_step_entry.grid(row=6, column=1, padx=10, pady=5)

        # Create widgets
        self.go_to_label = tk.Button(self.separator1, text="GoTo (°):", command=lambda: go_to(self)).grid(row=7, column=0)

        self.go_to_entry = tk.Entry(self.separator1, width=self.button_width)
        self.go_to_entry.grid(row=7, column=1)
        self.go_to_entry.insert(0, "0.0")

        self.undo_button = tk.Button(self.separator1, text="Undo", command= lambda: undo(self))
        self.undo_button.grid(row=7, column=2)

        ########### label for expected time #######################

        self.expected_time_value = "0.0"
        self.expected_speed_value = "0.0"
        #self.expected_time_label1 = tk.Label(self.separator1, text="Exp. time, gonio velocity:")
        #self.expected_time_label1.grid(row=8, column=0, sticky="w")
        self.expected_time_label2 = tk.Label(self.separator1, text="Exp. time: " + self.expected_time_value + " s,\tgonio velocity: " + self.expected_speed_value + " °/s")
        self.expected_time_label2.grid(row=8, column=0, sticky="ew", columnspan = 3)

        ############# Camera Settings for Diffraction #############
        self.zero_pos_row = 9

        # add separator
        self.separ1 = ttk.Separator(self, orient = "horizontal").grid(row = 7, column = 0, columnspan = 4, sticky = "ew")

        self.separator1point5 = tk.Frame(self, height=100, width=325)
        self.separator1point5.grid(row=self.zero_pos_row, column=self.zero_pos_col, rowspan=4, columnspan=4)
        self.label1point5 = tk.Label(self.separator1point5, text=" Camera Setting for Diffraction").grid(row=0, column=0, columnspan = 2, sticky="w")

        # Create widgets for Exposure label and Entry
        self.exposure_label = tk.Label(self.separator1point5, text="Exposure (s):").grid(row=1, column=0, padx=5, pady=5, sticky="w")

        self.exposure_entry = tk.Entry(self.separator1point5, width = self.button_width)
        self.exposure_entry.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.exposure_entry.insert(0, "0.5")

        # Create widgets for Binning label and Combobox
        self.binning_label = tk.Label(self.separator1point5, text="Binning:").grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="w")
        self.binning_values = ["1", "2", "4", "8"]

        self.binning_combobox = ttk.Combobox(self.separator1point5, values=self.binning_values, width = self.button_width)
        self.binning_combobox.grid(row=1, column=3, columnspan=2, padx=5, pady=5, sticky="w")
        self.binning_combobox.current(1)  # set default value to first item in list

        # Create widgets for Processing label and Combobox
        self.processing_label = tk.Label(self.separator1point5, text="Processing:").grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.processing_values = ["Gain normalized", "Background subtracted", "Unprocessed"]

        self.processing_combobox = ttk.Combobox(self.separator1point5, values=self.processing_values)
        self.processing_combobox.grid(row=2, column=1, columnspan=2, padx=5, pady=5, sticky="w")
        self.processing_combobox.current(0)  # set default value to first item in list

        # Create widget for Acquire Camera Image button
        self.acquire_button = tk.Button(self.separator1point5, text="Acquire Camera Image", command=lambda: acquire_image_and_show(self, self.exposure_value(), self.binning_value(), self.processing_value()))
        self.acquire_button.grid(row=3, column=0, columnspan=3, padx=5, pady=5)

        self.zero_pos_row = 0

        ############# Beam Settings #############
        self.zero_pos_row = 14

        # add separator
        self.separ2 = ttk.Separator(self, orient="horizontal").grid(row=13, column=0, columnspan=4, sticky="ew")

        self.separator2 = tk.Frame(self, height=100, width=325)
        self.separator2.grid(row=self.zero_pos_row, column=self.zero_pos_col, rowspan=4, columnspan=4)
        self.label2 = tk.Label(self.separator2, text="Beam Setting").grid(row=0, column=0, columnspan = 2, sticky = "w")

        # Checkbox and buttons in first row
        self.beamblank_checkbox_var = tk.BooleanVar()
        self.beamblank_checkbox = tk.Checkbutton(self.separator2, text="Beam Blank",command=lambda: self.tem.beam_blank(self.beamblank_value()), variable=self.beamblank_checkbox_var)
        self.beamblank_checkbox.grid(row=1, column=0)

        self.get_beam_settings_button = tk.Button(self.separator2, text="Get Beam Settings", command=lambda: get_beam_intensity(self))
        self.get_beam_settings_button.grid(row=1, column=1)

        # Box image and buttons in second row
        self.path1 = "green.gif"
        self.path2 = "white.gif"
        #breakpoint()
        # add separator
        self.zero_pos_row = 20
        self.separator3 = tk.Frame(self, height=100, width=325)
        self.separator3.grid(row=self.zero_pos_row, column=self.zero_pos_col, rowspan=4, columnspan=4)

        self.box_img = tk.PhotoImage(file=self.path1)  # Replace with your image path
        self.box_img2 = tk.PhotoImage(file=self.path2)  # Replace with your image path

        self.box_image_label = tk.Label(self.separator3, image=self.box_img2)
        self.box_image2_label = tk.Label(self.separator3, image=self.box_img2)

        self.box_image_label.grid(row=2, column=0, pady=10)
        self.box_image2_label.grid(row=2, column=3, pady=10)

        self.set_img_settings_button = tk.Button(self.separator3, text="Set img Setting", command=lambda: set_image_setting(self))
        self.set_img_settings_button.grid(row=2, column=1, padx=5, pady=5)

        self.set_diff_settings_button = tk.Button(self.separator3, text="Set Diff Setting", command=lambda: set_diff_setting(self))
        self.set_diff_settings_button.grid(row=2, column=2, padx=5, pady=5)

        #separator4
        # self.zero_pos_row = 25
        #
        # self.separator4 = tk.Frame(self, height=100, width=325)
        # self.separator4.grid(row=self.zero_pos_row, column=self.zero_pos_col, rowspan=4, columnspan=4)

        # Beam Shift Calibration in third row
        # Labels and entry in third row
        self.beam_shift_calibration_label = tk.Label(self.separator3, text="Beam Shift Calibration")
        self.beam_shift_calibration_label.grid(row=3, column=0, columnspan = 2, sticky = "w", pady= 5)
        # ################################ beam shift value is not used right now ##########################
        # self.beam_shift_value_label = tk.Label(self.separator4, text="Beam Shift Value")
        # self.beam_shift_value_label.grid(row=4, column=0, columnspan = 1, sticky = "w")
        # self.beam_shift_var = tk.DoubleVar()
        # self.beam_shift_entry = tk.Entry(self.separator4, textvariable=self.beam_shift_var, width = 10)
        # self.beam_shift_entry.grid(row=4, column=1, sticky = "w", padx = 5)
        ############################### we can remove up to here #########################################
        # Buttons in fourth row
        self.test_beam_shift_value_button = tk.Button(self.separator3, text="Test Beam Shift Value", command=lambda: test_beam_shift(self))
        self.test_beam_shift_value_button.grid(row=4, column=0, columnspan = 2, pady = 5)
        self.calibrate_button = tk.Button(self.separator3, text="   Calibrate   ", command=lambda: calibrate_beam_shift(self))
        self.calibrate_button.grid(row=4, column=2, columnspan = 2, pady = 5)

        ############# Crystal Tracking File #############
        # separator5
        self.zero_pos_row = 30
        self.separ5 = ttk.Separator(self, orient="horizontal").grid(row=29, column=0, columnspan=4, sticky="ew")

        self.separator5 = tk.Frame(self, height=100, width=325)
        self.separator5.grid(row=self.zero_pos_row, column=0, rowspan=4, columnspan=4)
        self.tracking_label = tk.Label(self.separator5, text="Tracker method:").grid(row = 0, column = 0, sticky = "w", pady= 10)

        self.track_method_values = ["KF", "patchworkCC","pureKF", "CC", "manual", "debug", "no tracking", "prague_cred_method", "tracking_precision", "semi-manual stepwise", "a priori + in situ"]

        self.method_combobox = ttk.Combobox(self.separator5, values=self.track_method_values)
        self.method_combobox.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.method_combobox.current(6)  # set default value to first item in list ################################################

        # alpha backlash
        self.a_backlash_var = tk.BooleanVar()
        self.backlash_correction_check = tk.Checkbutton(self.separator5, text="alpha backlash", variable=self.a_backlash_var)
        self.backlash_correction_check.grid(row=0, column=2, columnspan=2, padx=5, sticky="w")
        self.backlash_correction_check.deselect()
        # other axis backlash, these checkboxes are in the additional space
        self.x_backlash_var = tk.BooleanVar()
        self.y_backlash_var = tk.BooleanVar()
        self.z_backlash_var = tk.BooleanVar()
        self.init_position_var = tk.BooleanVar()

        self.hiper_var = tk.BooleanVar()
        self.hiper_correction_check = tk.Checkbutton(self.separator5, text="HiPer gonio?", variable=self.hiper_var)
        self.hiper_correction_check.grid(row=1, column=2, columnspan=2, padx=5, sticky="w")
        self.hiper_correction_check.deselect()

        # Buttons in fourth row
        #def acquire_tracking_images(self, tracking_path = None):
        #return self.tracking_images, self.track_angles
        self.acquire_tracking_button = tk.Button(self.separator5, text="Acquire (...)", command=lambda: acquire_tracking_images(self, tracking_path=None))
        self.acquire_tracking_button.grid(row=2, column=0, pady=5, sticky = "w")

        #def process_tracking_images(self, tracking_path, tracking_images, track_angles, method):
        #return self.tracking_positions, self.track_result
        self.process_tracking_button = tk.Button(self.separator5, text="Process (...)", command=lambda: process_tracking_images(self, tracking_images = self.tracking_images, track_angles= None, method= self.get_tracking_method()))
        self.process_tracking_button.grid(row=2, column=1, pady=5, sticky = "w")

        self.display_tracking_button = tk.Button(self.separator5, text="Display Track",
                                                 command=lambda: display_crystal_tracking(self, method = self.get_tracking_method()))
        self.display_tracking_button.grid(row=2, column=2, pady=5, sticky="w")

        self.tracking_label2 = tk.Label(self.separator5, text="Tracker Tilt Step (°):").grid(row=3, column=0, sticky="w", pady=10)

        self.tracking_step_var = tk.DoubleVar()
        self.tracking_step_var.set(1.0)

        self.tracking_step_entry = tk.Entry(self.separator5, textvariable=self.tracking_step_var, width=10)
        self.tracking_step_entry.grid(row=3, column=1, sticky="w", padx=5)

        self.reset_cry_tracking_button = tk.Button(self.separator5, text="Reset Cryst Track. Steps", command=lambda: reset_tracking_images(self))
        self.reset_cry_tracking_button.grid(row=4, column = 0, pady=5, sticky="ew")

        self.generate_tracking_button = tk.Button(self.separator5, text="Generate Tracking File", command=lambda: generate_tracking_file(self, method = self.get_tracking_method(), text ="tracking.txt"))
        self.generate_tracking_button.grid(row=4, column = 1, pady=5, sticky="ew")

        ############# Acquisition #############
        # separator6_5
        self.zero_pos_row = 42

        self.separator6_5 = tk.Frame(self, height=100, width=200)
        self.separator6_5.grid(row=self.zero_pos_row, column=0, rowspan=4, columnspan=3, sticky = 'ew')

        ######### destination_folder_data #######
        tk.Label(self.separator6_5, text='Directory:').grid(row=0, column=0, sticky='ew')
        tk.Label(self.separator6_5, text='Sample name:').grid(row=1, column=0, sticky='ew')

        self.var_dir = tk.StringVar(value=self.get_work_dir())
        self.var_sample = tk.StringVar(value='experiment')
        self.var_exp_num = tk.IntVar(value=1)

        self.dir_entry = tk.Entry(self.separator6_5, width=30, textvariable=self.var_dir)
        self.dir_entry.grid(row=0, column=1, sticky='ew')

        self.browse_button = tk.Button(self.separator6_5, text='Browse..', command=self.browse_dir)
        self.browse_button.grid(row=0, column=2, sticky='ew')

        self.sample_entry = tk.Entry(self.separator6_5, width=20, textvariable=self.var_sample)
        self.sample_entry.grid(row=1, column=1, sticky='ew')

        self.exp_num_spin = tk.Spinbox(self.separator6_5, width=7, from_=0, to=9999, increment=1, textvariable=self.var_exp_num)
        self.exp_num_spin.grid(row=1, column=2)

        self.open_dir_button = tk.Button(self.separator6_5, text='open dir', command=self.open_dir)
        self.open_dir_button.grid(row=0, column=3, sticky='ew')

        self.delete_last_button = tk.Button(self.separator6_5, text='del last exp', command=self.delete_last)
        self.delete_last_button.grid(row=1, column=3, sticky='ew')

        ###########################################
        # separator6
        self.zero_pos_row = 58

        self.separ6 = ttk.Separator(self, orient="horizontal").grid(row=41, column=0, columnspan=4, sticky="ew")

        self.separator6 = tk.Frame(self, height=100, width=325)
        self.separator6.grid(row=self.zero_pos_row, column=0, rowspan=4, columnspan=4)

        self.acquisition_label = tk.Label(self.separator6, text="Acquisition").grid(row=0, column=0,sticky = "w", pady=4)

        self.load_tracking_button = tk.Button(self.separator6, text="Load Tracking File", command=lambda: load_tracking_file(self))
        self.load_tracking_button.grid(row=1, column=0, padx = 15)

        self.initialize_beam_button = tk.Button(self.separator6, text="Initialize Beam Position", command=lambda: initialize_beam_position(self))
        self.initialize_beam_button.grid(row=1, column=1, padx = 15)

        self.tracking_label2 = tk.Label(self.separator6, text="Tracking File: ").grid(row=2, column=0, pady=5)

        self.tracking_label3 = tk.Label(self.separator6, text="File Not Loaded!").grid(row=2, column=1, sticky="w", pady=5)

        # separator7
        self.zero_pos_row = 68
        self.separator7 = tk.Frame(self, height=100, width=325)
        self.separator7.grid(row=self.zero_pos_row, column=0, rowspan=4, columnspan=4)

        self.start_experiment_button = tk.Button(self.separator7, text="Start", command=lambda: start_experiment(self))
        self.start_experiment_button.grid(row=3, column=0, pady=5, padx = 15)

        self.stop_experiment_button = tk.Button(self.separator7, text="Stop", command=lambda: stop_experiment(self))
        self.stop_experiment_button.grid(row=3, column=1, pady=5, padx = 15)

        self.load_initial_parameters_button = tk.Button(self.separator7, text="Load Initial Parameters", command=lambda: load_initial_parameters_experiment(self))
        self.load_initial_parameters_button.grid(row=3, column=2, pady=5, padx = 15)

        self.pets_checkbox_var = tk.BooleanVar()
        self.pets_checkbox = tk.Checkbutton(self.separator7, text="pets auto run", variable=self.pets_checkbox_var)
        self.pets_checkbox.grid(row=3, column=3)
        self.pets_checkbox.select()

        self.eadt_checkbox_var = tk.BooleanVar()
        self.eadt_checkbox = tk.Checkbutton(self.separator7, text="eadt auto run", variable=self.eadt_checkbox_var)
        self.eadt_checkbox.grid(row=2, column=3)
        self.eadt_checkbox_var.set(False)

        if self.brand in ["power_user", "fei", "fei_temspy"]:
            # Create a button that opens additional space
            open_space_button = tk.Button(self.separator1, text="extra feature", command=self.open_additional_space)
            open_space_button.grid(row=1, column=3, padx=5, pady=5, sticky="w")

    def open_additional_space(self):
        if self.brand in ["power_user", "fei", "fei_temspy"]:
            # Create a new Toplevel window
            self.new_window = tk.Toplevel(self.separator1)
            self.new_window.title("<< additional features >>")
            self.new_window.geometry("315x700")

            # Add new buttons and labels to the new window
            #label = tk.Label(self.new_window, text="re evaluate tracking precision")
            #label.grid(row=0, column=1, padx=5, pady=5, sticky="w")

            new_window_label1 = tk.Label(self.new_window, text="re-evaluate a tracking precision experiment").grid(row=0, column=1, padx=5, pady=5, sticky="w")
            new_button = tk.Button(self.new_window, text="re-evaluate tracking precision", command=lambda: re_evaluate_tracking_precision(self))
            new_button.grid(row=1, column=1, padx=5, pady=5, sticky="w")

            new_window_label2 = tk.Label(self.new_window, text="re-evaluate a crystal tracking path").grid(row=2, column=1, padx=5, pady=5, sticky="w")
            new_button = tk.Button(self.new_window, text="re-evaluate crystal tracking path", command=lambda: re_evaluate_crystal_tracking_path(self))
            new_button.grid(row=3, column=1, padx=5, pady=5, sticky="w")

            new_window_label3 = tk.Label(self.new_window, text="find the optimum eucentric Z height in pixels").grid(row=4, column=1, padx=5, pady=5, sticky="w")
            new_button = tk.Button(self.new_window, text="optimum eucentric height evaluation", command=lambda: automatic_eucentric_height(self))
            new_button.grid(row=5, column=1, padx=5, pady=5, sticky="w")

            new_window_label4 = tk.Label(self.new_window, text="backlash characterization").grid(row=6, column=1, padx=5, pady=5, sticky="w")
            new_button = tk.Button(self.new_window, text="single axis backlash experiment", command=lambda: backlash_data_acquisition(self))
            new_button.grid(row=7, column=1, padx=5, pady=5, sticky="w")

            new_window_label5 = tk.Label(self.new_window, text="re-evaluate backlash data").grid(row=8, column=1, padx=5, pady=5, sticky="w")
            new_button = tk.Button(self.new_window, text="re-evaluate single axis backlash data", command=lambda: re_evaluate_backlash_data(self))
            new_button.grid(row=9, column=1, padx=5, pady=5, sticky="w")

            # here to add the single axis backlash choice
            label_backlash_single_axis = tk.Label(self.new_window, text="backlash correction for individual axis for 3DED").grid(row=10, column=1, padx=5, pady=5, sticky="w")
            self.separator_backlash = tk.Frame(self.new_window, height=100, width=325, bd=10)
            self.separator_backlash.grid(row=11, column=0, rowspan=4, columnspan=6)
            self.x_backlash_correction_check = tk.Checkbutton(self.separator_backlash, text="x", variable=self.x_backlash_var)
            self.x_backlash_correction_check.grid(row=0, column=0, columnspan=1, padx=5, sticky="w")
            self.x_backlash_correction_check.deselect()

            self.y_backlash_correction_check = tk.Checkbutton(self.separator_backlash, text="y", variable=self.y_backlash_var)
            self.y_backlash_correction_check.grid(row=0, column=2, columnspan=1, padx=5, sticky="w")
            self.y_backlash_correction_check.deselect()

            self.z_backlash_correction_check = tk.Checkbutton(self.separator_backlash, text="z", variable=self.z_backlash_var)
            self.z_backlash_correction_check.grid(row=0, column=4, columnspan=1, padx=5, sticky="w")
            self.z_backlash_correction_check.deselect()

            self.init_position_check = tk.Checkbutton(self.separator_backlash, text="init_position", variable=self.init_position_var)
            self.init_position_check.grid(row=0, column=6, columnspan=1, padx=5, sticky="w")
            self.init_position_check.deselect()

            # z scan eucentric height method calculation
            new_window_label6 = tk.Label(self.new_window, text="z_scan eucentric height").grid(row=16, column=1, padx=5, pady=5, sticky="w")
            new_button_6 = tk.Button(self.new_window, text="z_scan_eucentric_height", command=lambda: eucentric_height_z_scan(self))
            new_button_6.grid(row=17, column=1, padx=5, pady=5, sticky="w")

            # z scan data acquisition
            new_window_label7 = tk.Label(self.new_window, text="z_scan eucentric height").grid(row=18, column=1, padx=5, pady=5, sticky="w")
            new_button_7 = tk.Button(self.new_window, text="acquire z_scan", command=lambda: acquire_z_scan_tem_mode(self))
            new_button_7.grid(row=19, column=1, padx=5, pady=5, sticky="w")

            # add here combobox for self.speed_tracking to change it dinamically
            # if the widget is closed the speed is set to 0.3, otherwise you can change it as you wish

            if self.brand in ["fei", "fei_temspy"]:
                self.speed_values = ["1", "0.7", "0.3", "0.066642775", "0.025674144"] #these are strings that need to be float later
                speed_text = "speed gonio for general movements (fei a.u.):"
            elif self.brand in ["jeol"]:
                self.speed_values = ["8.371", "8.298", "8.1926", "7.3747", "6.6557", "5.7389",
                                     "4.921", "4.1031", "3.2852", "2.4673", "1.6494", "0.8315"] # these are in deg/s
                speed_text = "speed gonio for general movements (deg/s):"


            self.speed_label = tk.Label(self.new_window, text=speed_text).grid(row=21, column=1, columnspan=1, padx=5, pady=5, sticky="w")
            self.speed_combobox = ttk.Combobox(self.new_window, values=self.speed_values)
            self.speed_combobox.grid(row=23, column=1, columnspan=2, padx=5, pady=5, sticky="w")
            if self.brand in ["fei", "fei_temspy"]:
                self.speed_combobox.current(2)  # set default to 10 deg/s speed gonio for general movements #FEI
            elif self.brand in ["jeol"]:
                self.speed_combobox.current(2) # set default to 8.1926 deg/s speed gonio for general movements #JEOL






    #functions to get the widgets values correctly typecasted
    def seq_value(self):
        return self.seq_var.get()

    def cont_value(self):
        return self.cont_var.get()

    def tem_value(self):
        return self.tem_var.get()

    def stem_value(self):
        return self.stem_var.get()

    def stem_pixeltime_value(self):
        return float(self.stem_pixeltime_entry.get())

    def tem_imagetime_value(self):
        return int(float(self.tem_imagetime_entry.get())*1000)

    def angle_value(self):
        return float(self.angle_entry.get())

    def final_angle_value(self):
        return float(self.final_angle_entry.get())

    def tilt_step_value(self):
        return float(self.tilt_step_entry.get())

    def go_to_value(self):
        return float(self.go_to_entry.get()) #deg

    def exposure_value(self):
        return int(float(self.exposure_entry.get())*1000)

    def binning_value(self):
        return int(self.binning_combobox.get())

    def processing_value(self):
        return self.processing_combobox.get()

    def beamblank_value(self):
        return self.beamblank_checkbox_var.get()

    # def beam_shift_value(self):
    #     return self.beam_shift_var.get()
    def tracking_step_value(self):
        return self.tracking_step_var.get()
    def get_tracking_method(self):
        return self.method_combobox.get()
    def get_KL_value(self):
        return str(self.KL_combobox.get())
    def get_mag_value(self):
        return str(self.mag_combobox.get())
    def get_dir_value(self):
        return str(self.var_dir.get())
    def get_sample_value(self):
        return str(self.var_sample.get())
    def get_exp_num_value(self):
        return int(self.var_exp_num.get())
    def get_diff_lens_value(self):
        return float(self.diff_lens_entry.get())
    def get_stem_image_size_value(self):
        return str(self.image_size_combobox.get())
    def get_stem_binning_value(self):
        return int(self.stem_binning_combobox.get())

    def get_a_backlash_correction_value(self): # this is alpha
        return self.a_backlash_var.get()
    def get_x_backlash_correction_value(self):
        return self.x_backlash_var.get()
    def get_y_backlash_correction_value(self):
        return self.y_backlash_var.get()
    def get_z_backlash_correction_value(self):
        return self.z_backlash_var.get()
    def get_init_position_value(self):
        return self.init_position_var.get()
    def get_high_performance_value(self):
        return self.hiper_var.get()
    def get_speed_tracking(self):
        return float(self.speed_combobox.get())
    ############################################### values
    def get_work_dir(self):
        date = datetime.datetime.now().strftime("%d_%m_%Y")
        if self.brand == "power_user":
            return "empty"
        else:
            return self.cam_table["default_dir"] + os.sep + date
    def browse_dir(self):
        try:
            dir_ = askdirectory(parent=self, title='Select working directory')
        except:
            return
        date = datetime.datetime.now().strftime("%d_%m_%Y")
        if os.path.split(dir_)[1] != date:
            dir_ = dir_ + os.sep + date

        self.var_dir.set(dir_)
        self.update_experiment_number()
        return dir_
    def open_dir(self):
        dir_ = self.get_dir_value()
        try:
            os.startfile(dir_)
        except FileNotFoundError:
            print("directory not found")
    def exp_name(self, full_path = False):
        if full_path == False:
            return str(self.get_sample_value()) + "_" + str(self.get_exp_num_value())
        else:
            return self.get_dir_value() + os.sep + str(self.get_sample_value()) + "_" + str(self.get_exp_num_value())

    def delete_last(self):
        old_dir = self.exp_name(full_path=True)
        date = datetime.datetime.now().strftime('_%H_%M')
        new_dir = self.get_dir_value() + os.sep + "to_delete_"+str(self.exp_name()+date)
        if os.path.exists(old_dir) == True:
            os.replace(old_dir, new_dir)
            print('Marked %s to delete it' %new_dir)
        else:
            print('folder does not exist')
        self.update_experiment_number()

    def update_experiment_number(self):
        name = self.get_dir_value() + os.sep + self.get_sample_value() + "_"
        number = 1

        while os.path.exists(name+str(number)):
            number += 1
        self.var_exp_num.set(number)
        return number

    def widget_status(self):
        if self.brand in ["power_user"]:
            ########## add here the calculator for the expected time of acquisition #######
            try:
                if self.cont_value() == True:
                    # tilt_step(deg/img)/exposure(s) = FPS (deg/s)
                    # final - initial = total angle (deg)
                    # total angle(deg) / FPS(deg/s) = time (s)
                    fps = np.round(self.tilt_step_value() / (self.exposure_value() / 1000), 2)
                    total_angle = abs(self.final_angle_value() - self.angle_value()) ############# wrong calculation
                    self.expected_time_value = str(np.round(total_angle / fps, 2))
                    self.expected_speed_value = str(fps)
                else:
                    tilt_step = self.tilt_step_value()
                    exposure = (self.exposure_value() / 1000)
                    total_angle = abs(self.final_angle_value() - self.angle_value())

                    self.expected_time_value = str(np.round((total_angle / tilt_step) * exposure, 2))
                    self.expected_speed_value = str(np.round(self.speed_tracking, 4))

            except Exception as err:

                self.expected_time_value = "???"
                self.expected_speed_value = "???"

            self.expected_time_label2.config(text="Exp. time: " + self.expected_time_value + " s,\tgonio velocity: " + self.expected_speed_value + " °/s")
            ###############################################################################

            self.after(200, self.widget_status)  # call the function again after 100ms
            return

        if self.camera in ["timepix1", "merlin"]:
            self.processing_combobox.current(2) # set unprocessed
            self.processing_combobox.config(state=tk.DISABLED)
            self.binning_combobox.current(0) # set to binning 1
            self.binning_combobox.config(state=tk.DISABLED)

        #sequential vs continous checkbox auto disabling
        if self.seq_value() == True:
            self.cont_check.config(state=tk.DISABLED)
        else:
            self.cont_check.config(state=tk.NORMAL)

        if self.cont_value() == True:
            self.seq_check.config(state=tk.DISABLED)
        else:
            self.seq_check.config(state=tk.NORMAL)
        # auto check for tem or stem mode
        if self.tem.get_instrument_mode() == "STEM":
            self.stem_var.set(True)
            self.tem_var.set(False)

        else:
            self.stem_var.set(False)
            self.tem_var.set(True)

        self.tem_hide()

        # tem vs stem checkbox auto disabling
        if self.tem_value() == True:
            self.stem_check.config(state=tk.DISABLED)
            self.diff_lens_entry.config(state=tk.DISABLED)
            self.stem_pixeltime_entry.config(state=tk.DISABLED)
            self.tem_imagetime_entry.config(state=tk.NORMAL)
        else:
            self.stem_check.config(state=tk.NORMAL)
            self.diff_lens_entry.config(state=tk.NORMAL)
            self.stem_pixeltime_entry.config(state=tk.NORMAL)
            self.tem_imagetime_entry.config(state=tk.DISABLED)

        if self.stem_value() == True:
            self.tem_check.config(state=tk.DISABLED)
        else:
            self.tem_check.config(state=tk.NORMAL)

        ########## add here the calculator for the expected time of acquisition #######
        try:
            if self.cont_value() == True:
                # tilt_step(deg/img)/exposure(s) = FPS (deg/s)
                # final - initial = total angle (deg)
                # total angle(deg) / FPS(deg/s) = time (s)
                fps = np.round(self.tilt_step_value() / (self.exposure_value() / 1000), 2)
                total_angle = abs(self.final_angle_value() - self.angle_value())
                self.expected_time_value = str(np.round(total_angle / fps, 2))
                self.expected_speed_value = str(fps)
            else:
                tilt_step = self.tilt_step_value()
                exposure = (self.exposure_value() / 1000)
                total_angle = abs(self.final_angle_value() - self.angle_value())
                num_images = total_angle/tilt_step
                if self.brand in ["fei", "fei_temspy"]:
                    if self.speed_tracking == 1:
                        speed = 40 #deg/s
                    elif self.speed_tracking == 0.7:
                        speed = 20  # deg/s
                    elif self.speed_tracking == 0.3:
                        speed = 10  # deg/s
                    elif self.speed_tracking == 0.066:
                        speed = 2  # deg/s
                    elif self.speed_tracking == 0.025:
                        speed = 1  # deg/s
                else: # this will be the case for jeol whihc is already in deg/s
                    speed = self.speed_tracking
                self.expected_time_value = str(np.round(((num_images)*exposure)+(num_images*(self.tilt_step_value()/speed)), 2))
                self.expected_speed_value = str(np.round(speed, 4))

        except Exception as err:
            self.expected_time_value = "???"
            self.expected_speed_value = "???"

        self.expected_time_label2.config(text="Exp. time: " + self.expected_time_value + " s,\tgonio velocity: " + self.expected_speed_value + " °/s")
        ###############################################################################

        # update speed_tracking dynamically:
        try:
            self.speed_tracking = self.get_speed_tracking()
        except:
            self.speed_tracking = 0.3

            ## this variable is added to reduce the speed of the goniometer during normal movements
            # self.speed_tracking = 1
            # self.speed_tracking = 0.7                                                                                                                    ##################changed these stuff
            # self.speed_tracking = 0.3
            # self.speed_tracking = 0.066642775
            # self.speed_tracking = 0.025674144

        self.after(200, self.widget_status)  # call the function again after 100ms

    def abort_data_acquisition(self):
        try:
            #self.tem.thread_beam.terminate()
            #self.tem.thread_stage.terminate()
            self.tem.beam_blank(True)
            if self.tem.get_screen_position() != 'DOWN':
                if self.cam.is_cam_bottom_mounted():
                    self.tem.move_screen(False)
        except:
            pass

    def live_view_gui(self):
        if self.cam.is_cam_streaming:
            self.server = self.cam.enable_streaming()
            time.sleep(1)
            from client_sharedmemory_stream import camera_client
            from multiprocessing import Process
            import atexit
            p1 = Process(target=camera_client, args=(self.server.shared_mem_name, self.server.frame_shape))
            atexit.register(p1.terminate)
            p1.start()
        else:
            pass






if __name__ == '__main__':
    root = tk.Tk(baseName="fastADT")
    app = FastADT(master = root)
    app.mainloop()


