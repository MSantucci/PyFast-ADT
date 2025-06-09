# handpanel for digital micrograph in tkinter for FEI tecnai 
# the script generate a tkinter window with the basic commands used in the handpanel of a tecnai microscope, necessary to handle 3ded experiments
try:
    import temscript
except Exception as err:
    print('import failed:', err)
try:
    import DigitalMicrograph as DM
    DM.ClearResults()
    #ciao questo l'ho modificato
except Exception as err:
    print('import failed:', err)
import tkinter as tk
from tkinter import ttk
import os
import sys
import numpy as np
import time
print(sys.path)

dir_path = r"/python_FEI_FastADT/07042023_build_up/main"
if dir_path in sys.path:
    print('dir already there')
else:
    sys.path.append(dir_path)

class HandPanel(tk.Toplevel):
    def __init__(self, master=None, brand = 'fei', camera = 'timepix1'):
        super().__init__(master)
        # camera initialization
        self.camera = camera
        if self.camera == 'timepix1':
            from adaptor.camera.adaptor_timepix1 import Cam_timepix1
            self.cam = Cam_timepix1(_id=0)
        elif self.camera == 'xf416r':
            from adaptor.camera.adaptor_xf416r_GPU import Cam_xf416r
            self.cam = Cam_xf416r()
        elif self.camera == 'us4000':
            from adaptor.camera.adaptor_us4000 import Cam_us4000
            self.cam = Cam_us4000()
        elif self.camera == 'us2000':
            from adaptor.camera.adaptor_us2000 import Cam_us2000
            self.cam = Cam_us2000()
        elif self.camera == 'merlin':
            from adaptor.camera.adaptor_merlin import Cam_merlin
            self.cam = Cam_merlin()
        elif self.camera == 'ceta':
            from adaptor.camera.adaptor_ceta import Cam_ceta
            self.cam = Cam_ceta(instance_gui = self)
        elif self.camera == 'medipix3':
            from adaptor.camera.adaptor_serval import Cam_medipix3
            self.cam = Cam_medipix3(instance_gui = self)
        else:
            print('camera not supported or recognized, exit')
            return
        self.cam_table = self.cam.load_calibration_table()

        self.brand = brand
        if self.brand == 'fei':
            from adaptor.microscope.adaptor_fei import Tem_fei
            self.tem = Tem_fei(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table, master = self)
        elif self.brand == 'jeol':
            from adaptor.microscope.adaptor_jeol import Tem_jeol
            self.tem = Tem_jeol(cam_table=self.cam_table, master = self)
        elif self.brand == 'gatan_fei':
            from adaptor.microscope.adaptor_gatan_fei import Tem_gatan_fei
            self.tem = Tem_gatan_fei(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table, master = self)
        elif self.brand == 'gatan_jeol':
            from adaptor.microscope.adaptor_gatan_jeol import Tem_gatan_jeol
            self.tem = Tem_gatan_jeol(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table, master = self)
        elif self.brand == 'fei_temspy':
            from adaptor.microscope.adaptor_fei_temspy import Tem_fei_temspy
            self.tem = Tem_fei_temspy(ip=self.cam_table["ip"][0], port=self.cam_table["ip"][1], cam_table=self.cam_table, master = self)
        else:
            print('brand not supported or recognized, exit')
            return

        # self.speed_tracking = 1
        # self.speed_tracking = 0.7                                                                                                                    ##################changed these stuff
        self.speed_tracking = 0.3
        # self.speed_tracking = 0.066642775
        # self.speed_tracking = 0.025674144

        self.zero_pos_row = 0
        self.zero_pos_col = 1
        self.master = master
        self.grid()
        self.create_widgets()
        self.title("Hand Panels")
        self.geometry("1000x230")

    def create_widgets(self):
        ############# stage section #############
        # add separator
        self.separator1 = tk.Frame(self, height=10, width=10)
        self.separator1.grid(row=self.zero_pos_row, column=self.zero_pos_col, rowspan=4)
        
        # Create buttons Stage
        self.button_width = 5
        self.button_height = 2
        self.stage_y_up = tk.Button(self, text="Y+", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_stage_up(self.stage_ampl_value()))
        self.stage_y_down = tk.Button(self, text="Y-", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_stage_down(self.stage_ampl_value()))
        self.stage_x_left = tk.Button(self, text="X-", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_stage_left(self.stage_ampl_value()))
        self.stage_x_right = tk.Button(self, text="X+", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_stage_right(self.stage_ampl_value()))
        
        self.stage_z_up = tk.Button(self, text="Z+", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_stage_z_up(self.stage_ampl_value()))
        self.stage_z_down = tk.Button(self, text="Z-", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_stage_z_down(self.stage_ampl_value()))
        # add label
        self.stage_label = tk.Label(self, text = 'Stage Ctrl\n').grid(row = self.zero_pos_row, column = self.zero_pos_col+2, columnspan=2, sticky='w')
        # Add buttons to GUI in a cross shape
        self.stage_y_up.grid(row=self.zero_pos_row+1, column=self.zero_pos_col+2)
        self.stage_y_down.grid(row=self.zero_pos_row+3, column=self.zero_pos_col+2)
        self.stage_x_left.grid(row=self.zero_pos_row+2, column=self.zero_pos_col+1)
        self.stage_x_right.grid(row=self.zero_pos_row+2, column=self.zero_pos_col+3)

        self.stage_z_up.grid(row=self.zero_pos_row+1, column=self.zero_pos_col-1)
        self.stage_z_down.grid(row=self.zero_pos_row+2, column=self.zero_pos_col-1)

        # add spinbox for amplitude of the movement
        self.stage_amplitude_var = tk.DoubleVar()
        self.stage_amplitude_var.set(0.1)
        self.stage_amplitude_spinbox = tk.Spinbox(self, from_=0, to=1, increment=0.01, textvariable=self.stage_amplitude_var, width = 5)
        self.stage_amplitude_spinbox.grid(row=self.zero_pos_row+1, column=self.zero_pos_col+3)


        ############# beam shift section #############
        # add separator 
        self.separator2 = tk.Frame(self, height=10, width=40)
        self.separator2.grid(row=self.zero_pos_row, column=self.zero_pos_col+4, rowspan=4)
        
        # Create buttons BeamShift
        self.zero_pos_row = 0
        self.zero_pos_col = 6
        self.beam_up = tk.Button(self, text="Up", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_beam_up(self.beam_ampl_value()))
        self.beam_down = tk.Button(self, text="Down", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_beam_down(self.beam_ampl_value()))
        self.beam_left = tk.Button(self, text="Left", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_beam_left(self.beam_ampl_value()))
        self.beam_right = tk.Button(self, text="Right", width = self.button_width, height = self.button_height, command=lambda: self.tem.move_beam_right(self.beam_ampl_value()))
        # add label
        self.beam_label = tk.Label(self, text = 'Beam Shift Ctrl\n').grid(row = self.zero_pos_row, column = self.zero_pos_col+2, columnspan=2, sticky='w')
        # Add buttons to GUI in a cross shape
        self.beam_up.grid(row=self.zero_pos_row+1, column=self.zero_pos_col+2)
        self.beam_down.grid(row=self.zero_pos_row+3, column=self.zero_pos_col+2)
        self.beam_left.grid(row=self.zero_pos_row+2, column=self.zero_pos_col+1)
        self.beam_right.grid(row=self.zero_pos_row+2, column=self.zero_pos_col+3)

        # add spinbox for amplitude of the movement
        self.beam_amplitude_var = tk.DoubleVar()
        self.beam_amplitude_var.set(0.1)
        self.beam_amplitude_spinbox = tk.Spinbox(self, from_=0, to=1, increment=0.01, textvariable=self.beam_amplitude_var, width=5)
        self.beam_amplitude_spinbox.grid(row=self.zero_pos_row + 1, column=self.zero_pos_col + 3)
        self.zero_pos_col = 0
        ############# toogle section #############
        # add separator 
        self.separator3 = tk.Frame(self, height=10, width=40)
        self.separator3.grid(row=self.zero_pos_row, column=self.zero_pos_col+10, rowspan=4)
        
        # Create buttons BeamShift
        self.zero_pos_row = 0
        self.zero_pos_col = 12

        self.checked_diff = tk.BooleanVar()
        self.checked_wobbler = tk.BooleanVar()
        self.checked_screen = tk.BooleanVar()
        self.checked_blank = tk.BooleanVar()

        self.toogle_diff = tk.Checkbutton(self, text="Diffraction", command=lambda: self.tem.diffraction(self.checked_diff_value(), kl = self.kl_var.get()), variable = self.checked_diff)
        self.toogle_euc_focus = tk.Button(self, text="Euc. Focus", command=self.tem.euc_focus)
        self.toogle_wobbler = tk.Checkbutton(self, text="Wobbler", command=lambda: self.tem.wobbler(self.checked_wobbler_value()), variable = self.checked_wobbler)
        self.toogle_screen = tk.Checkbutton(self, text="Screen lift", command=lambda: self.tem.move_screen(self.checked_screen_value()), variable = self.checked_screen)
        self.toogle_blank = tk.Checkbutton(self, text="Beam Blank", command=lambda: self.tem.beam_blank(self.checked_blank_value()), variable=self.checked_blank)

        # add label
        self.toogle_label = tk.Label(self, text = 'toogles\n').grid(row = self.zero_pos_row, column = self.zero_pos_col)

        # Add checkbuttons to GUI in a cross shape
        self.toogle_diff.grid(row=self.zero_pos_row+1, column=self.zero_pos_col, sticky = 'w')
        self.toogle_euc_focus.grid(row=self.zero_pos_row+2, column=self.zero_pos_col, sticky = 'w')
        self.toogle_wobbler.grid(row=self.zero_pos_row+3, column=self.zero_pos_col, sticky = 'w')
        self.toogle_screen.grid(row=self.zero_pos_row+4, column=self.zero_pos_col, sticky = 'w')
        self.toogle_blank.grid(row=self.zero_pos_row + 5, column=self.zero_pos_col, sticky='w')
        self.zero_pos_col = 0

        ############# projector section #############
        # add spinbox for the magnification and kl
        # add separator
        self.separator4 = tk.Frame(self, height=10, width=40)
        self.separator4.grid(row=self.zero_pos_row, column=self.zero_pos_col + 15, rowspan=4)

        # Create buttons BeamShift
        self.zero_pos_row = 0
        self.zero_pos_col = 17

        self.mag_var = tk.StringVar(value = '2400')
        self.kl_var = tk.StringVar(value = '1000')

        self.mags = ttk.Combobox(self, textvariable=self.mag_var, state = 'readonly')
        self.mags['values'] = list(self.tem.mag_index_table.keys())
        self.mags.bind('<<ComboboxSelected>>', func = lambda event: self.tem.set_magnification(self.tem.mag_index_table[self.mag_var.get()][0]))

        self.kls = ttk.Combobox(self, textvariable=self.kl_var, state = 'readonly')
        self.kls['values'] = list(self.tem.kl_index_table.keys())
        self.kls.bind('<<ComboboxSelected>>', func=lambda event: self.tem.set_KL(self.tem.kl_index_table[self.kl_var.get()][0]))
        self.kls["state"]= tk.DISABLED

        # add label
        self.proj_label = tk.Label(self, text='projector\n').grid(row=self.zero_pos_row, column=self.zero_pos_col)
        self.mag_label = tk.Label(self, text='magnification\n').grid(row=self.zero_pos_row+1, column=self.zero_pos_col)
        self.kl_label = tk.Label(self, text='camera length\n').grid(row=self.zero_pos_row+3, column=self.zero_pos_col)
        # Add checkbuttons to GUI in a cross shape
        self.mags.grid(row=self.zero_pos_row + 2, column=self.zero_pos_col, sticky='w')
        self.kls.grid(row=self.zero_pos_row + 4, column=self.zero_pos_col, sticky='w')

        self.zero_pos_col = 0

        ############# beam size for imaging and for diffraction section #############
        # add separator
        self.separator5 = tk.Frame(self, height=10, width=40)
        self.separator5.grid(row=self.zero_pos_row, column=self.zero_pos_col + 19, rowspan=4)

        # Create buttons BeamShift
        self.zero_pos_row = 0
        self.zero_pos_col = 21

        self.set_beam_image = tk.Button(self, text="set beam img", width=self.button_width+10, height=self.button_height, command=lambda: self.tem.set_intensity(slot=1))
        self.get_beam_image = tk.Button(self, text="store", width=self.button_width, height=self.button_height, command=lambda: self.tem.get_intensity(slot=1))
        self.set_beam_diff = tk.Button(self, text="set beam diff", width=self.button_width+10, height=self.button_height, command=lambda: self.tem.set_intensity(slot=2))
        self.get_beam_diff = tk.Button(self, text="store", width=self.button_width, height=self.button_height, command=lambda: self.tem.get_intensity(slot=2))

        # add label
        self.beam_label1 = tk.Label(self, text='Beam size (C2%)\n ').grid(row=self.zero_pos_row, column=self.zero_pos_col + 1, columnspan=2, sticky='w')
        # Add buttons to GUI in a cross shape
        self.set_beam_image.grid(row=self.zero_pos_row + 1, column=self.zero_pos_col + 1)
        self.get_beam_image.grid(row=self.zero_pos_row + 1, column=self.zero_pos_col + 2)
        self.set_beam_diff.grid(row=self.zero_pos_row + 2, column=self.zero_pos_col + 1, pady= 5)
        self.get_beam_diff.grid(row=self.zero_pos_row + 2, column=self.zero_pos_col + 2, pady= 5)

        #add intensity up and down
        # add spinbox for amplitude of the intensity
        self.intensity_ampl_var = tk.DoubleVar()
        self.intensity_ampl_var.set(0.02)
        self.intensity_amplitude_spinbox = tk.Spinbox(self, from_=0, to=1, increment=0.01, textvariable=self.intensity_ampl_var, width=15)
        self.intensity_amplitude_spinbox.grid(row=self.zero_pos_row + 4, column=self.zero_pos_col + 1, columnspan=2)

        self.intensity_up = tk.Button(self, text="int +", width=self.button_width, height=self.button_height, command=lambda: self.tem.set_intensity(intensity = self.tem.get_intensity() + self.intensity_ampl_value(), slot = 0))
        self.intensity_down = tk.Button(self, text="int -", width=self.button_width, height=self.button_height, command=lambda: self.tem.set_intensity(intensity = self.tem.get_intensity() - self.intensity_ampl_value(), slot = 0))
        # add label
        self.intensity_up.grid(row=self.zero_pos_row + 3, column=self.zero_pos_col + 1)
        self.intensity_down.grid(row=self.zero_pos_row + 3, column=self.zero_pos_col + 2)

        self.zero_pos_col = 0

        if self.brand in ['fei_temspy', 'fei']:
            self.generate_bot_frame()

        self.after(300, self.widget_status)

    def widget_status(self):
        if self.tem.tem.get_instrument_mode() == "STEM":
            self.beam_up = tk.Button(self, text="Up", width=self.button_width, height=self.button_height,
                                     command=lambda: self.tem.move_stem_beam_up(self.beam_ampl_value()))
            self.beam_down = tk.Button(self, text="Down", width=self.button_width, height=self.button_height,
                                       command=lambda: self.tem.move_stem_beam_down(self.beam_ampl_value()))
            self.beam_left = tk.Button(self, text="Left", width=self.button_width, height=self.button_height,
                                       command=lambda: self.tem.move_stem_beam_left(self.beam_ampl_value()))
            self.beam_right = tk.Button(self, text="Right", width=self.button_width, height=self.button_height,
                                        command=lambda: self.tem.move_stem_beam_right(self.beam_ampl_value()))

            # self.set_beam_image = tk.Button(self, text="set beam img", width=self.button_width + 10, height=self.button_height, command=lambda: self.tem.set_intensity(slot=1))
            # self.get_beam_image = tk.Button(self, text="store", width=self.button_width, height=self.button_height, command=lambda: self.tem.get_intensity(slot=1))
            # self.set_beam_diff = tk.Button(self, text="set beam diff", width=self.button_width + 10, height=self.button_height, command=lambda: self.tem.set_intensity(slot=2))
            # self.get_beam_diff = tk.Button(self, text="store", width=self.button_width, height=self.button_height, command=lambda: self.tem.get_intensity(slot=2))
            # self.intensity_up = tk.Button(self, text="int +", width=self.button_width, height=self.button_height, command=lambda: self.tem.set_intensity(intensity=self.tem.get_intensity() + self.intensity_ampl_value(), slot=0))
            # self.intensity_down = tk.Button(self, text="int -", width=self.button_width, height=self.button_height, command=lambda: self.tem.set_intensity(intensity=self.tem.get_intensity() - self.intensity_ampl_value(), slot=0))

        else:
            self.beam_up = tk.Button(self, text="Up", width=self.button_width, height=self.button_height,
                                     command=lambda: self.tem.move_beam_up(self.beam_ampl_value()))
            self.beam_down = tk.Button(self, text="Down", width=self.button_width, height=self.button_height,
                                       command=lambda: self.tem.move_beam_down(self.beam_ampl_value()))
            self.beam_left = tk.Button(self, text="Left", width=self.button_width, height=self.button_height,
                                       command=lambda: self.tem.move_beam_left(self.beam_ampl_value()))
            self.beam_right = tk.Button(self, text="Right", width=self.button_width, height=self.button_height,
                                        command=lambda: self.tem.move_beam_right(self.beam_ampl_value()))
            # self.set_beam_image = tk.Button(self, text="set beam img", width=self.button_width + 10, height=self.button_height, command=lambda: self.tem.set_intensity(slot=1))
            # self.get_beam_image = tk.Button(self, text="store", width=self.button_width, height=self.button_height, command=lambda: self.tem.get_intensity(slot=1))
            # self.set_beam_diff = tk.Button(self, text="set beam diff", width=self.button_width + 10, height=self.button_height, command=lambda: self.tem.set_intensity(slot=2))
            # self.get_beam_diff = tk.Button(self, text="store", width=self.button_width, height=self.button_height, command=lambda: self.tem.get_intensity(slot=2))
            # self.intensity_up = tk.Button(self, text="int +", width=self.button_width, height=self.button_height, command=lambda: self.tem.set_intensity(intensity=self.tem.get_intensity() + self.intensity_ampl_value(), slot=0))
            # self.intensity_down = tk.Button(self, text="int -", width=self.button_width, height=self.button_height, command=lambda: self.tem.set_intensity(intensity=self.tem.get_intensity() - self.intensity_ampl_value(), slot=0))

        # Create buttons BeamShift
        self.zero_pos_row = 0
        self.zero_pos_col = 6
        self.beam_up.grid(row=self.zero_pos_row + 1, column=self.zero_pos_col + 2)
        self.beam_down.grid(row=self.zero_pos_row + 3, column=self.zero_pos_col + 2)
        self.beam_left.grid(row=self.zero_pos_row + 2, column=self.zero_pos_col + 1)
        self.beam_right.grid(row=self.zero_pos_row + 2, column=self.zero_pos_col + 3)
        # self.zero_pos_col = 21
        #
        # self.intensity_up.grid(row=self.zero_pos_row + 3, column=self.zero_pos_col + 1)
        # self.intensity_down.grid(row=self.zero_pos_row + 3, column=self.zero_pos_col + 2)
        # self.set_beam_image.grid(row=self.zero_pos_row + 1, column=self.zero_pos_col + 1)
        # self.get_beam_image.grid(row=self.zero_pos_row + 1, column=self.zero_pos_col + 2)
        # self.set_beam_diff.grid(row=self.zero_pos_row + 2, column=self.zero_pos_col + 1, pady=5)
        # self.get_beam_diff.grid(row=self.zero_pos_row + 2, column=self.zero_pos_col + 2, pady=5)
        self.zero_pos_col = 0


    def stage_ampl_value(self):
        return self.stage_amplitude_var.get()

    def beam_ampl_value(self):
        return self.beam_amplitude_var.get()

    def checked_diff_value(self):
        diff = self.checked_diff.get()
        if diff:
            self.kls["state"] = tk.NORMAL
        else:
            self.kls["state"] = tk.DISABLED
        return diff

    def checked_wobbler_value(self):
        return self.checked_wobbler.get()

    def checked_screen_value(self):
        return self.checked_screen.get()

    def checked_blank_value(self):
        return self.checked_blank.get()

    def intensity_ampl_value(self):
        return self.intensity_ampl_var.get()

    def manual_backlash_tui(self):
        shift_movement = 5
        self.tem.set_alpha(0, velocity = 0.3) # go to 0 deg to perform the backlash correction
        time.sleep(1)
        pos = self.tem.get_stage()
        axes = {"x": pos["x"], "y": pos["y"],"z": pos["z"]}

        for axis in axes:
            choosen_pos = pos[axis]
            sign_pos = np.sign(choosen_pos)
            if sign_pos == 0:  # set as positive sign if the coordinate is exactly 0
                sign_pos = 1

            if self.brand in ["fei", "fei_temspy"]:
                print("starting backlash correction for %s axis" % str(axis))
                self.tem.set_xyz_tui(**{axis: choosen_pos - (sign_pos * shift_movement)})
                time.sleep(1)
                self.tem.set_xyz_tui(**{axis: choosen_pos})
                time.sleep(1)

            else:
                print("starting backlash correction for %s axis" % str(axis))
                self.tem.set_stage_position(**{axis: choosen_pos - (sign_pos * shift_movement)})
                time.sleep(1)
                self.tem.set_stage_position(**{axis: choosen_pos})
                time.sleep(1)




    def generate_bot_frame(self):
        # Create another new window
        new_window = tk.Toplevel(self)
        new_window.title("Bot Tab")
        new_window.geometry("200x250")
        self.bot_1 = tk.Button(new_window, text="HAADF position", command=lambda: self.tem.client.client_send_action({"check_HAADF_position": 0}))
        self.bot_2 = tk.Button(new_window, text="click HAADF", command=lambda: self.tem.client.client_send_action({"click_HAADF": 0}))
        self.bot_3 = tk.Button(new_window, text="diff into imag", command=lambda: self.tem.client.client_send_action({"diff_into_imag": 0}))
        self.bot_4 = tk.Button(new_window, text="imag into diff", command=lambda: self.tem.client.client_send_action({"image_into_diff": 0.333333}))
        self.bot_5 = tk.Button(new_window, text="manual backlash tui", command=lambda: self.manual_backlash_tui())
        self.bot_1.pack()
        self.bot_2.pack()
        self.bot_3.pack()
        self.bot_4.pack()
        self.bot_5.pack()

if __name__ == '__main__':
    root = tk.Tk(baseName = "hand")
    root.title('hand panel Fast ADT')
    root.geometry("1000x230")
    app = HandPanel(master=root)
    app.mainloop()


print('ending code')