import sys
sys.path.append(r'C:\PyFast-ADT\main')
import time
try:
    from .adaptor_tem import Tem_base
except:
    from adaptor_tem import Tem_base
import numpy as np
import json
import os
import threading
import math
try:
    from fast_adt_func import read_tracking_file
except:
    prev_path = os.getcwd()
    os.chdir(r'C:\PyFast-ADT\main')
    from fast_adt_func import read_tracking_file
    os.chdir(prev_path)
from scipy.interpolate import interp1d
import pandas as pd
import pythoncom
import win32com.client
try:
    import comtypes
    import comtypes.client
except:
    pass
import atexit

class Tem_jeol(Tem_base): # this is self.tem in FAST-ADT_GUI.py
    """every angle for moving the satge must be in deg as input and output, velocity for the stage in radian/s and um for the stage xyz movement"""
    def __init__(self, cam_table = None, master = None): # removed ip and port as optional parameter
        super().__init__()
        self.tem = None
        self.connect()
        if cam_table != None:
            self.load_calibration_table(cam_table)
        else: self.cam_table = None
        self.result = []
        self.master = master
        self.calibrated_speed = None
        self.FUNCTION_MODES = ('mag1', 'mag2', 'lowmag', 'samag', 'diff')
    # stage movements
    def move_stage_up(self, stage_ampl):
        """stage ampl is in um and need to be converted to m to work in FEI/ThermoFisher (from API FEI works in m).
        before was set at -90 as fei, but is needed another 90 deg clockwise to be sincronized properly that's why now
        is -180 if it doesn't work try 0 deg """
        print("I'm going stage up %s!" % str(stage_ampl))
        stage_pos = self.get_stage() # x,y,z um a, b in deg
        # print(stage_pos)
        shift = (0, stage_ampl) # in um
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["stage_rotation"])
        shift_rot_x = stage_pos['x'] + shift_rot_x # sum up um + um
        shift_rot_y = stage_pos['y'] + shift_rot_y
        # print(shift)
        self.set_stage_position(x = shift_rot_x, y = shift_rot_y)

    def move_stage_down(self, stage_ampl):
        print("I'm going stage down %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        shift = (0, stage_ampl)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["stage_rotation"])
        shift_rot_x = stage_pos['x'] - shift_rot_x
        shift_rot_y = stage_pos['y'] - shift_rot_y
        # print(shift)
        self.set_stage_position(x = shift_rot_x, y = shift_rot_y)

    def move_stage_left(self, stage_ampl):
        print("I'm going stage left %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        shift = (stage_ampl, 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["stage_rotation"])
        shift_rot_x = stage_pos['x'] - shift_rot_x
        shift_rot_y = stage_pos['y'] - shift_rot_y
        # print(shift)
        self.set_stage_position(x = shift_rot_x, y = shift_rot_y)

    def move_stage_right(self, stage_ampl):
        print("I'm going stage right %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        shift = (stage_ampl, 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["stage_rotation"])
        shift_rot_x = stage_pos['x'] + shift_rot_x
        shift_rot_y = stage_pos['y'] + shift_rot_y
        # print(shift)
        self.set_stage_position(x = shift_rot_x, y = shift_rot_y)

    def move_stage_z_up(self, stage_ampl):
        print("I'm going stage Z up %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        # print(stage_pos)
        shift = stage_pos['z'] + stage_ampl
        # print(shift)
        self.set_stage_position(z = shift)

    def move_stage_z_down(self, stage_ampl):
        print("I'm going stage Z down %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        # print(stage_pos)
        shift = stage_pos['z'] - stage_ampl
        # print(shift)
        self.set_stage_position(z = shift)

    # beamshift movements
    def move_beam_up(self, beam_ampl):
        print("I'm going beam up %s!" % str(beam_ampl))
        beam_pos = self.get_beam_shift()
        shift = (0, beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["TEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.def3.SetCLA1(beam_pos[0], beam_pos[1]) # we must check if this is the real beam shift! and if float is ok!

    def move_beam_down(self, beam_ampl):
        print("I'm going beam down %s!" % str(beam_ampl))
        beam_pos = self.get_beam_shift()
        shift = (0, - beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["TEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.def3.SetCLA1(beam_pos[0], beam_pos[1])

    def move_beam_left(self, beam_ampl):
        print("I'm going beam left %s!" % str(beam_ampl))
        beam_pos = self.get_beam_shift()
        shift = (- beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["TEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.def3.SetCLA1(beam_pos[0], beam_pos[1])

    def move_beam_right(self, beam_ampl):
        print("I'm going beam right %s!" % str(beam_ampl))
        beam_pos = self.get_beam_shift()
        shift = (beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["TEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.def3.SetCLA1(beam_pos[0], beam_pos[1])

    # beamshift movements in stem
    def move_stem_beam_up(self, beam_ampl):
        print("I'm going beam up %s!" % str(beam_ampl))
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        shift = (0, beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["STEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        #self.client.client_send_action({"set_stem_beam": beam_pos})
        pass

    def move_stem_beam_down(self, beam_ampl):
        print("I'm going beam down %s!" % str(beam_ampl))
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        shift = (0, - beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["STEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        #self.client.client_send_action({"set_stem_beam": beam_pos})
        pass

    def move_stem_beam_left(self, beam_ampl):
        print("I'm going beam left %s!" % str(beam_ampl))
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        shift = (- beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["STEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        #self.client.client_send_action({"set_stem_beam": beam_pos})
        pass

    def move_stem_beam_right(self, beam_ampl):
        print("I'm going beam right %s!" % str(beam_ampl))
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        shift = (beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["STEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        #self.client.client_send_action({"set_stem_beam": beam_pos})
        pass

    def getFunctionMode(self): ######################### function to test, check out how to select different modes
        """return one of these strings: Mag1, mag2, lowmag, samag, diff."""
        mode, name, result = self.eos3.GetFunctionMode()
        print("check line 161 adaptor_jeol, mode", mode, "name", name, "result", result)
        return self.FUNCTION_MODES[mode]

    def setFunctionMode(self, value):
        """you can input in value both a string or an integer of the corresponding mode to change
        Mag1:0, mag2:1, lowmag:2, samag:3, diff:4."""
        if isinstance(value, str):
            try:
                value = self.FUNCTION_MODES.index(value)
            except ValueError:
                print('Unrecognized function mode:', value)
        self.eos3.SelectFunctionMode(value)

    # toogle functions
    def diffraction(self, checked_diff_value, kl = None):
        """" this function set directly to 350 KL to fix a bug in temscript, this is ok only for the tecnai spirit,
        we should set in an external file which is the standard KL to set"""
        if checked_diff_value:
            print("mag before diff:", self.get_magnification())
            self.previous_mag = self.get_magnification()
            self.setFunctionMode("diff") ### check if run
            if kl != None:
                kl = self.kl_index_table[kl][0] #to change here
                self.set_KL(kl)
            else:
                self.set_KL(9) #to change here
            pass

        else:
            self.setFunctionMode("samag") ### check if run and if samag is correct # ask magda
            #time.sleep(0.2)
            self.set_magnification(self.previous_mag)
        pass

    def get_defocus(self):
        """function that return the objective lens defocus value as an integer?? there are 2 functions OLc and OLf,
         meaning to coarse and fine, to checkout what is useful"""
        value, result = self.lens3.GetOLc()
        print("value coarse", value, "result coarse", result)
        value, result = self.lens3.GetOLf()
        print("value fine", value, "result fine", result)
        return value

    def set_defocus(self, defocus):
        """function that set the objective lens defocus value from an integer?? there are 2 functions OLc and OLf,
        meaning to coarse and fine, to checkout what is settlable"""
        print("defocus before:", self.get_defocus())
        try:
            self.lens3.SetOLc(defocus)
            print("defocus set from Olc:", self.get_defocus())
            self.lens3.SetOLf(defocus)
            print("defocus set from Olf:", self.get_defocus())
        except Exception as err:
            print(err)

    def euc_focus(self):
        print("Not implemented")

    def wobbler(self, checked_wobbler_value):
        if checked_wobbler_value:
            print("wobbler on")
            self.fake()
        else:
            print("wobbler off")
            self.fake()
        print("Not implemented")

    def move_screen(self, checked_screen_value):
        """function to move the fluorescent screen if checked_screen_value is True will move up,
        otherwise move it down."""

        UP, DOWN = 2, 0
        if checked_screen_value:
            print("screen up")
            self.screen2.SelectAngle(UP)
        else:
            print("screen down")
            self.screen2.SelectAngle(DOWN)

    def get_screen_position(self):
        """return if the screen is up or down as 'UP' or 'DOWN'. """
        value = self.screen2.GetAngle()[0]
        UP, DOWN = 2, 0
        if value == UP:
            print("screen up")
            return 'UP'
        elif value == DOWN:
            print("screen down")
            return 'DOWN'
        else:
            print("unkown screen position for get_screen_position", value)
            return value

    def beam_blank(self, checked_blank_value):
        """function to enable beam blanking if checked_blank_value is True blank the beam,
                otherwise unblank it"""
        if checked_blank_value:
            print("beam blank on")
            self.def3.SetBeamBlank(True)
        else:
            print("beam blank off")
            self.def3.SetBeamBlank(False)

    def apply_rotation(self, vector, theta=216.4):
        """input an x,y vector and return a rotated vector of theta degree, usually used for syncronize
         the movement of the beam shift with the user reference. the rotation is counterclockwise!!"""
        print("before:", vector)
        vector = np.asarray(vector)
        print("as np array:", vector)
        theta = np.radians(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        rotated_vector = rotation_matrix.dot(vector)
        print("rotated_vector_in_the_function:", rotated_vector)
        return tuple(rotated_vector)

    def fake(self):
        print("i'm a fake function used as placeholder")

    def get_stage(self, standard = False):
        """return the current stage position of x, y, z in um a and b in deg.
        from default API from JEOL return nm and deg"""
        x, y, z, a, b, result = self.stage3.GetPos()
        print("get stage for debug, units nm and deg: x", x, "y", y, "z", z, "a", a, "b", b, "result", result)
        x = x * 10 ** -3 # um
        y = y * 10 ** -3 # um
        z = z * 10 ** -3 # um
        pos = {"x": x, "y": y, "z": z, "a": a, "b": b}
        if standard == False:
            return pos

        elif standard == True:
            pos["a"] = (pos["a"], "deg")
            pos["b"] = (pos["b"], "deg")
            pos["x"] = (pos["x"], "um")
            pos["y"] = (pos["y"], "um")
            pos["z"] = (pos["z"], "um")
            return pos

    def set_stage_position(self, x = None, y = None, z = None, a = None, b = None):
        """ set the stage position of x, y, z provided in um a and b given in radians.
        JEOL API works in nm and deg, so we need to convert the values."""
        if z is not None:
            self.stage3.SetZ(z*(10**3))
            #time.sleep(0.3)
        if a is not None:
            self.stage3.SetTiltXAngle(a)
            #time.sleep(0.3)
        if b is not None:
            self.stage3.SetTiltYAngle(b)
            #time.sleep(0.3)
        if x is not None:
            self.stage3.SetX(x*(10**3))
            #time.sleep(0.3)
        if y is not None:
            self.stage3.SetY(y*(10**3))
            #time.sleep(0.3)


    def get_beam_shift(self):
        "get the beam shift value in ?? units, it's using CLA1 lens in JEOL i don't know if it is correct"
        x, y, result = self.def3.GetCLA1()
        print("beam shift line 306 x", x, "y", y, "result", result)
        return x, y

    def set_beam_shift(self, beam_pos):
        "set the beam shift value in ?? units"
        x, y = beam_pos
        self.def3.SetCLA1(x, y)

    def connect(self): # to test if pythocom works out otherwise we need to use comtypes and drop pyautogui because the bots are only for tecnai
        """microscope remote connection trough JEOL COM interface generated by TemExt.dll"""
        ############################# initialize using pythoncom ######################################################
        # try:
        #     pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
        # except:
        #     pythoncom.CoInitialize()

        # Get the JEOL COM library and create the TEM3 object
        # CLSID_TEM3 = ('{CE70FCE4-26D9-4BAB-9626-EC88DB7F6A0A}',3,0)
        # self.tem3 = win32com.client.Dispatch('{CE70FCE4-26D9-4BAB-9626-EC88DB7F6A0A}')
        # If the CLSID_TEM3 is not working, try using the ProgID
        # self.tem3 = win32com.client.Dispatch("TEM3.Application")
        ###############################################################################################################

        ############################## Initialize using comtypes ######################################################
        try:
            comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
            print("starting jeol come with mta")
        except OSError:
            comtypes.CoInitialize()
            print("starting jeol com with sta")

        # get the JEOL COM library and create the TEM3 object
        temext = comtypes.client.GetModule(('{CE70FCE4-26D9-4BAB-9626-EC88DB7F6A0A}', 3, 0))
        self.tem3 = comtypes.client.CreateObject(temext.TEM3, comtypes.CLSCTX_ALL)
        ################################################################################################################

        # initialize each interface from the TEM3 object
        self.camera3 = self.tem3.CreateCamera3()
        # self.detector3 = self.tem3.CreateDetector3()
        # self.feg3 = self.tem3.CreateFEG3()
        # self.filter3 = self.tem3.CreateFilter3()
        # self.gun3 = self.tem3.CreateGun3()
        # self.mds3 = self.tem3.CreateMDS3()
        self.apt3 = self.tem3.CreateApt3()
        self.screen2 = self.tem3.CreateScreen2()
        self.def3 = self.tem3.CreateDef3()
        self.eos3 = self.tem3.CreateEOS3()
        self.ht3 = self.tem3.CreateHT3()
        self.lens3 = self.tem3.CreateLens3()
        self.stage3 = self.tem3.CreateStage3()
        atexit.register(comtypes.CoUninitialize)

    def get_projection_mode(self):
        a = self.getFunctionMode()
        if a != 'diff':
            return 'IMAGING'
        else:
            return 'DIFFRACTION'

    # def getDiffFocus(self, confirm_mode: bool = True) -> int:
    #     if confirm_mode and (not self.getFunctionMode() == 'diff'):
    #         # raise JEOLValueError("Must be in 'diff' mode to get DiffFocus")
    #         print("line 419 you are not in diffraction to get diffocus, raise exception")
    #         raise Exception
    #     value, result = self.lens3.GetIL1()
    #     print("value", value, "result", result)
    #     return value
    #
    # def setDiffFocus(self, value: int, confirm_mode: bool = True):
    #     """IL1."""
    #     if confirm_mode and (not self.getFunctionMode() == 'diff'):
    #         # raise JEOLValueError("Must be in 'diff' mode to set DiffFocus")
    #         print("line 428 you are not in diffraction to get diffocus, raise exception")
    #         raise Exception
    #     self.lens3.setDiffFocus(value)

    def set_magnification(self, mag_value):
        """ tricky function there are 2 ways to get the info using: getMagnification that return a value integer
        or using getMagnificationIndex that return an integer as index"""
        #mag_value = self.mag_index_table[str(mag_value)]
        print("mag:", mag_value)
        print('if_mode_imaging?:', self.get_projection_mode())
        if self.get_projection_mode() == "IMAGING":
            self.eos3.SetSelector(mag_value)
        else:
            print("to change the mag you need to be in imaging!!")

    def get_magnification(self):
        """ tricky function there are 2 ways to get the info using: getMagnification that return a value integer
        or using getMagnificationIndex that return an integer as index"""
        # getmagvalue function here
        value, unit_str, label_str, result = self.eos3.GetMagValue()
        print("getmagnification function value", value, "unit_str", unit_str, "label_str", label_str, "result", result)

        # getmagindex function here, up to now this is the one used!
        selector, mag, status = self.eos3.GetCurrentMagSelectorID()
        print("getmagnificationindex function selector", selector, "mag", mag, "status", status)
        return mag


    def set_KL(self, kl_value):
        #kl_value = self.kl_index_table[str(kl_value)]
        print("kl:", kl_value)
        print('if_mode_diffraction?:', self.get_projection_mode())
        if self.get_projection_mode() == "DIFFRACTION":
            self.eos3.SetSelector(kl_value)
        else:
            print("to change the KL you need to be in diffraction!!")

    def get_KL(self):
        """"return the KL value should be in mm??"""
        return self.get_magnification()

    def get_intensity(self, slot=0):
        """ from instamatic the brightness control in the 2100f is CL3. for now implemented using CL3"""
        if self.get_instrument_mode() == "TEM":
            self.beam_intensity, _ = self.lens3.GetCL3()
            print("debug line 425 beam_intensity from C3 lens", self.beam_intensity, "result", _ )

            if slot == 1:
                self.beam_intensity_1 = self.beam_intensity
            elif slot == 2:
                self.beam_intensity_2 = self.beam_intensity
            else:
                return self.beam_intensity
        else:
            self.beam_intensity = self.get_defocus()
            if slot == 1:
                self.beam_intensity_1 = self.beam_intensity
            elif slot == 2:
                self.beam_intensity_2 = self.beam_intensity
            else:
                return self.beam_intensity
            print("Not implemented in STEM mode")

    def set_intensity(self, intensity=0, slot=0):
        """from instamatic the brightness control in the 2100f is CL3. for now implemented using CL3"""
        if self.get_instrument_mode() == "TEM":
            if slot == 1:
                try:
                    self.lens3.GetCL3(self.beam_intensity_1)
                except Exception as err:
                    print("stored value 1: ", self.beam_intensity_1, "error:", err)
            elif slot == 2:
                try:
                    self.lens3.GetCL3(self.beam_intensity_2)
                except Exception as err:
                    print("stored value 2: ", self.beam_intensity_2, "error:", err)
            else:
                self.lens3.GetCL3(intensity)
        else:
            if slot == 1:
                try:
                    self.set_defocus(self.beam_intensity_1)
                except Exception as err:
                    print("stored value 1: ", self.beam_intensity_1, "error:", err)
            elif slot == 2:
                try:
                    self.set_defocus(self.beam_intensity_2)
                except Exception as err:
                    print("stored value 2: ", self.beam_intensity_2, "error:", err)
            else:
                self.set_defocus(intensity)
            print("Not implemented in JEOL STEM mode")

    def get_spotsize(self):
        """return the cl1 index value, in jeol this should be the the spotsize,
        0-based indexing for GetSpotSize, add 1 to be consistent with JEOL software."""
        value, result = self.eos3.GetSpotSize()
        print("value", value, "result", result)
        return value + 1

    def set_spotsize(self, value):
        """Set the spotsize"""
        self.eos3.selectSpotSize(value - 1)

    def load_calibration_table(self, cam_table):
        """jeol have different mode we need to check the indexes such as mag1 mag2 samag lowmag diff"""
        self.cam_table = cam_table
        # magnification calibration spirit with screen up
        self.mag_index_table = self.cam_table["IMAGING"]
        self.kl_index_table = self.cam_table["DIFFRACTION"]

    def set_alpha(self, angle, velocity=1):
        """stage alpha movement, velocity in jeol cannot be set directly, the function will wait for the goniometer
         to stop, the angle should be provided in deg"""
        self.stage3.SetTiltXAngle(angle)
        delay = 0.3 #seconds
        time.sleep(delay)  # skip the first readout delay, necessary on NeoARM200
        while self.isStageMoving():
            print("debug line in set_alpha, stage is moving?", self.isStageMoving() == True)
            if delay > 0:
                time.sleep(delay)

    def isStageMoving(self):
        x, y, z, a, b, result = self.stage3.GetStatus()
        print("isstagemoving? return x", x, "y", y, "z", z, "a", a, "b", b, "result", result)
        return x or y or z or a or b

    def microscope_thread_setup(self, tracking_file = "tracking.txt", tracking_dict = None, timer = None, event = None, stop_event = None):
        """"this function read the tracking file and set up the threads necessary for the acqusition. 3 sockets are necessary to work.
        if tracking_positions == None and experiment_type == "continuous", the stage is threaded only for continuous rotation (trackless experiment).
        if tracking positions != None and experiment_type == "continuous", the stage is threaded for continuous rotation and the beam is threaded for tracking.
        if experiment_type == "stepwise", the beam only is threaded for tracking waiting to pass the target angle to apply the tracking beamshift.
        in the case of cred, results are displayed in the variable self.result """
        if tracking_dict["stem_mode"] == True:
            stem = tracking_dict["stem_mode"]
        else:
            stem = False
        self.result = []
        # initialize the thread for the stage
        try:
            # self.tem_stage = temscript.RemoteMicroscope((self.cam_table["ip"][0], self.cam_table["ip"][1] + 1))
            self.tem_stage = self
            print("tem_stage_thread connected")
            time.sleep(0.33)
            # self.tem_beam = temscript.RemoteMicroscope((self.cam_table["ip"][0], self.cam_table["ip"][1] + 2))
            self.tem_beam = self

            print("tem_beam_thread connected")
            time.sleep(0.33)
        except Exception as e:
                print("no connection please be sure that the sockets are open in the microscope and try again")
                raise Exception

        print("initialization of the parameter for the acquisition for the passive thread")

        # section to read the tracking file and extract the info
        # start_angle, target_angle, rotation_speed, experiment_type, tracking_step, tracking_positions,
        # tracking file is a txt file to read using read_tracking_file method from fast_adt_func.py
        if tracking_dict == None:
            tracking_dict = read_tracking_file(tracking_file)
        else:
            tracking_dict = tracking_dict
        print(tracking_dict)
        start_angle = tracking_dict["start_angle"]
        target_angle = tracking_dict["target_angle"]
        rotation_speed = tracking_dict["rotation_speed"]
        experiment_type = tracking_dict["experiment_type"]
        tracking_step = tracking_dict["tracking_step"]
        tracking_positions = tracking_dict["tracking_positions"]
        ###################
        mag = tracking_dict["mag"]
        kl = tracking_dict["kl"]

        if experiment_type == "continuous":
            rotation_speed_input = self.calc_stage_speed(rotation_speed)
        else:
            rotation_speed_input = "fake"

        target_angle = round(target_angle, 3)
        print("target angle: ", target_angle, " rotation speed (deg/s): ", rotation_speed, "rotation speed (a.u.): ", rotation_speed_input)
        target_angle_rad = math.radians(target_angle)
        print("target angle in rad: ", target_angle_rad)

        if tracking_dict["tracking_method"] == "prague_cred_method":
            step = int((abs(start_angle)+abs(target_angle))//5)
            #if start_angle > target_angle:
            #    ranges = list(np.linspace(start_angle, target_angle+5, 5))
            #else:
            ranges = list(np.linspace(start_angle, target_angle, step + 1))
            counter = int(len(ranges)/2)
            thread_beam_list = []
            thread_stage_list = []
            for i in range(counter+1):
                start_event = threading.Event()
                stop_event = threading.Event()
                result_list = []
                first = ranges[i]
                second = ranges[i + 1]
                if len(tracking_positions) != 0 and experiment_type == "continuous":
                        if stem == True:
                            thread_beam = threading.Thread(target=self.beamshift_tracking_stem,
                                                                kwargs={"tracking_dict": tracking_dict, "result": result_list, "timer": timer, "event": start_event,"stop_event": stop_event}, )

                        else:
                            thread_beam = threading.Thread(target=self.beamshift_tracking,
                                                                kwargs={"tracking_dict": tracking_dict, "result": result_list, "timer": timer, "event": start_event, "stop_event": stop_event}, )

                        thread_stage = threading.Thread(target=self.continuous_rotation,
                                                             kwargs={"a": second, "speed": rotation_speed_input, "event": start_event, "stop_event": stop_event}, )

                elif len(tracking_positions) == 0 and experiment_type == "continuous":
                    thread_beam = threading.Thread(target=self.angle_tracking,
                                                        kwargs={"final_angle": second, "result": result_list, "timer": timer, "event": start_event, "stop_event": stop_event}, )
                    # angle_tracking(self, final_angle, result: list):

                    thread_stage = threading.Thread(target=self.continuous_rotation,
                                                         kwargs={"a": second, "speed": rotation_speed_input, "event": start_event, "stop_event": stop_event}, )

                thread_beam_list.append((thread_beam, start_event, result_list))
                thread_stage_list.append((thread_stage, start_event))

            self.angular_range = abs(target_angle - start_angle)
            print("goniometer from: ", start_angle, "into: ", target_angle, "angular range: ", self.angular_range, " deg")

            return thread_stage_list, thread_beam_list

        # 2 threads are generated one for the stage and one for the acquisition, because the socket used for the stage cannot respond up to the end of its movement/task
        if len(tracking_positions) != 0 and experiment_type == "continuous":
            if stem == True:
                self.thread_beam = threading.Thread(target=self.beamshift_tracking_stem,
                                                    kwargs={"tracking_dict": tracking_dict,"result": self.result, "timer": timer, "event": event, "stop_event": stop_event}, )

            else:
                self.thread_beam = threading.Thread(target=self.beamshift_tracking,
                                                    kwargs={"tracking_dict": tracking_dict,"result": self.result, "timer": timer, "event": event, "stop_event": stop_event}, )

            self.thread_stage = threading.Thread(target=self.continuous_rotation,
                                                 kwargs={"a": target_angle, "speed": rotation_speed_input, "event": event, "stop_event": stop_event}, )

        elif len(tracking_positions) != 0 and experiment_type == "stepwise":
            if stem == True:
                self.thread_beam = threading.Thread(target=self.beamshift_tracking_stem,
                                                    kwargs={"tracking_dict": tracking_dict,"result": self.result, "event": event, "stop_event": stop_event}, )

            else:
                self.thread_beam = threading.Thread(target=self.beamshift_tracking,
                                                    kwargs={"tracking_dict": tracking_dict,"result": self.result, "event": event, "stop_event": stop_event}, )

            self.thread_stage = threading.Thread(target=self.fake)



        elif len(tracking_positions) == 0 and experiment_type == "continuous":

            self.thread_beam = threading.Thread(target=self.angle_tracking,
                                                kwargs = {"final_angle": target_angle, "result": self.result, "timer": timer, "event": event, "stop_event": stop_event},)
            #angle_tracking(self, final_angle, result: list):

            self.thread_stage = threading.Thread(target=self.continuous_rotation,
                                                 kwargs={"a": target_angle, "speed": rotation_speed_input, "event": event, "stop_event": stop_event}, )

        elif len(tracking_positions) == 0 and experiment_type == "stepwise":
            self.thread_beam = threading.Thread(target=self.fake)

            self.thread_stage = threading.Thread(target=self.fake)

        else:
            print("error in microscope_thread_setup function, aborting")
            raise Exception

        self.angular_range = abs(target_angle - start_angle)
        print("goniometer from: ", start_angle, "into: ", target_angle, "angular range: ", self.angular_range, " deg")

        return self.thread_stage, self.thread_beam

    def calc_stage_speed(self, speed):
        """"calculate the speed in rad/s for jeol. speed is provided in degrees/s and return it in rad/s."""
        try:
            self.calibrated_speed = None
            cwd = os.getcwd()
            table = cwd + os.sep + r"adaptor/camera/lookup_table/jem2100f_speed_lookuptable.csv"
            speed_table = pd.read_csv(table, sep='\t')
            speed_table_loaded = True
        except Exception as err:
            speed_table_loaded = False
            print("speed table not found")

        if speed_table_loaded == True:
            # Calculate the absolute difference
            speed_table['difference'] = abs(speed_table['deg/s'] - speed)
            # Find the index of the minimum difference
            closest_index = speed_table['difference'].idxmin()
            # Get the closest value
            closest_value = speed_table.loc[closest_index, 'deg/s']
            # Optionally, you can drop the 'difference' column if you no longer need it
            read_table = speed_table.drop(columns=['difference'])
            self.calibrated_speed = read_table.loc[closest_index]
            print(f'The closest value to the chosen speed: {speed} is {self.calibrated_speed["deg/s"]}, overall self.calibrated_speed:', self.calibrated_speed)
            speed = self.calibrated_speed["rad/s"]
            return speed

        else:
            return np.deg2rad(speed)

    def angle_tracking(self, final_angle, result: list, timer = None, event = None, stop_event = None):
        get_angle = self.get_stage
        if get_angle()["a"] > final_angle:
            towards_positive = False
        else:
            towards_positive = True
        event.wait()
        if stop_event != None and stop_event.is_set() == True:
            return
        if timer == None:
            i_time = time.monotonic_ns()
        else:
            ref_timings = {}
            i_time = time.monotonic_ns()
            ref_time = timer
            ref_timings["start_i_time"] = i_time
            ref_timings["start_ref_time"] = ref_time

        res_t = []
        res_a = []
        while True:
            pos = get_angle()["a"]
            x_time = (time.monotonic_ns() - i_time) / 1000000000
            if res_t == []:
                res_t.append(x_time)
                res_a.append(pos)
            elif pos != res_a[-1]:
                res_t.append(x_time)
                res_a.append(pos)

            time.sleep(0.10)
            if towards_positive == True:
                if get_angle()["a"] > final_angle - 1:
                    break
            else:
                if get_angle()["a"] < final_angle - 1:
                    break
        if timer != None:
            ref_timings["end_angle_tracking"] = time.monotonic_ns()
        # fit
        interpolate_function = interp1d(res_t[4:], res_a[4:], fill_value='extrapolate')
        interpolate_function_inverse = interp1d(res_a[4:], res_t[4:], fill_value='extrapolate')
        fit_t = list(np.linspace(res_t[1], res_t[-1], 50))
        fit_a = []

        fit_a = list(interpolate_function(fit_t))
        grad_a = np.gradient(fit_a)
        grad_t = np.gradient(fit_t)
        slope = np.round(np.mean(grad_a / grad_t), 3)

        result.append(interpolate_function)
        result.append(slope)
        result.append(res_t)
        result.append(res_a)
        result.append(interpolate_function_inverse)
        if timer != None:
            result.append(ref_timings)

    def beamshift_tracking(self, tracking_dict, result=None, timer = None, event = None, stop_event = None):
        """this thread live in a separate socket and check continuously the position of the stage.
        when the angle is in the one to apply the tracking, the beamshift is applied to track the crystal.
        the thread work passivly waiting for the angle to be reached/passed to work"""
        # beam_pos is in m and track_beam must be in the same syst of reference and unit!
        # this tracking work from negative to positive values! the opposite will probably not work!
        experiment_type = tracking_dict["experiment_type"]
        tracking_step = tracking_dict["tracking_step"]
        tracking_positions = tracking_dict["tracking_positions"]
        ub_class = tracking_dict["ub_class"]

        if experiment_type == "continuous":
            track_time = tracking_dict["tracking_times"]

        beam_pos = self.tem_beam.get_beam_shift()
        _, x0_p, y0_p = tracking_positions[0]

        mode = tracking_dict["projection_mode"]
        if mode == "DIFFRACTION":
            mode = "IMAGING"
            mag = tracking_dict["mag"]
        else:
            mag = tracking_dict["experimental_mag"]

        #tracking_delay = self.cam_table["Tracking_delay"]
        ub_class.angle_x = self.cam_table[mode][str(mag)][2]
        ub_class.scaling_factor_x = self.cam_table[mode][str(mag)][3]
        ub_class.angle_y = self.cam_table[mode][str(mag)][4]
        ub_class.scaling_factor_y = self.cam_table[mode][str(mag)][5]
        beam_p = ub_class.beamshift_to_pix(beam_pos, ub_class.angle_x, ub_class.scaling_factor_x,
                                           180 - ub_class.angle_y, ub_class.scaling_factor_y)
        if tracking_positions[0][0] > tracking_positions[-1][0]:
            toward_positive = False
        else:
            toward_positive = True
        # print("toward positive ? = ", toward_positive)
        #time.sleep(tracking_delay)  # parameter added by checking the difference in time between the starting of the tracking and the first image collected
        event.wait()
        if stop_event != None and stop_event.is_set() == True:
            return
        #########################
        beam_pos = self.tem_beam.get_beam_shift()
        _, x0_p, y0_p = tracking_positions[0]
        ub_class.angle_x = self.cam_table[mode][str(mag)][2]
        ub_class.scaling_factor_x = self.cam_table[mode][str(mag)][3]
        ub_class.angle_y = self.cam_table[mode][str(mag)][4]
        ub_class.scaling_factor_y = self.cam_table[mode][str(mag)][5]
        beam_p = ub_class.beamshift_to_pix(beam_pos, ub_class.angle_x, ub_class.scaling_factor_x,
                                           180 - ub_class.angle_y, ub_class.scaling_factor_y)
        #########################
        # new part for the fit of the angle during tracking
        if timer == None:
            i_time = time.monotonic_ns()
        else:
            ref_timings = {}
            i_time = time.monotonic_ns()
            ref_time = timer
            ref_timings["start_i_time"] = i_time
            ref_timings["start_ref_time"] = ref_time

        res_t = []
        res_a = []
        #live = []
        #i = 0
        #live1 = np.array([0])
        #live_time_track = 360 * 1000000000
        if experiment_type == "continuous":
            for parameters, timing in zip(tracking_positions, track_time):
                track_alpha, track_x, track_y = parameters
                track_alpha = round(track_alpha, 4)
                track_beam = (beam_p[0] + (track_x - x0_p),
                              beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
                track_beam = ub_class.pix_to_beamshift(track_beam, ub_class.angle_x, ub_class.scaling_factor_x,
                                                       180 - ub_class.angle_y, ub_class.scaling_factor_y)
                while True:
                    x_time = (time.monotonic_ns() - i_time) / 1000000000
                    current_position = self.tem_beam.get_stage_position()
                    current_alpha = round(current_position["a"], 4)
                    print("current_alpha = ", current_alpha)
                    print("track_alpha = ", track_alpha)
                    if res_t == []:
                        res_t.append(x_time)
                        res_a.append(current_alpha)
                    elif current_alpha != res_a[-1]:
                        res_t.append(x_time)
                        res_a.append(current_alpha)
                        # if len(res_t) > 3:
                        #     interpolate_function = interp1d(res_a[1:], res_t[1:], fill_value='extrapolate')
                        #     live_time_track = interpolate_function(track_alpha)
                        #     # live.append(list(live1))

                    if toward_positive == True:
                        #if round(current_alpha, 4) >= track_alpha:
                        #    self.tem_beam.set_beam_shift(track_beam)
                        #    break
                        if x_time >= timing:
                            self.tem_beam.set_beam_shift(track_beam)
                            break
                        else:
                            pass

                    elif toward_positive == False:
                        #if round(current_alpha, 4) <= track_alpha:
                        #    self.tem_beam.set_beam_shift(track_beam)
                        #    break
                        if x_time >= timing:
                            self.tem_beam.set_beam_shift(track_beam)
                            break
                        else:
                            pass
                    time.sleep(0.01)

            if timer != None:
                ref_timings["end_angle_tracking"] = time.monotonic_ns()

            # fit
            interpolate_function = interp1d(res_t[4:], res_a[4:], fill_value='extrapolate')

            fit_t = list(np.linspace(res_t[1], res_t[-1], 50))
            fit_a = []

            fit_a = list(interpolate_function(fit_t))
            grad_a = np.gradient(fit_a)
            grad_t = np.gradient(fit_t)
            slope = np.round(np.mean(grad_a / grad_t), 3)

            result.append(interpolate_function)
            result.append(slope)
            result.append(res_t)
            result.append(res_a)
            result.append(None)
            if timer != None:
                result.append(ref_timings)

        elif experiment_type == "stepwise":
        #     for parameters in tracking_positions:
        #         track_alpha, track_x, track_y = parameters
        #         track_alpha = round(np.deg2rad(track_alpha), 4)
        #         track_beam = (beam_p[0] + (track_x - x0_p),
        #                       beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
        #         track_beam = ub_class.pix_to_beamshift(track_beam, ub_class.angle_x, ub_class.scaling_factor_x,
        #                                                180 - ub_class.angle_y, ub_class.scaling_factor_y)
        #         while True:
        #             current_position = self.tem_beam.get_stage_position()
        #             current_alpha = round((current_position["a"]), 4)
        #             print("current_alpha = ", current_alpha)
        #             print("track_alpha = ", track_alpha)
        #
        #             if toward_positive == True:
        #                 if round(current_alpha, 4) >= track_alpha:
        #                     self.tem_beam.set_beam_shift(track_beam)
        #                     break
        #                 else:
        #                     pass
        #
        #             elif toward_positive == False:
        #                 if round(current_alpha, 4) <= track_alpha:
        #                     self.tem_beam.set_beam_shift(track_beam)
        #                     break
        #                 else:
        #                     pass
        #             time.sleep(0.01)
            pass

        time.sleep(1)
        print("beamshift tracking finished, reset original beam shift")
        self.tem_beam.set_beam_shift(beam_pos)

    def beamshift_tracking_stem(self, tracking_dict, result=None, timer = None, event = None, stop_event = None):
        """this thread live in a separate socket and check continuously the position of the stage.
        when the angle is in the one to apply the tracking, the beamshift is applied to track the crystal.
        the thread work passivly waiting for the angle to be reached/passed to work"""

        """ in ub_class this time is passed the fastadt_gui class to don't change everything and be able toi retrieve the haadf info"""
        # beam_pos is in m and track_beam must be in the same syst of reference and unit!
        # this tracking work from negative to positive values! the opposite will probably not work!
        experiment_type = tracking_dict["experiment_type"]
        tracking_step = tracking_dict["tracking_step"]
        tracking_positions = tracking_dict["tracking_positions"]
        ub_class = tracking_dict["ub_class"]

        if experiment_type == "continuous":
            track_time = tracking_dict["tracking_times"]

        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        _, x0_p, y0_p = tracking_positions[0]

        illumination_mode = tracking_dict["illumination_mode"]
        # kl = tracking_dict["experimental_kl"]
        mode = tracking_dict["projection_mode"]
        mag = tracking_dict["experimental_mag"]

        # convert the beamposition in pix
        haadf_table = ub_class.haadf.load_calibration_table()
        haadf_size = ub_class.haadf_cam_size
        calibration = haadf_table[mode][illumination_mode][mag][1] * tracking_dict["stem_binning_value"] / 1000000000
        tracking_delay = haadf_table["Tracking_delay"]
        beam_p_x = round(beam_pos[0] / calibration, 0)
        beam_p_y = round(beam_pos[1] / calibration, 0)
        beam_p = (beam_p_x, beam_p_y)
        print("\nthread_beam beam_pos nm", beam_pos, "\nbeam_pix", beam_p)

        if tracking_positions[0][0] > tracking_positions[-1][0]:
            toward_positive = False
        else:
            toward_positive = True
        # print("toward positive ? = ", toward_positive)

        # new part for the fit of the angle during tracking
        time.sleep(tracking_delay)  # parameter added by checking the difference in time between the starting of the tracking and the first image collected
        event.wait()
        if stop_event != None and stop_event.is_set() == True:
            return
        ################################
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        _, x0_p, y0_p = tracking_positions[0]
        beam_p_x = round(beam_pos[0] / calibration, 0)
        beam_p_y = round(beam_pos[1] / calibration, 0)
        beam_p = (beam_p_x, beam_p_y)
        print("\nthread_beam beam_pos nm", beam_pos, "\nbeam_pix", beam_p)
        ################################
        if timer == None:
            i_time = time.monotonic_ns()
        else:
            ref_timings = {}
            i_time = time.monotonic_ns()
            ref_time = timer
            ref_timings["start_i_time"] = i_time
            ref_timings["start_ref_time"] = ref_time

        res_t = []
        res_a = []
        # live = []
        # i = 0
        # live1 = np.array([0])
        # live_time_track = 360 * 1000000000
        if experiment_type == "continuous":
            for parameters, timing in zip(tracking_positions, track_time):
                track_alpha, track_x, track_y = parameters
                track_alpha = round(track_alpha, 4)
                track_beam = (beam_p[0] + (track_x - x0_p),
                              beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
                # conversion pix to beamshift
                track_beam = ((track_beam[0] * calibration), (track_beam[1] * calibration))

                while True:
                    x_time = (time.monotonic_ns() - i_time) / 1000000000
                    current_position = self.tem_beam.get_stage_position()
                    current_alpha = round((current_position["a"]), 4)
                    print("current_alpha = ", current_alpha)
                    print("track_alpha = ", track_alpha)
                    if res_t == []:
                        res_t.append(x_time)
                        res_a.append(current_alpha)
                    elif current_alpha != res_a[-1]:
                        res_t.append(x_time)
                        res_a.append(current_alpha)
                        # if len(res_t) > 3:
                        #     interpolate_function = interp1d(res_a[1:], res_t[1:], fill_value='extrapolate')
                        #     live_time_track = interpolate_function(track_alpha)
                        #     # live.append(list(live1))

                    if toward_positive == True:
                        #if round(current_alpha, 4) >= track_alpha:
                        #    self.client.client_send_action({"set_stem_beam": track_beam})
                        #    break
                        if x_time >= timing:
                            self.client.client_send_action({"set_stem_beam": track_beam})
                            break
                        else:
                            pass

                    elif toward_positive == False:
                        #if round(current_alpha, 4) <= track_alpha:
                        #    self.client.client_send_action({"set_stem_beam": track_beam})
                        #    break
                        if x_time >= timing:
                            self.client.client_send_action({"set_stem_beam": track_beam})
                            break
                        else:
                            pass
                    time.sleep(0.01)

            if timer != None:
                ref_timings["end_angle_tracking"] = time.monotonic_ns()

            # fit
            interpolate_function = interp1d(res_t[4:], res_a[4:], fill_value='extrapolate')

            fit_t = list(np.linspace(res_t[1], res_t[-1], 50))
            fit_a = []

            fit_a = list(interpolate_function(fit_t))
            grad_a = np.gradient(fit_a)
            grad_t = np.gradient(fit_t)
            slope = np.round(np.mean(grad_a / grad_t), 3)

            result.append(interpolate_function)
            result.append(slope)
            result.append(res_t)
            result.append(res_a)
            result.append(None)
            if timer != None:
                result.append(ref_timings)

        elif experiment_type == "stepwise":
        #     for parameters in tracking_positions:
        #         track_alpha, track_x, track_y = parameters
        #         track_alpha = round(np.deg2rad(track_alpha), 4)
        #         track_beam = (beam_p[0] + (track_x - x0_p),
        #                       beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
        #         # conversion pix to beamshift
        #         track_beam = ((track_beam[0] * calibration), (track_beam[1] * calibration))
        #
        #         while True:
        #             current_position = self.tem_beam.get_stage_position()
        #             current_alpha = round((current_position["a"]), 4)
        #             print("current_alpha = ", current_alpha)
        #             print("track_alpha = ", track_alpha)
        #
        #             if toward_positive == True:
        #                 if round(current_alpha, 4) >= track_alpha:
        #                     self.client.client_send_action({"set_stem_beam": track_beam})
        #                     break
        #                 else:
        #                     pass
        #
        #             elif toward_positive == False:
        #                 if round(current_alpha, 4) <= track_alpha:
        #                     self.client.client_send_action({"set_stem_beam": track_beam})
        #                     break
        #                 else:
        #                     pass
        #             time.sleep(0.01)
            pass

        time.sleep(1)
        #if reset == True:
        print("beamshift tracking finished, reset original beam shift")
        self.client.client_send_action({"set_stem_beam": beam_pos})

    def get_voltage(self):
        """Get the actual accelaration voltage in kV."""
        value, status = self.ht3.GetHTValue()
        return value/1000

    def test_tracking(self, tracking, ub_class=None):
        """this thread live in a separate socket and check continuously the position of the stage.
        when the angle is in the one to apply the tracking, the beamshift is applied to track the crystal.
        the thread work passivly waiting for the angle to be reached/passed to work"""
        # beam_pos is in m and track_beam must be in the same syst of reference and unit!
        # this tracking work from negative to positive values! the opposite will probably not work!
        tracking_positions = tracking

        beam_pos = self.get_beam_shift()
        _, x0_p, y0_p = tracking_positions[0]
        mode = self.get_projection_mode()
        mag = str(round(self.get_magnification()))
        ub_class.angle_x = self.cam_table[mode][str(mag)][2]
        ub_class.scaling_factor_x = self.cam_table[mode][str(mag)][3]
        ub_class.angle_y = self.cam_table[mode][str(mag)][4]
        ub_class.scaling_factor_y = self.cam_table[mode][str(mag)][5]

        beam_p = ub_class.beamshift_to_pix(beam_pos, ub_class.angle_x, ub_class.scaling_factor_x,
                                           180 - ub_class.angle_y, ub_class.scaling_factor_y)

        for parameters in tracking_positions:
            _, track_x, track_y = parameters
            track_beam = (beam_p[0] + (track_x - x0_p),
                          beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
            track_beam = ub_class.pix_to_beamshift(track_beam, ub_class.angle_x, ub_class.scaling_factor_x,
                                                   180 - ub_class.angle_y, ub_class.scaling_factor_y)

            self.set_beam_shift(track_beam)
            time.sleep(0.75)

        print("beamshift tracking finished, reset original beam shift")
        self.set_beam_shift(beam_pos)

    def test_tracking_stem(self, ub_class):
        """this thread live in a separate socket and check continuously the position of the stage.
        when the angle is in the one to apply the tracking, the beamshift is applied to track the crystal.
        the thread work passivly waiting for the angle to be reached/passed to work"""

        """ in ub_class this time is passed the fastadt_gui class to don't change everything and be able toi retrieve the haadf info"""

        tracking_dict = read_tracking_file(ub_class, "tracking.txt")
        print(tracking_dict)
        start_angle = tracking_dict["start_angle"]
        target_angle = tracking_dict["target_angle"]
        rotation_speed = tracking_dict["rotation_speed"]
        experiment_type = tracking_dict["experiment_type"]
        tracking_step = tracking_dict["tracking_step"]
        tracking_positions = tracking_dict["tracking_positions"]

        # beam_pos is in m and track_beam must be in the same syst of reference and unit!
        # this tracking work from negative to positive values! the opposite will probably not work!
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        _, x0_p, y0_p = tracking_positions[0]

        illumination_mode = self.tem.get_illumination_mode()
        # kl = (self.tem.get_KL())
        mode = self.get_projection_mode()
        mag = str(round(self.tem.get_indicated_magnification()))

        # convert the beamposition in pix
        haadf_table = ub_class.haadf.load_calibration_table()
        calibration = haadf_table[mode][illumination_mode][mag][1] * ub_class.get_stem_binning_value()/ 1000000000

        beam_p_x = round(beam_pos[0] / calibration, 0)
        beam_p_y = round(beam_pos[1] / calibration, 0)
        beam_p = (beam_p_x, beam_p_y)
        print("\nthread_beam beam_pos nm", beam_pos, "\nbeam_pix", beam_p)

        # new part for the fit of the angle during tracking
        for parameters in tracking_positions:
            track_alpha, track_x, track_y = parameters
            track_alpha = round(track_alpha, 4)
            track_beam = (beam_p[0] + (track_x - x0_p),
                          beam_p[1] + (track_y - y0_p))  # now the path is relative to the initial beam position
            # conversion pix to beamshift
            track_beam = ((track_beam[0] * calibration), (track_beam[1] * calibration))
            print("track_beam", track_beam)

            self.client.client_send_action({"set_stem_beam": track_beam})
            time.sleep(0.33)

        time.sleep(1)
        print("beamshift tracking finished, reset original beam shift")
        self.client.client_send_action({"set_stem_beam": beam_pos})

    def continuous_rotation(self, a, speed, event = None, stop_event = None):
        """in jeol the gonio velocity can be changed only using goniotool an external exe from service.
        self.tem_stage is an object of the class RemoteMicroscope in fei that control the stage rotation,
        settled up by the function self.microscope_thread_setup()"""
        if event != None:
            event.wait()
        if stop_event != None and stop_event.is_set() == True:
            return
        if self.get_stage()["a"] > a:
            towards_positive = False
        else: towards_positive = True

        self.set_alpha(angle = a) # this is going at the max velocity

        while True:
            angl = self.get_stage()["a"]
            if  angl >= (a-0.1):
                break
            else: print(angl)
        print("rotation_finished")

    def get_illumination_mode(self):
        """return micro or nanoprobe for the condenser minilens"""
        print("Not implemented in JEOL1")
        return None

    def set_illumination_mode(self, mode):
        """set micro or nanoprobe for the condenser minilens"""
        print("Not implemented in JEOL2")


    def get_instrument_mode(self):
        """return the current instrument mode i.e. TEM/STEM"""
        # print("Not implement in JEOL3") commented because otherwise will always print the message to check if tem or stem mode is on
        return "TEM"

    def set_instrument_mode(self, mode):
        """set the current instrument mode i.e. TEM/STEM"""
        print("Not implement in JEOL4")

if __name__ == "__main__":
    tem = Tem_jeol()