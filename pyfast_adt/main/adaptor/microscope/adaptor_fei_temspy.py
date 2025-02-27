import sys
sys.path.append(r'L:\Marco\tracking_routine_project\marco_module\mod_scripts\python_FEI_FastADT\07042023_build_up\main')
print(sys.path)
import time
from .adaptor_tem import Tem_base
import numpy as np
import json
import os
import temscript
import threading
import math
from fast_adt_func import read_tracking_file
from .temspy_socket import SocketServerClient
from scipy.interpolate import interp1d
import pandas as pd
class Tem_fei_temspy(Tem_base): # this is self.tem in FAST-ADT_GUI.py
    """every angle for moving the satge must be in deg as input and output, velocity for the stage in radian/s and um for the stage xyz movement"""
    def __init__(self, ip = '192.168.21.1', port = 8080, cam_table = None, master = None):
        super().__init__()
        self.tem = None
        self.connect(ip, port)
        if cam_table != None:
            self.load_calibration_table(cam_table)
        else: self.cam_table = None
        self.result = []
        self.master = master
        # ports 8080, 8081, 8082 are used by pyFastADT, 8083 is used to control temspy
        self.client = SocketServerClient(mode='client', host=ip, port=8083, tem = "f30")
        self.client.start()
        self.client.client_send_action({"check_configuration": ["f30"]})
        self.calibrated_speed = None
    # stage movements
    def move_stage_up(self, stage_ampl):
        print("I'm going stage up %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        # print(stage_pos)
        shift = (0, stage_ampl) #um
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["stage_rotation"])
        shift_rot_x = stage_pos['x'] + shift_rot_x
        shift_rot_y = stage_pos['y'] + shift_rot_y
        # print(shift)
        self.tem.set_stage_position(x=shift_rot_x, y=shift_rot_y)

    def move_stage_down(self, stage_ampl):
        print("I'm going stage down %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        shift = (0, stage_ampl)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["stage_rotation"])
        shift_rot_x = stage_pos['x'] - shift_rot_x
        shift_rot_y = stage_pos['y'] - shift_rot_y
        # print(shift)
        self.tem.set_stage_position(x=shift_rot_x, y=shift_rot_y)

    def move_stage_left(self, stage_ampl):
        print("I'm going stage left %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        shift = (stage_ampl, 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["stage_rotation"])
        shift_rot_x = stage_pos['x'] - shift_rot_x
        shift_rot_y = stage_pos['y'] - shift_rot_y
        # print(shift)
        self.tem.set_stage_position(x=shift_rot_x, y=shift_rot_y)

    def move_stage_right(self, stage_ampl):
        print("I'm going stage right %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        shift = (stage_ampl, 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["stage_rotation"])
        shift_rot_x = stage_pos['x'] + shift_rot_x
        shift_rot_y = stage_pos['y'] + shift_rot_y
        # print(shift)
        self.tem.set_stage_position(x=shift_rot_x, y=shift_rot_y)

    def move_stage_z_up(self, stage_ampl):
        print("I'm going stage Z up %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        # print(stage_pos)
        shift = stage_pos['z'] + stage_ampl
        # print(shift)
        self.tem.set_stage_position(z=shift)

    def move_stage_z_down(self, stage_ampl):
        print("I'm going stage Z down %s!" % str(stage_ampl))
        stage_pos = self.get_stage()
        # print(stage_pos)
        shift = stage_pos['z'] - stage_ampl
        # print(shift)
        self.tem.set_stage_position(z=shift)

    # beamshift movements
    def move_beam_up(self, beam_ampl):
        print("I'm going beam up %s!" % str(beam_ampl))
        beam_pos = self.get_beam_shift()
        shift = (0, beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["TEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.tem.set_beam_shift(beam_pos)

    def move_beam_down(self, beam_ampl):
        print("I'm going beam down %s!" % str(beam_ampl))
        beam_pos = self.get_beam_shift()
        shift = (0, - beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["TEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.tem.set_beam_shift(beam_pos)

    def move_beam_left(self, beam_ampl):
        print("I'm going beam left %s!" % str(beam_ampl))
        beam_pos = self.get_beam_shift()
        shift = (- beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["TEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.tem.set_beam_shift(beam_pos)

    def move_beam_right(self, beam_ampl):
        print("I'm going beam right %s!" % str(beam_ampl))
        beam_pos = self.get_beam_shift()
        shift = (beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["TEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.tem.set_beam_shift(beam_pos)

    # beamshift movements in stem
    def move_stem_beam_up(self, beam_ampl):
        print("I'm going beam up %s!" % str(beam_ampl))
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        shift = (0, beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["STEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.client.client_send_action({"set_stem_beam": beam_pos})

    def move_stem_beam_down(self, beam_ampl):
        print("I'm going beam down %s!" % str(beam_ampl))
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        shift = (0, - beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["STEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.client.client_send_action({"set_stem_beam": beam_pos})

    def move_stem_beam_left(self, beam_ampl):
        print("I'm going beam left %s!" % str(beam_ampl))
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        shift = (- beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["STEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.client.client_send_action({"set_stem_beam": beam_pos})

    def move_stem_beam_right(self, beam_ampl):
        print("I'm going beam right %s!" % str(beam_ampl))
        beam_pos = self.client.client_send_action({"get_stem_beam": 0})["get_stem_beam"]
        shift = (beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=self.cam_table["STEM_beam_rotation"])
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)
        self.client.client_send_action({"set_stem_beam": beam_pos})

    # toogle functions
    def diffraction(self, checked_diff_value, kl = None):
        """" this function set directly to 350 KL to fix a bug in temscript, this is ok only for the tecnai spirit,
        we should set in an external file which is the standard KL to set"""
        if checked_diff_value:
            print("mag before diff:", self.tem.get_magnification_index())
            self.previous_mag = self.tem.get_magnification_index()
            self.tem.set_projection_mode('DIFFRACTION')
            #time.sleep(0.2)
            if kl != None:
                kl = self.kl_index_table[kl][0]
                self.set_KL(kl)
            else:
                self.set_KL(9)

        else:
            self.tem.set_projection_mode('IMAGING')
            #time.sleep(0.2)
            self.tem.set_magnification_index(self.previous_mag)

    def get_defocus(self):
        return self.tem.get_defocus()

    def set_defocus(self, defocus):
        self.tem.set_defocus(defocus)

    def euc_focus(self):
        print("euc_focus on")
        self.tem.set_defocus(0.0)

    def wobbler(self, checked_wobbler_value):
        if checked_wobbler_value:
            print("wobbler on")
            self.fake()
        else:
            print("wobbler off")
            self.fake()

    def move_screen(self, checked_screen_value):
        if checked_screen_value:
            print("screen up")
            self.tem.set_screen_position('UP')
        else:
            print("screen down")
            self.tem.set_screen_position('DOWN')
    def get_screen_position(self):
        return self.tem.get_screen_position()

    def beam_blank(self, checked_blank_value):
        if checked_blank_value:
            print("beam blank on")
            self.tem.set_beam_blanked(True)
        else:
            print("beam blank off")
            self.tem.set_beam_blanked(False)

    def apply_rotation(self, vector, theta=216.4):
        # input an x,y vector and return a rotated vector of theta degree,
        # in order to syncronize the movement of the beam shift with the user reference
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
        """return it in deg and um if standard == True return also the measuring units"""
        if standard == False:
            pos = self.tem.get_stage_position()
            pos["a"] = np.rad2deg(pos["a"])
            pos["b"] = np.rad2deg(pos["b"])
            pos["x"] = pos["x"] * 10 ** 6
            pos["y"] = pos["y"] * 10 ** 6
            pos["z"] = pos["z"] * 10 ** 6
            return pos

        elif standard == True:
            pos = self.tem.get_stage_position()
            pos["a"] = (np.rad2deg(pos["a"]), "deg")
            pos["b"] = (np.rad2deg(pos["b"]), "deg")
            pos["x"] = (pos["x"]*10**6, "um")
            pos["y"] = (pos["y"]*10**6, "um")
            pos["z"] = (pos["z"]*10**6, "um")
            return pos

    def set_stage_position(self, x = None, y = None, z = None, a = None, b = None):
        if z is not None:
            self.tem.set_stage_position(z = z * 10 ** -6)
        if a is not None:
            self.tem.set_stage_position(a = np.deg2rad(a))
        if b is not None:
            self.tem.set_stage_position(b = np.deg2rad(b))
        if x is not None:
            self.tem.set_stage_position(x = x * 10 ** -6)
        if y is not None:
            self.tem.set_stage_position(y = y * 10 ** -6)
    def get_beam_shift(self):
        """return the value of the beam shift"""
        return self.tem.get_beam_shift()

    def set_beam_shift(self, beam_pos):
        """set the value of the beam shift"""
        self.tem.set_beam_shift(beam_pos)

    def connect(self, ip, port):
        # microscope remote connection on port 8080
        self.ip = ip
        self.port = port
        try:
            self.tem = temscript.RemoteMicroscope((self.ip, self.port))
            print("tem connected in remote mode at: %s, %s"%(self.ip, self.port))
        except Exception as err:
            print('unable to connect the tem at: %s, %s'%(self.ip, self.port), '\nerror: ',err)

    def get_projection_mode(self):
        return self.tem.get_projection_mode()

    def set_magnification(self, mag_value):
        #mag_value = self.mag_index_table[str(mag_value)]
        print("mag:", mag_value)
        print('if_mode_imaging?:', self.tem.get_projection_mode())
        if self.tem.get_projection_mode() == "IMAGING":
            self.tem.set_magnification_index(mag_value)
        else:
            print("to change the mag you need to be in imaging!!")

    def get_magnification(self):
        return int(self.tem.get_indicated_magnification())

    def set_KL(self, kl_value):
        #kl_value = self.kl_index_table[str(kl_value)]
        print("kl:", kl_value)
        print('if_mode_diffraction?:', self.tem.get_projection_mode())
        if self.tem.get_projection_mode() == "DIFFRACTION":
            self.tem.set_magnification_index(kl_value)
        else:
            print("to change the KL you need to be in diffraction!!")

    def get_KL(self):
        """"return the KL value in mm"""
        return int(round(float(self.tem.get_indicated_camera_length())*1000,0))

    def get_intensity(self, slot=0):
        if self.tem.get_instrument_mode() == "TEM":
            self.beam_intensity = self.tem.get_intensity()
            if slot == 1:
                self.beam_intensity_1 = self.beam_intensity
            elif slot == 2:
                self.beam_intensity_2 = self.beam_intensity
            else:
                return self.beam_intensity
        else:
            self.beam_intensity = self.tem.get_defocus()
            if slot == 1:
                self.beam_intensity_1 = self.beam_intensity
            elif slot == 2:
                self.beam_intensity_2 = self.beam_intensity
            else:
                return self.beam_intensity

    def set_intensity(self, intensity=0, slot=0):
        if self.tem.get_instrument_mode() == "TEM":
            if slot == 1:
                try:
                    self.tem.set_intensity(self.beam_intensity_1)
                except Exception as err:
                    print("stored value 1: ", self.beam_intensity_1, "error:", err)
            elif slot == 2:
                try:
                    self.tem.set_intensity(self.beam_intensity_2)
                except Exception as err:
                    print("stored value 2: ", self.beam_intensity_2, "error:", err)
            else:
                self.tem.set_intensity(intensity)
        else:
            if slot == 1:
                try:
                    self.tem.set_defocus(self.beam_intensity_1)
                except Exception as err:
                    print("stored value 1: ", self.beam_intensity_1, "error:", err)
            elif slot == 2:
                try:
                    self.tem.set_defocus(self.beam_intensity_2)
                except Exception as err:
                    print("stored value 2: ", self.beam_intensity_2, "error:", err)
            else:
                self.tem.set_defocus(intensity)

    def get_spotsize(self):
        """return the c1 index value, in fei/termofisher this is the spotsize"""
        return self.tem.get_spot_size_index()
    def load_calibration_table(self, cam_table):
        self.cam_table = cam_table
        # magnification calibration spirit with screen up
        self.mag_index_table = self.cam_table["IMAGING"]
        self.kl_index_table = self.cam_table["DIFFRACTION"]

    def set_alpha(self, angle, velocity=1): #deg
        angle = np.deg2rad(angle) #rad
        self.tem.set_stage_position(a = angle)

    def set_alpha_temspy(self, angle, velocity=1, event = None, stop_event = None): #deg
        """this is not really compatible with the prague method right now, because the bot start changing the value
        and the wait event will only press the button but all the threads starts togheter"""
        #angle = np.deg2rad(angle)
        print("debug line:", angle, velocity)
        self.client.client_send_action({"cred_temspy_setup": (np.round(angle, 4), np.round(velocity, 4), "A")})
        if event:
            event.wait()
        if stop_event != None and stop_event.is_set() == True:
            return
        self.client.client_send_action({"cred_temspy_go": 0})
        #self.tem.set_stage_position(a=angle, velocity = velocity)

    def set_xyz_temspy(self, value, axis, velocity=1, event = None, stop_event = None):
        """move an axis of the gonio using compustage temspy"""
        #angle = np.deg2rad(angle)
        self.client.client_send_action({"cred_temspy_setup": (np.round(value, 4), np.round(velocity, 4), str(axis))})
        if event:
            event.wait()
        if stop_event != None and stop_event.is_set() == True:
            return
        time.sleep(0.1)
        self.client.client_send_action({"cred_temspy_go": "True"})


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
            self.tem_stage = temscript.RemoteMicroscope((self.cam_table["ip"][0], self.cam_table["ip"][1] + 1))
            print("tem_stage_thread connected")
            time.sleep(0.33)
            self.tem_beam = temscript.RemoteMicroscope((self.cam_table["ip"][0], self.cam_table["ip"][1] + 2))
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
        target_angle_rad = np.deg2rad(target_angle)
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
                                                                kwargs={"tracking_dict": tracking_dict, "result": result_list, "timer": timer, "event": start_event, "stop_event": stop_event }, )

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

    def calc_stage_speed(self, speed): #deg
        """"calculate the speed in degrees/s for the tecnai series, 1 is equivalent to the maximum (normalized).
        speed is provided in degrees/s and return it in rad/s."""
        try:
            self.calibrated_speed = None
            cwd = os.getcwd()
            table = cwd + os.sep + r"adaptor/camera/lookup_table/f30_speed_lookuptable.csv"
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
            print(f'The closest value to the chosen speed: {speed} is {self.calibrated_speed["deg/s"]}')
            speed = self.calibrated_speed["rad/s"]

            return speed

        else:
            speed = 200.54 * ((79.882 * (speed ** -1.001)) ** -1.057)
            speed = np.deg2rad(speed)
            if speed > 1:
                speed = 1
            return speed

    def angle_tracking(self, final_angle, result: list, timer = None, event = None, stop_event = None):
        get_angle = self.tem_beam.get_stage_position
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
        return self.tem.get_voltage()

    def test_tracking(self, tracking, ub_class=None):
        """this thread live in a separate socket and check continuously the position of the stage.
        when the angle is in the one to apply the tracking, the beamshift is applied to track the crystal.
        the thread work passivly waiting for the angle to be reached/passed to work"""
        # beam_pos is in m and track_beam must be in the same syst of reference and unit!
        # this tracking work from negative to positive values! the opposite will probably not work!
        tracking_positions = tracking

        beam_pos = self.tem.get_beam_shift()
        _, x0_p, y0_p = tracking_positions[0]
        mode = self.tem.get_projection_mode()
        mag = str(round(self.tem.get_indicated_magnification()))
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

            self.tem.set_beam_shift(track_beam)
            time.sleep(0.75)

        print("beamshift tracking finished, reset original beam shift")
        self.tem.set_beam_shift(beam_pos)

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
        mode = self.tem.get_projection_mode()
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
        """ modified cred function for fei_temspy because the stage is rotated by the bot.
        so it's normal that the wait here is commented because the wait is in the set_alpha_temspy function.
        a input in deg and speed in a.u. (radians)"""

        # if event != None:
        #    event.wait()
        # if stop_event != None and stop_event == True:
        #     return
        if self.tem_stage.get_stage_position()["a"] > a:
            towards_positive = False
        else: towards_positive = True
        self.set_alpha_temspy(a, speed, event = event, stop_event = stop_event)

        if stop_event != None and stop_event.is_set() == True:
            return

        while True:
            angl = np.rad2deg(self.tem_stage.get_stage_position()["a"])
            if angl >= (a - 0.1):
                break
            else:
                print('debug line here 1065 cont_rotqation method', angl, a)
        print("rotation_finished")

    def get_illumination_mode(self):
        """return micro or nanoprobe for the condenser minilens fei"""
        return self.tem.get_illumination_mode()

    def set_illumination_mode(self, mode):
        """set micro or nanoprobe for the condenser minilens fei"""
        self.tem.set_illumination_mode(mode)

    def get_instrument_mode(self):
        """return the current instrument mode i.e. TEM/STEM"""
        return self.tem.get_instrument_mode()

    def set_instrument_mode(self, mode):
        """set the current instrument mode i.e. TEM/STEM"""
        self.tem.set_instrument_mode(mode)
