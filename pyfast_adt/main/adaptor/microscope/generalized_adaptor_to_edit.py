from .adaptor_tem import Tem_base
import numpy as np
import math

# import temscript # here is necessary pyjem or whatever control the com of jeol

class Tem_jeol(Tem_base):
    def __init__(self, ip='192.168.21.1', port=8080, cam_table = None):
        super().__init__()
        self.tem = None
        self.connect(ip, port)
        if cam_table != None:
            self.load_calibration_table(cam_table)
        else: self.cam_table = None
    # stage movements
    def move_stage_up(self, stage_ampl):
        print("I'm going stage up %s!" % str(stage_ampl))

        stage_pos = self.get_stage()  ###########

        # apply a rotation to the shift vector
        shift = (0, stage_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=-90)
        shift_rot_x = stage_pos['x'] + shift_rot_x
        shift_rot_y = stage_pos['y'] + shift_rot_y

        # set the correct shift
        self.tem.set_stage_position(x=shift_rot_x, y=shift_rot_y)  ###########

    def move_stage_down(self, stage_ampl):
        print("I'm going stage down %s!" % str(stage_ampl))

        stage_pos = self.get_stage()  ################

        shift = (0, stage_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=-90)
        shift_rot_x = stage_pos['x'] - shift_rot_x
        shift_rot_y = stage_pos['y'] - shift_rot_y

        # print(shift)
        self.tem.set_stage_position(x=shift_rot_x, y=shift_rot_y)  ###############

    def move_stage_left(self, stage_ampl):
        print("I'm going stage left %s!" % str(stage_ampl))

        stage_pos = self.get_stage()  #################

        shift = (stage_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=-90)
        shift_rot_x = stage_pos['x'] - shift_rot_x
        shift_rot_y = stage_pos['y'] - shift_rot_y

        # print(shift)
        self.tem.set_stage_position(x=shift_rot_x, y=shift_rot_y)  ############

    def move_stage_right(self, stage_ampl):
        print("I'm going stage right %s!" % str(stage_ampl))

        stage_pos = self.get_stage()  #################

        shift = (stage_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift, theta=-90)
        shift_rot_x = stage_pos['x'] + shift_rot_x
        shift_rot_y = stage_pos['y'] + shift_rot_y

        # print(shift)
        self.tem.set_stage_position(x=shift_rot_x, y=shift_rot_y)  ##############

    def move_stage_z_up(self, stage_ampl):
        print("I'm going stage Z up %s!" % str(stage_ampl))

        stage_pos = self.get_stage()  ##############

        # print(stage_pos)
        shift = stage_pos['z'] + stage_ampl * (10 ** -6)

        # print(shift)
        self.tem.set_stage_position(z=shift)  #############

    def move_stage_z_down(self, stage_ampl):
        print("I'm going stage Z down %s!" % str(stage_ampl))

        stage_pos = self.get_stage()  ############

        # print(stage_pos)
        shift = stage_pos['z'] - stage_ampl * (10 ** -6)

        # print(shift)
        self.tem.set_stage_position(z=shift)  ###########

    # beamshift movements
    def move_beam_up(self, beam_ampl):
        print("I'm going beam up %s!" % str(beam_ampl))

        beam_pos = self.get_beam_shift()  ###########

        shift = (0, beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift)
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)

        self.tem.set_beam_shift(beam_pos)  #############

    def move_beam_down(self, beam_ampl):
        print("I'm going beam down %s!" % str(beam_ampl))

        beam_pos = self.get_beam_shift()  ##########

        shift = (0, - beam_ampl * (10 ** -6))
        shift_rot_x, shift_rot_y = self.apply_rotation(shift)
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)

        self.tem.set_beam_shift(beam_pos)  ############

    def move_beam_left(self, beam_ampl):
        print("I'm going beam left %s!" % str(beam_ampl))

        beam_pos = self.get_beam_shift()  ############

        shift = (- beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift)
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)

        self.tem.set_beam_shift(beam_pos)  ############

    def move_beam_right(self, beam_ampl):
        print("I'm going beam right %s!" % str(beam_ampl))

        beam_pos = self.get_beam_shift()  #########

        shift = (beam_ampl * (10 ** -6), 0)
        shift_rot_x, shift_rot_y = self.apply_rotation(shift)
        beam_pos = (beam_pos[0] + shift_rot_x, beam_pos[1] + shift_rot_y)

        self.tem.set_beam_shift(beam_pos)  ########

    # toogle functions
    def diffraction(self, checked_diff_value, kl = None):
        if checked_diff_value:
            print("diffraction on, this funct is bugged")

            self.tem.set_projection_mode('DIFFRACTION')  ###########
        else:
            print("diffraction off, this funct is bugged")

            self.tem.set_projection_mode('IMAGING')  ############

    def euc_focus(self):
        print("euc_focus on")

        self.tem.set_defocus(0.0)  #############

    def wobbler(self, checked_wobbler_value):
        if checked_wobbler_value:
            print("wobbler on")

            self.fake()  ##########
        else:
            print("wobbler off")

            self.fake()  ############

    def move_screen(self, checked_screen_value):
        if checked_screen_value:
            print("screen up")

            self.tem.set_screen_position('UP')  ##############
        else:
            print("screen down")

            self.tem.set_screen_position('DOWN')  ###########

    def get_screen_position(self):
        pass

    def apply_rotation(self, vector, theta=212):
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

    def get_stage(self):
        return self.tem.get_stage_position()  ############

    def get_beam_shift(self):
        return self.tem.get_beam_shift()  ###########

    def connect(self, ip, port):
        # microscope remote connection on port 8080
        self.ip = ip
        self.port = port
        try:
            self.tem = temscript.RemoteMicroscope((self.ip, self.port))  ###########
            print("tem connected in remote mode at: %s, %s" % (self.ip, self.port))
        except Exception as err:
            print('unable to connect the tem at: %s, %s' % (self.ip, self.port), 'error: ', err)

    def microscope_thread_continuous(self, tracking_file):
        # start_angle, target_angle, rotation_speed, experiment_type, tracking_step, tracking_positions,
        pass

    def calc_stage_speed(self, speed):
        """"calculate the speed in degrees/s for the tecnai series, 1 is equivalent to the maximum (normalized).
        speed is provided in degrees/s and return it in rad/s."""
        speed = 200.54 * ((79.882 * (speed ** -1.001)) ** -1.057)
        speed = math.radians(speed)
        if speed > 1:
            speed = 1
        return speed

    def beamshift_tracking(self, experiment_type, tracking_step, tracking_positions):
        pass

    def get_voltage(self):
        pass

    def get_projection_mode(self):
        pass