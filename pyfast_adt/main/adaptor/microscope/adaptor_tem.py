from abc import ABC, abstractmethod

class Tem_base(ABC):
    """every angle for moving the satge must be in deg as input and output, velocity for the stage in radian/s and um for the stage xyz movement"""
    def __init__(self):
        self.var = None
        self.x = 0
        self.y = 0
        self.beam_intensity_1 = None
        self.beam_intensity_2 = None

    @abstractmethod
    def move_stage_up(self, stage_ampl):
        '''' take a float as stage ampl (um) and move the stage on +y of that quantity, rotation and scale to um already inside'''
        pass

    @abstractmethod
    def move_stage_down(self, stage_ampl):
        pass

    @abstractmethod
    def move_stage_left(self, stage_ampl):
        pass

    @abstractmethod
    def move_stage_right(self, stage_ampl):
        pass

    @abstractmethod
    def move_stage_z_up(self, stage_ampl):
        pass

    @abstractmethod
    def move_stage_z_down(self, stage_ampl):
        pass

    @abstractmethod
    def move_beam_up(self, beam_ampl):
        pass
    @abstractmethod
    def move_beam_down(self, beam_ampl):
        pass

    @abstractmethod
    def move_beam_left(self, beam_ampl):
        pass

    @abstractmethod
    def move_beam_right(self, beam_ampl):
        pass

    @abstractmethod
    def diffraction(self, checked_diff_value, kl = None):
        pass

    @abstractmethod
    def euc_focus(self):
        pass

    @abstractmethod
    def wobbler(self, checked_wobbler_value):
        pass

    @abstractmethod
    def move_screen(self, checked_screen_value):
        pass

    @abstractmethod
    def get_screen_position(self):
        pass

    @abstractmethod
    def beam_blank(self, checked_blank_value):
        pass

    @abstractmethod
    def apply_rotation(self, vector, theta=212):
        return (self.x, self.y)

    @abstractmethod
    def fake(self):
        print("i'm a fake function used as placeholder from the abstract class")

    @abstractmethod
    def get_stage(self, standard = False):
        return dict(zip(['x','y','z','a','b'], [0.0, 0.0, 0.0, 0.0, 0.0]))

    @abstractmethod
    def set_stage_position(self, x = None, y = None, z = None, a = None, b = None):
        pass

    @abstractmethod
    def get_beam_shift(self):
        return (0.0, 0.0)

    @abstractmethod
    def set_beam_shift(self, beam_pos):
        pass

    @abstractmethod
    def connect(self, ip, port):
        pass
    @abstractmethod
    def get_projection_mode(self):
        pass

    @abstractmethod
    def set_magnification(self, mag_value):
        pass

    @abstractmethod
    def get_magnification(self):
        pass

    @abstractmethod
    def set_KL(self, kl_value):
        pass

    @abstractmethod
    def get_KL(self):
        pass

    @abstractmethod
    def get_intensity(self, slot=0):
        """slot: 0,1,2, where 0 is not store the c2% value, 1 is store in beam_intensity_1, and 2 in the beam_intensity_2"""
        pass

    @abstractmethod
    def set_intensity(self, intensity=0, slot=0):
        pass

    @abstractmethod
    def load_calibration_table(self, cam_table):
        pass

    @abstractmethod
    def set_alpha(self, angle, velocity = 1):
        pass

    @abstractmethod
    def microscope_thread_setup(self, tracking_file = "tracking.txt", tracking_dict = None):
        # start_angle, target_angle, rotation_speed, experiment_type, tracking_step, tracking_positions,
        pass

    @abstractmethod
    def calc_stage_speed(self, speed):
        pass
    @abstractmethod
    def beamshift_tracking(self, tracking_dict, result):
        pass
    @abstractmethod
    def beamshift_tracking_stem(self, tracking_dict, result):
        pass
    @abstractmethod
    def get_voltage(self):
        pass

    @abstractmethod
    def get_illumination_mode(self):
        pass
    @abstractmethod
    def set_illumination_mode(self, mode):
        pass

    @abstractmethod
    def get_instrument_mode(self):
        pass

    @abstractmethod
    def set_instrument_mode(self, mode):
        pass