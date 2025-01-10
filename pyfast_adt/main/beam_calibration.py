import numpy as np
import matplotlib.pyplot as plt
import cv2
# pix_to_beamshift(target_p, angle, scaling_factor) this is the main to use to get the userbeam shift value
# this is initialized in fast_adt_gui.py and used in fast_adt_func calling    self.ub
class BeamCalibration:
    def __init__(self, cam_table):
        self.angle = None
        self.scaling_factor = None
        self.target_p = None
        self.target = None
        self.beam_coordinates = None
        self.cam_table = cam_table

    def input_list(self, beam_coordinates):
        if len(beam_coordinates) != 4:
            raise ValueError("input list must have 4 elements")

        out_ = []
        out_p = []
        for beam_pos, cam_pos in beam_coordinates:
            out_.append(np.asarray(beam_pos))
            out_p.append(np.asarray(cam_pos))

        one_four_vec = out_[3] - out_[0]
        two_three_vec = out_[2] - out_[1]
        one_two_vec = out_[1] - out_[0]
        four_three_vec = out_[2] - out_[3]

        cam_14_p = out_p[3] - out_p[0]
        cam_23_p = out_p[2] - out_p[1]
        cam_12_p = out_p[1] - out_p[0]
        cam_43_p = out_p[2] - out_p[3]

        return one_four_vec, two_three_vec, one_two_vec, four_three_vec, cam_14_p, cam_23_p, cam_12_p, cam_43_p

    def input(self, one, two, three, four):
        """workflow to calibrate the beam shift in temscript space wrt the camera space.
        input the position from temscript of the user beam shift as tuple of 2 elements (x,y)"""
        one = np.array(one)
        two = np.array(two)
        three = np.array(three)
        four = np.array(four)

        one_four_vec = four - one
        two_three_vec = three - two
        one_two_vec = two - one
        four_three_vec = three - four

        return one_four_vec, two_three_vec, one_two_vec, four_three_vec

    def calculate_correlation(self, one_four_vec, two_three_vec, one_two_vec, four_three_vec, camera_size = (1024,1024), cam_14_p = None, cam_23_p = None, cam_12_p = None, cam_43_p = None):
        if cam_14_p is None and cam_23_p is None:
            cam1_p = np.array([camera_size[0] / 4, camera_size[1] / 4])
            cam2_p = np.array([camera_size[0] / 4, (3 * camera_size[1] / 4)])
            cam3_p = np.array([(3 * camera_size[0] / 4), (3 * camera_size[1] / 4)])
            cam4_p = np.array([(camera_size[1] / 4), (3 * camera_size[1] / 4)])

            cam_14_p = cam4_p - cam1_p
            cam_23_p = cam3_p - cam2_p
            cam_12_p = cam2_p - cam1_p
            cam_43_p = cam3_p - cam4_p

        angle_14 = np.rad2deg(np.arccos(np.dot(cam_14_p, one_four_vec) / (np.linalg.norm(cam_14_p) * np.linalg.norm(one_four_vec))))
        scaling_factor_14 = np.linalg.norm(one_four_vec) / np.linalg.norm(cam_14_p)
        print("angle_14:", angle_14)
        print("scaling_14:", scaling_factor_14)

        angle_23 = np.rad2deg(np.arccos(np.dot(cam_23_p, two_three_vec) / (np.linalg.norm(cam_23_p) * np.linalg.norm(two_three_vec))))
        scaling_factor_23 = np.linalg.norm(two_three_vec) / np.linalg.norm(cam_23_p)

        print("angle_23:", angle_23)
        print("scaling_23:", scaling_factor_23)

        angle_12 = np.rad2deg(np.arccos(np.dot(cam_12_p, one_two_vec) / (np.linalg.norm(cam_12_p) * np.linalg.norm(one_two_vec))))
        scaling_factor_12 = np.linalg.norm(one_two_vec) / np.linalg.norm(cam_12_p)
        print("angle_12:", angle_12)
        print("scaling_12:", scaling_factor_12)
        angle_12 -= 90

        angle_43 = np.rad2deg(np.arccos(np.dot(cam_43_p, four_three_vec) / (np.linalg.norm(cam_43_p) * np.linalg.norm(four_three_vec))))
        scaling_factor_43 = np.linalg.norm(four_three_vec) / np.linalg.norm(cam_43_p)
        print("angle_43:", angle_43)
        print("scaling_43:", scaling_factor_43)
        print("to check if the angle is correct, the angle between 14 and 43 should be +90 degrees")
        angle_43 -= 90

        mean_angle_y = np.mean([angle_14, angle_23])
        mean_scaling_y = np.mean([scaling_factor_14, scaling_factor_23])

        mean_angle_x = mean_angle_y - (np.mean([angle_12, angle_43]))
        mean_scaling_x = np.mean([scaling_factor_12, scaling_factor_43])


        print("mean angle_x:", mean_angle_x)
        print("mean scaling_x:", mean_scaling_x)
        print("mean angle_y:", mean_angle_y)
        print("mean scaling_y:", mean_scaling_y)

        self.angle_x = mean_angle_x
        self.scaling_factor_x = mean_scaling_x
        self.angle_y = mean_angle_y
        self.scaling_factor_y = mean_scaling_y

        result = [angle_14, angle_23, scaling_factor_14, scaling_factor_23, mean_angle_x, mean_scaling_x, mean_angle_y, mean_scaling_y]
        return mean_angle_x, mean_scaling_x, mean_angle_y, mean_scaling_y, result

    def pix_to_beamshift(self, target_p, angle_x, scaling_factor_x, angle_y, scaling_factor_y):
        self.target_p = target_p
        target = self.apply_rotation(np.asarray(target_p) * scaling_factor_y, theta=-angle_y, integer=False)
        #target = self.apply_affine(np.asarray(target) * 1, phi=-angle_x, integer=False)
        self.target = target
        return target
    def beamshift_to_pix(self, target, angle_x, scaling_factor_x, angle_y, scaling_factor_y):
        self.target = target
        target_p = self.apply_rotation(np.asarray(target) / scaling_factor_y, theta=angle_y, integer=False)
        #target_p = self.apply_affine(np.asarray(target_p) / 1, phi=angle_x, integer=False)
        self.target_p = target_p
        return target_p

    def apply_rotation(self, vector, theta=216.4, integer = True):
        # input an x,y vector and return a rotated vector of theta degree,
        # in order to syncronize the movement of the beam shift with the user reference
        vector = np.asarray(vector)
        theta = np.radians(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_vector = np.dot(rotation_matrix, vector)
        if integer != True:
            return tuple(rotated_vector)
        else:
            return tuple(rotated_vector.astype(int))

    def apply_affine(self, vector, phi=216.4, integer = True):
        # input an x,y vector and return an affine sheared vector of phi degree, this apply only shear on the x axis (BS)
        vector = np.asarray(vector)
        phi = np.radians(phi)
        rotation_matrix = np.array([[1, np.tan(phi)], [0, 1]])
        rotated_vector = np.dot(rotation_matrix, vector)
        if integer != True:
            return tuple(rotated_vector)
        else:
            return tuple(rotated_vector.astype(int))

    def list_pix_to_beamshift(self, list_target_p = None, angle_x = 0, scaling_factor_x = 1, angle_y = 0, scaling_factor_y = 1, default = True):
        if default == True:
            list_target_pix = [(0, 512), (0, 256), (0, 0), (0, -256), (0, -512), (512, 0), (256, 0), (0, 0), (-256, 0),
                             (-512, 0)]
        else: list_target_pix = list_target_p

        list_target = []
        for target_pix in list_target_pix:
            target = self.apply_rotation(np.asarray(target_pix) * scaling_factor_y, theta=angle_y, integer=False)
            #target = self.apply_affine(np.asarray(target) * 1, phi=angle_x, integer=False)
            list_target.append(target)

        return list_target, list_target_pix


    #############################################################################################3#### using also the pixel position
    def run_from_list(self, beam_coordinates = None, camera_size = None):
        if beam_coordinates == None:
            beam_coordinates = [([2.094709377873173e-06, -3.1028543454116137e-07], (257, 257)),
             ([-7.193106921013629e-07, -2.329228235050808e-06], (769, 256)),
             ([-2.81322915981069e-06, 3.83095468695652e-07], (768, 768)),
             ([6.587567081341254e-09, 2.3967952780632976e-06], (257, 768))]

        one_four_vec, two_three_vec, one_two_vec, four_three_vec, cam_14_p, cam_23_p, cam_12_p, cam_43_p = self.input_list(beam_coordinates)

        angle_x, scaling_factor_x, angle_y, scaling_factor_y, res = self.calculate_correlation(one_four_vec, two_three_vec, one_two_vec, four_three_vec, cam_14_p = cam_14_p, cam_23_p= cam_23_p, cam_12_p= cam_12_p, cam_43_p= cam_43_p, camera_size=camera_size)
        list_target, _ = self.list_pix_to_beamshift(angle_x = angle_x, scaling_factor_x= scaling_factor_x, angle_y = angle_y, scaling_factor_y= scaling_factor_y, default = True)

        self.angle_x = angle_x
        self.scaling_factor_x = scaling_factor_x
        self.angle_y = angle_y
        self.scaling_factor_y = scaling_factor_y
        print("parameters for calibration saved")
        #print(list_target)

        # scale = 2e-5
        # plt.quiver(one_four_vec[0], one_four_vec[1], scale = scale, color = "b")
        # plt.quiver(two_three_vec[0], two_three_vec[1], scale = scale, color = "b")
        # scale = 2000
        # plt.quiver(cam_14_p[0], cam_14_p[1], scale = scale, color = "c")
        # plt.quiver(cam_23_p[0], cam_23_p[1], scale = scale, color = "c")
        # ###############
        # scale = 2e-5
        # plt.quiver(list_target[0][0], list_target[0][1], scale = scale, color = "r")
        # plt.quiver(list_target[5][0], list_target[5][1], scale = scale, color = "g")
        #
        # plt.show()
        return angle_x, scaling_factor_x, angle_y, scaling_factor_y, res, list_target



