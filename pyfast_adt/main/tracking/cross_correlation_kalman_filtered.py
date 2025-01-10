# test KF CC correction
# initial linear KF model used ref. here:
# https://github.com/RahmadSadli/Kalman-Filter/blob/master/KalmanFilter.py
# https://faculty.sist.shanghaitech.edu.cn/faculty/luoxl/class/2017Fall_EE251/ClassProjects/Xiaohe_He/EE251_final_project_XiaoheHe_48655395.pdf
# more advanced KF methods are from the package filterpy and its relative pdf book; ref here:
# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
# https://drive.google.com/file/d/0By_SW19c1BfhSVFzNHc0SjduNzg/view?usp=sharing&resourcekey=0-41olC9ht9xE3wQe2zHZ45A

import cv2, os, math
import numpy as np
import matplotlib.pyplot as plt
#from py_utility.kalman_filter_rahmadsadli.KalmanFilter import KalmanFilter
try:
    from .KalmanFilter import KalmanFilter
except:
    from KalmanFilter import KalmanFilter
from copy import deepcopy
import pyautogui
from tkinter import Tk
from tkinter.filedialog import asksaveasfilename
from tkinter import messagebox
from filterpy.common import Q_discrete_white_noise

class Tomography_tracker:
    """this is the core of the patchworkCC algorithm, main method run the tracking asking the user to selct a ROI to
    track, and return the tracked positions as a list of tuples (predicted_pos (pure KF), template_match (CC),
    filtered_pos (CC corrected by KF))) just to clarify:

    --self.CC_positions is brutally pure CC
    --self.template_matching_result is CC with on top the patchwork to remove ambiguities
    --predicted_pos is predicted by pureKF using self.template_matching_result to learn(newton law) (-1 iteration)
    --filtered_pos is template_matching res. corrected by the KF (CC+newton) == more math

    the self.support1 contain the tracked positions of all the methods.

    this method is based on a liner KF at constant velocity where the acceleration act as control input if not
    overwritten in the init step. to overwrite with another KF model after construct the class should be possible
    to just assign self.KF to another KF model.
    if intersted check at the bottom the function self.select_other_KF_model(KF_model = your_model)"""

    def __init__(self, images = None, visualization = False, existing_roi = None, dt = 0.1, exp_type = "continuous"):
        print("dt kf= ", dt)
        self.KF = KalmanFilter(dt, 0.1, 0.1, 0.1, 0.01, 0.01)
        self.KF_lost_position = []
        self.KF_corrected_position = []
        self.lost_counter = 0
        self.support1 = []
        self.series = images
        self.n = None
        self.backup_roi = None
        self.series_support = []
        self.support_manual = []
        self.exp_type = exp_type
        self.list_templates = []

        self.counter_img = None

        # advanced options
        self.visualization = visualization
        self.existing_roi = existing_roi
        # these need to be imported from the GUI
        self.initial_angle, self.last_angle, self.angle_step = -65, 65, 5
        # self.custom_model = [is a custom model?, "name_model_here", required_datapoints]
        self.custom_model = [False, "linear_KF_2D", 0]


    def main(self):
        self.res_distances = []
        if type(self.series) == list and len(self.series) > 0:                                                                                          # load all the images in a buffer
            for frame in self.series:
                self.img = cv2.imread(frame)
                self.img = cv2.normalize(self.img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.series_support.append(self.img)
        elif type(self.series) == np.ndarray:
            for frame in self.series:
                self.img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.series_support.append(self.img)

        # check if a custom model is loaded and if some information are necessary to start properly the model:
        # if self.custom_model[0] == True and self.custom_model[2] > 0:
        if self.custom_model[0] == True:
            res = []
            self.user_defined_ROI(img = self.series_support[0])
            self.existing_roi = self.template
            self.orig_template = self.template.copy()
            # find the center of the ROI
            x,y = int(self.roi[0] + self.roi[2] / 2), int(self.roi[1] + self.roi[3] / 2)
            res.append([x,y]) # append the first result from the ROI of the user

            for iiii in range(self.custom_model[2]):
                # iterate over the next images and perform basic CC on the template to guess the initial parameters for the KF
                self.img = cv2.bilateralFilter(self.series_support[iiii+1], 9, 150, 150)
                self.match = cv2.matchTemplate(self.img, self.template, cv2.TM_CCOEFF_NORMED)
                min_val, self.max_val, min_loc, self.max_loc = cv2.minMaxLoc(self.match)
                self.y, self.x = np.array(self.template.shape[:2]) / 2
                self.template_matching_result = [self.max_loc[0] + self.x, self.max_loc[1] + self.y] # new center of the object found in the CC
                res.append(self.template_matching_result)

            # set up parameters for the following KFs
            x_std_P, y_std_P = 50., 50.
            x_std_R, y_std_R = 2., 2. # trust measurements
            noise_std_Q = 5.
            vel_std_P, acc_std_P = 50., 50.

            if self.custom_model[2] == 0:
                # set that you need to update the state vector with initial pos based on the initial result of the CC
                x0, y0 = res[0][0], res[0][1]
                self.KF.kf.x = np.array([x0, y0], dtype=float)
                self.KF.kf.P = np.diag([x_std_P ** 2, y_std_P ** 2])
                self.KF.kf.R = np.diag([x_std_R ** 2, y_std_R ** 2])
                self.KF.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1., var=noise_std_Q**2)

            elif self.custom_model[2] == 1:
                # set that you need to update the state vector with initial pos and velocity based on the initial result of the CC
                x0, y0 = res[0][0], res[0][1]
                velx0, vely0 = res[1][0] - res[0][0], res[1][1] - res[0][1]
                self.KF.kf.x = np.array([x0, velx0, y0, vely0], dtype=float)
                self.KF.kf.P = np.diag([x_std_P ** 2, vel_std_P ** 2, y_std_P ** 2, vel_std_P ** 2])
                self.KF.kf.R = np.diag([x_std_R ** 2, y_std_R ** 2])
                self.KF.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1., var=noise_std_Q ** 2)
                self.KF.kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1., var=noise_std_Q ** 2)


            elif self.custom_model[2] == 2:
                # set that you need to update the state vector with initial pos, velocity and acc.
                x0, y0 = res[0][0], res[0][1]
                velx0, vely0 = res[1][0] - res[0][0], res[1][1] - res[0][1]
                accx0, accy0 = (res[2][0] - res[1][0]) - (res[1][0] - res[0][0]) , (res[2][1] - res[1][1]) - (res[1][1] - res[0][1])
                self.KF.kf.x = np.array([x0, velx0, accx0, y0, vely0, accy0], dtype=float)
                self.KF.kf.P = np.diag([x_std_P ** 2, vel_std_P ** 2, acc_std_P ** 2, y_std_P ** 2, vel_std_P ** 2, acc_std_P ** 2])
                self.KF.kf.R = np.diag([x_std_R ** 2, y_std_R ** 2])
                self.KF.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=1., var=noise_std_Q ** 2)
                self.KF.kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=1., var=noise_std_Q ** 2)
                self.KF.kf.Q[4:6, 4:6] = Q_discrete_white_noise(2, dt=1., var=noise_std_Q ** 2)


                # Iterate over the frames
        for self.n, frame in enumerate(self.series_support):
            self.CC_positions = []
            print("\ncompute image: ", self.n+1, "/", len(self.series))
            self.img = frame

            if self.n != 0:
                self.img = cv2.bilateralFilter(self.img, 9, 150, 150)
                #self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
            if self.n == 0:                                                                                                                             # ask the user to select a ROI in the image
                self.user_defined_ROI(img = self.img)
                self.orig_template = self.template.copy()

            # Perform template matching to obtain object position in the current frame
            try:                                                                                                                                            # perform CC using the template
                self.match = cv2.matchTemplate(self.img, self.template, cv2.TM_CCOEFF_NORMED)
            except cv2.error as err:
                print(err)
                print("target lost, try manual mode or adjust eucentric height")
                break

            # fake out max_val for first run through loop
            self.match2 = self.match.copy()
            self.max_list = []
            self.thresholds = []
            if self.n == 0:
                min_val, self.max_val, min_loc, self.max_loc = cv2.minMaxLoc(self.match)
                self.max_list.append(self.max_loc)
                self.thresholds.append(self.max_val)
                self.max_val_threshold = self.thresholds[0]

            self.threshold1 = self.max_val_threshold * 0.05
            print("peaks_thresh range:",  1.0 , "-", round((self.max_val_threshold - self.threshold1), 3))
            self.y, self.x = np.array(self.template.shape[:2]) / 2

            # here search for all the maximum of the CC map using a patchworkCC approach
            if self.n != 0:                                                                                                                                 # find the first top 10 maxima in the CC using patchworkCC
                i = 0
                # to let it enter the cycle
                self.max_val = 1
                while self.max_val > self.threshold1:
                    self.stopping = False
                    min_val, self.max_val, min_loc, self.max_loc = cv2.minMaxLoc(self.match2)
                    if self.visualization == True:
                        print("max found at: ", self.max_loc[0] + self.x, self.max_loc[1] + self.y, "value", round(self.max_val,2))
                    self.max_list.append(self.max_loc)
                    self.thresholds.append(self.max_val)

                    # using the max coordinates, fill in that area with a patchwork of zeros (i.e. min of the CCmap)
                    y_1 = self.max_loc[1] - int(self.y)
                    if y_1 < 0: y_1 = 0
                    y_2 = self.max_loc[1] + int(self.y)
                    if y_2 > self.match2.shape[0]: y_2 = self.match2.shape[0]
                    x_1 = self.max_loc[0] - int(self.x)
                    if x_1 < 0: x_1 = 0
                    x_2 = self.max_loc[0] + int(self.x)
                    if x_2 > self.match2.shape[1]: x_2 = self.match2.shape[1]

                    # previous version
                    # match2[max_loc[1]-int(y):max_loc[1] + int(y), max_loc[0]-int(x): max_loc[0] + int(x)] = match2.min()
                    self.match2[y_1:y_2, x_1: x_2] = self.match2.min()

                    if i > 9:
                        if self.visualization == True:
                            print("more than 10 iterations found, stop it")
                        break
                    i += 1

            if self.visualization == True:
                print("number of maximum found in CC map:", len(self.max_list))

            self.max_loc = self.max_list[0]
            self.max_loc_kf = self.max_list[0]
            self.kf_match = self.max_loc_kf[0] + self.x, self.max_loc_kf[1] + self.y
            self.template_matching_result = self.max_loc[0] + self.x, self.max_loc[1] + self.y
            #stored the pureCC result
            self.CC_positions.append(self.template_matching_result)
            # if to check the model

            # Predict the next object position using the Kalman filter
            # Predict
            self.predicted_position = np.asarray(self.KF.predict(), dtype=np.float64)                                                                       # predict using the KF and use it to find the real object position
            if self.predicted_position.shape == (2,2):
                self.predicted_position = self.predicted_position[0, :]
            if self.n == 0:
                self.predicted_position = self.template_matching_result

            # corrected and checked, KF and CC are working the same
            self.templ_ratio = np.max(self.template.shape[:2])/np.min(self.template.shape[:2])
            self.threshold = np.max(self.template.shape[:2]) / 3
            if self.visualization == True:
                print("ROI ratio", self.templ_ratio)
            if self.templ_ratio > 1.5:
                if self.visualization == True:
                    print("ROI not a square like shape, calculate the mean value")
                #if templ_ratio > 2:
                self.threshold = np.mean(self.template.shape[:2]) / 4

            if self.threshold < 30: #before they were 30
                self.threshold = 30
            if self.visualization == True:
                print("\nthreshold: ", self.threshold, "templ shape:", self.template.shape[:2])
            self.distance = math.dist(self.template_matching_result, self.predicted_position)
            print("distance: ", round(self.distance,2), "CC_max", self.template_matching_result, "KF_pred", self.predicted_position)


            self.stop = False
            self.control_list = []
            if self.distance >= self.threshold and len(self.max_list) > 1:
                print("\nstart CC correction loop, searching for the more close maximum to correct ambiguities using KF")
                print("dist. threshold:", self.threshold, "peaks to test", len(self.max_list))
                i = 1
                #while stop == False and i < len(max_list):
                while i < len(self.max_list):
                    self.max_loc_kf = self.max_list[i]
                    self.kf_match = self.max_loc_kf[0] + self.x, self.max_loc_kf[1] + self.y
                    self.distance = math.dist(self.kf_match, self.predicted_position)
                    print("iter", i, ", KF distance: ", round(self.distance,3))
                    self.control_list.append((self.distance, self.kf_match, self.thresholds[i-1]))

                    if self.distance <= self.threshold:
                        self.stop = True

                    i += 1
                if self.stop == False:
                    print("target disappeared, reset to the KF prediction position")
                    self.template_matching_result = self.predicted_position
                    ## added now
                    self.max_loc_kf = self.predicted_position
                    self.KF_lost_position.append((self.n+1, self.predicted_position))
                    self.max_val_threshold = max(self.thresholds)
                    self.lost_counter += 1

                else:
                    # Find the element with the minimum distance
                    self.min_distance_tuple = min(self.control_list, key=lambda x: x[0])
                    ind = self.control_list.index(self.min_distance_tuple)
                    self.kf_match = self.min_distance_tuple[1]
                    self.distance = self.min_distance_tuple[0]
                    self.max_val_threshold = self.min_distance_tuple[2]
                    print("closer peak found at index: ", ind+1, self.kf_match, "distance: ", round(self.distance,2), "\n")
                    self.template_matching_result = self.kf_match
                    self.KF_corrected_position.append((self.n+1, self.kf_match))

            self.res_distances.append(self.distance)
            # Update and store the results to plot it in the end                                                                                           # update the KF with the new position
            update = self.KF.update(self.template_matching_result)
            if self.custom_model[0] == False:
                self.filtered_position = np.asarray(update, dtype=float)[0, :]
            else:
                self.filtered_position = np.asarray(update, dtype=float)

            self.support1.append((tuple(self.predicted_position), tuple(self.template_matching_result), tuple(self.filtered_position), tuple(self.CC_positions[0])))

            # run the plots
            self.plot_single_track(visualization=self.visualization)

            # Update the template for the next iteration
            self.template = self.img[int(self.template_matching_result[1] - self.roi[3] / 2):int(self.template_matching_result[1] + self.roi[3] / 2),      # update the template with the new result
                            int(self.template_matching_result[0] - self.roi[2] / 2):int(self.template_matching_result[0] + self.roi[2] / 2)]

            ### store the template of the first iteration ####
            self.list_templates.append(self.template)

        lost_l = []
        for index, _ in self.KF_lost_position:
            lost_l.append((str(index) + "/" + str(len(self.support1))))
        print("# reseted positions from KF: ", len(self.KF_lost_position), "/", len(self.support1))
        print("frames reseted:", lost_l)
        corr_l = []
        for index, _ in self.KF_corrected_position:
            corr_l.append((str(index) + "/" + str(len(self.support1))))
        print("# corrected positions from KF: ", len(self.KF_corrected_position), "/", len(self.support1))
        print("frame corrected :", corr_l)
        self.support_manual = []
        # ype cast self.support1 to become list(list(tuples()))

        return self.support1

    def user_defined_ROI(self, img):
        self.img = img
        self.img_or = self.img.copy()
        self.img_or_show = self.img.copy()  # added brg colors to the show
        if type(self.series) != list:
            self.img_or_show = cv2.merge([self.img_or_show, self.img_or_show, self.img_or_show])  ###
        self.img = cv2.bilateralFilter(self.img, 9, 150, 150)
        # self.img = cv2.GaussianBlur(self.img, (5, 5), 0)
        # roi, give back always the coordinate of the top_left corner (1/2 tuple) and the width and height (3/4 tuple) of the rectangle
        if self.existing_roi is not None:
            self.template = self.existing_roi
            self.match = cv2.matchTemplate(self.img, self.template, cv2.TM_CCOEFF_NORMED)
            min_val, self.max_val, min_loc, self.max_loc = cv2.minMaxLoc(self.match)
            self.y, self.x = np.array(self.template.shape[:2]) / 2
            self.template_matching_result = [self.max_loc[0] + self.x, self.max_loc[1] + self.y]
            self.roi = [self.max_loc[0], self.max_loc[1], self.template.shape[1], self.template.shape[0]]
        else:
            # put the help text
            anchor = self.img_or_show.shape[1]
            cv2.putText(self.img_or_show,
                        "left click and drag to select a ROI,\nthe target to track will be the center of the crosshair",
                        (int(anchor * 0.02), int(anchor * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                        cv2.LINE_AA)
            self.roi = cv2.selectROI("select roi", self.img_or_show)
            self.backup_roi = deepcopy(self.roi)
            ###########
            cv2.destroyAllWindows()
            self.template = self.img[int(self.roi[1]):int(self.roi[1] + self.roi[3]),
                            int(self.roi[0]):int(self.roi[0] + self.roi[2])]

            ### store the template of the first iteration ####
            if self.n != 0:
                self.list_templates.append(self.template)
                print("debug line to check consintency\n",
                      "template list size:", len(self.list_templates),
                      "imgs number:", len(self.series_support))

    def plot_single_track(self, visualization = False):
        if visualization == False:
            return
        if self.counter_img == None:
            self.counter_img = 1
        # subPlot the results for every image
        # first image is the original frame with the rectangles representing the trackers
        # second image is the template coming from the previous iteration
        # third image is the CC map with the CC maximums found

        fig, ax = plt.subplots(1, 4, figsize=(10, 5))
        fig.suptitle('compute image:%d'%(self.n+1), fontsize=8)
        ax[0].imshow(self.img)
        ax[0].set_title('Input frame')

        ax[1].imshow(self.template)
        ax[1].set_title('Template')
        ax[2].imshow(self.match)
        ax[2].set_title('CC map')
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        ax[3].imshow(self.match2)
        ax[3].set_title('CC map patchwork')
        ax[3].set_xticks([])
        ax[3].set_yticks([])
        rect1 = plt.Rectangle((self.template_matching_result[0]-self.roi[2]/2, self.template_matching_result[1]-self.roi[3]/2), self.roi[2], self.roi[3], color='r', fill=False)
        rect2 = plt.Rectangle((self.predicted_position[0]-self.roi[2]/2, self.predicted_position[1]-self.roi[3]/2), self.roi[2], self.roi[3], color='g', fill=False)

        rect3 = plt.Rectangle((self.max_loc[0]-self.roi[2]/2, self.max_loc[1]-self.roi[3]/2), self.roi[2], self.roi[3], color='r', fill=False)
        rect4 = plt.Rectangle((self.max_loc_kf[0]-self.roi[2]/2, self.max_loc_kf[1]-self.roi[3]/2, ), self.roi[2], self.roi[3], color='g', fill=False)

        circle1 = plt.Circle((self.template_matching_result[0], self.template_matching_result[1]), 5, color='r')
        circle2 = plt.Circle((self.predicted_position[0], self.predicted_position[1]), 5, color='g')
        circle3 = plt.Circle((self.filtered_position[0], self.filtered_position[1]), 5, color='b')

        circle4 = plt.Circle(self.max_loc, 5, color='r')
        circle5 = plt.Circle(self.max_loc_kf, 5, color='g')

        ax[0].add_patch(circle1)
        ax[0].add_patch(circle2)
        ax[0].add_patch(circle3)
        ax[1].legend(["r= template matching result", "g = predicted position", "u = filtered position"], fontsize = 'small', loc='upper right')
        ax[0].add_patch(rect1)
        ax[0].add_patch(rect2)

        ax[2].add_patch(circle4)
        ax[2].add_patch(circle5)
        ax[2].add_patch(rect3)
        ax[2].add_patch(rect4)
        if self.visualization == True:
            plt.savefig(r"L:\Marco\datasets\pyfastadt_tracking_test\tracking_precision_25022024\tracking_precision_crystal\CC_images\CC_correction_img%s.jpg" %str(self.counter_img).rjust(4, '0'))
            self.counter_img += 1
        plt.show()

    def plot_tracking(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.img_or)
        if len(self.support1) != 0:
            ax[0].plot([x[0][0] for x in self.support1], [x[0][1] for x in self.support1], linestyle = "dashed", color = 'g', label='Pure KF prediction')
            ax[0].plot([x[1][0] for x in self.support1], [x[1][1] for x in self.support1], linewidth = 2, linestyle = "dotted", color = 'black', label='patchworkCC result')
            ax[0].plot([x[2][0] for x in self.support1], [x[2][1] for x in self.support1], 'b', label='KF filtered position')
            ax[0].plot([x[3][0] for x in self.support1], [x[3][1] for x in self.support1], linestyle="dotted", color='r', label='Pure CC')

        if len(self.support_manual) != 0:
            ax[0].plot([x[0][0] for x in self.support_manual], [x[0][1] for x in self.support_manual], 'y', label='Manual')
            ax[1].plot([x[0][0] for x in self.support_manual], [x[0][1] for x in self.support_manual], 'y', marker = '^', label='Manual')

        if len(self.support1) != 0:
            ax[1].plot([x[0][0] for x in self.support1], [x[0][1] for x in self.support1], linestyle="dashed", color='g', marker='x',label='Pure KF prediction')
            ax[1].plot([x[1][0] for x in self.support1], [x[1][1] for x in self.support1], linewidth = 2, linestyle = "dotted", color = 'black', marker='o',label='patchworkCC result')
            ax[1].plot([x[2][0] for x in self.support1], [x[2][1] for x in self.support1], 'b', marker='*',label='KF Filtered position')
            ax[1].plot([x[3][0] for x in self.support1], [x[3][1] for x in self.support1], linestyle="dotted", marker='+',color='r', label='Pure CC')
            ax[1].invert_yaxis()
        for nn, data_ in enumerate(self.support1):
            #annotate patchworkCC result and KF filtered position
            ax[1].annotate(str(nn + 1), xy=(data_[1][0], -data_[1][1]))
            ax[1].annotate(str(nn + 1), xy=(data_[2][0], -data_[2][1]))

        if len(self.support_manual) == 0:
            col = 4
        elif len(self.support_manual) != 0 and len(self.support1) != 0:
            col = 5
        elif len(self.support_manual) != 0 and len(self.support1) == 0:
            col = 1

        ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1.20), ncol=col, fancybox=True, shadow=True)
        plt.show()

        if len(self.res_distances) > 0:
            frames_ = np.arange(0, len(self.res_distances))
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
            ax2.plot(frames_, self.res_distances)
            ax2.set_title("distance between KF predict and CC corrected position")
            ax2.set_xlabel("frames")
            ax2.set_ylabel("distance in pixels")
            plt.show()

    def plot_tracking_reevaluation(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(self.img_or)
        if len(self.support1) != 0:
            #ax[0].plot([x[0][0] for x in self.support1], [x[0][1] for x in self.support1], linestyle = "dashed", color = 'g', label='Pure KF prediction')
            ax[0].plot([x[1][0] for x in self.support1], [x[1][1] for x in self.support1], linewidth = 2, linestyle = "None", color = 'blue', label='patchworkCC result', marker='o', markersize = 1)
            #ax[0].plot([x[2][0] for x in self.support1], [x[2][1] for x in self.support1], 'b', label='KF filtered position')
            ax[0].plot([x[3][0] for x in self.support1], [x[3][1] for x in self.support1], linestyle="None", color='r', label='Pure CC', marker='+', markersize = 1)

        if len(self.support_manual) != 0:
            ax[0].plot([x[0][0] for x in self.support_manual], [x[0][1] for x in self.support_manual], 'y', label='Manual')
            ax[1].plot([x[0][0] for x in self.support_manual], [x[0][1] for x in self.support_manual], 'y', marker = '^', label='Manual')

        if len(self.support1) != 0:
            #ax[1].plot([x[0][0] for x in self.support1], [x[0][1] for x in self.support1], linestyle="dashed", color='g', marker='x',label='Pure KF prediction')
            ax[1].plot([x[1][0] for x in self.support1], [x[1][1] for x in self.support1], linewidth = 2, linestyle = "None", color = 'blue', marker='o',label='patchworkCC result', markersize = 5)
            #ax[1].plot([x[2][0] for x in self.support1], [x[2][1] for x in self.support1], 'b', marker='*',label='KF Filtered position')
            ax[1].plot([x[3][0] for x in self.support1], [x[3][1] for x in self.support1], linestyle="None", marker='+',color='r', label='Pure CC', markersize = 5)
            ax[1].invert_yaxis()
        # for nn, data_ in enumerate(self.support1):
        #     #annotate patchworkCC result and KF filtered position
        #     ax[1].annotate(str(nn + 1), xy=(data_[1][0], -data_[1][1]))
        #     ax[1].annotate(str(nn + 1), xy=(data_[2][0], -data_[2][1]))

        if len(self.support_manual) == 0:
            col = 4
        elif len(self.support_manual) != 0 and len(self.support1) != 0:
            col = 5
        elif len(self.support_manual) != 0 and len(self.support1) == 0:
            col = 1

        ax[0].legend(loc='upper left', bbox_to_anchor=(0, 1.20), ncol=col, fancybox=True, shadow=True)
        plt.show()

        if len(self.res_distances) > 0:
            frames_ = np.arange(0, len(self.res_distances))
            fig2, ax2 = plt.subplots(1, 1, figsize=(10, 5))
            ax2.plot(frames_, self.res_distances)
            ax2.set_title("distance between KF predict and CC corrected position")
            ax2.set_xlabel("frames")
            ax2.set_ylabel("distance in pixels")
            plt.show()
        return fig, ax, self.support1

    def save_tracking(self, datapoints):
        # parameters for the tracking file writing check!
        if not self.initial_angle or not self.last_angle or not self.angle_step:
            messagebox.showerror("Missing Info", "Please set the tracking parameters first")
            return False
        # write the tracking file
        self.output = asksaveasfilename(filetypes=[("txt file", ".txt")], defaultextension=".txt", title="Save crystal tracking file", initialfile = "crystal_tracking_file.txt")
        # self.output = r"L:\Marco\hardware_microscopes\TecnaiF30\sergi_track\Tracking\crystal_tracking_file.txt"

        header = "Initial Angle: %s \nLast Angle: %s \ntracking every %s degree \nX, Y position in pixels" %(str(self.initial_angle), str(self.last_angle), str(self.angle_step))
        # np.savetxt(self.output, tracking_xy, header = header, comments="",  delimiter = " , ", newline= "\n",fmt = "%.0f")
        np.savetxt(self.output, datapoints, header=header, comments="", delimiter=" , ", newline="\n", fmt="%.0f")
        return True

    # manual tracking section
    def draw_circle(self, event, x, y, flags, param): #this function must have this argument to work
        if event == cv2.EVENT_LBUTTONDOWN:
            if param == True:
                self.img_copy1 = self.img.copy()
                cv2.circle(self.img_copy1, (x, y), 15, (0, 0, 255), 2)
                cv2.imshow("Select object", self.img_copy1)
            else:
                cv2.circle(self.img, (x, y), 15, (0, 0, 255), 2)
            #cv2.putText(self.img, f"({x},{y})", (x + 17, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.support_manual.append(((x,y),(x+0.01,y+0.01)))
            #cv2.imshow("Select object", self.img)
            pyautogui.press("space")

    def plot_result(self, color = (255,0,0)):
        self.x = []
        self.y = []
        self.y1 = []
        for res in self.support_manual:
            self.x.append(res[0][0] + int(round((-res[0][0]+res[1][0])/2, 0)))
            self.y1.append(-(res[0][1] + int(round((-res[0][1]+res[1][1])/2, 0))))
            self.y.append((res[0][1] + int(round((-res[0][1] + res[1][1]) / 2, 0))))

        n = np.linspace(1, len(self.x), len(self.x))
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(self.x, self.y, linewidth = 1, color = "r")
        ax[1].plot(self.x, self.y1, linewidth=1, color="r")

        try:
            ax[0].imshow(self.img_or)
        except AttributeError as err:
            print(err)
            ax[0].imshow(self.img)
        for k, i in enumerate(n):
            i = int(i)
            ax[1].annotate(i, (self.x[k], self.y[k]), color = "r")
        plt.show()

        cv2.destroyAllWindows()
        return (self.support_manual, self.x, self.y)

    def manual_tracking(self, images=None, visualization = False):
        self.series_support_manual = []
        if type(images) == list and len(images) > 0:
            for frame in images:
                self.img = cv2.imread(frame)
                self.img = cv2.normalize(self.img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.img = cv2.merge([self.img, self.img, self.img])  ###
                self.series_support_manual.append(self.img)
        elif type(images) == np.ndarray:
            for frame in images:
                self.img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.img = cv2.merge([self.img, self.img, self.img])  ###
                self.series_support_manual.append(self.img)

        self.support_manual = []
        self.i = 0

        for n, frame in enumerate(self.series_support_manual):
            index_ = "img %s / %s" %(str(n+1), str(len(self.series_support_manual)))
            print(index_)
            self.img = frame
            #r = roi return [int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            try:
                # Define the parameters for the rectangle
                x1 = int(round(self.support1[n][2][0] - self.backup_roi[2] / 2))
                y1 = int(round(self.support1[n][2][1] - self.backup_roi[3] / 2))
                x2 = int(round(self.support1[n][2][0] + self.backup_roi[2] / 2))
                y2 = int(round(self.support1[n][2][1] + self.backup_roi[3] / 2))

                # Calculate the center of the rectangle
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Define the color for the crosshair (BGR format)
                color = (0, 0, 255)  # Red in BGR
                # Draw the rectangle on the image
                cv2.rectangle(self.img, (x1, y1), (x2, y2), (255,0,0), 1)

                # put the help text
                anchor = self.img.shape[1]
                cv2.putText(self.img, index_, (int(anchor * 0.015), int(anchor * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(self.img, "please don't use the space bar\nright click: new tracking position\n"
                                      "enter: accept previous tracking position", (anchor - int(anchor * 0.10), anchor - int(anchor * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                #
                # Draw horizontal and vertical lines for the crosshair
                cv2.line(self.img, (cx - int(round(self.backup_roi[2]/4)), cy), (cx + int(round(self.backup_roi[2]/4)), cy), color, 1)
                cv2.line(self.img, (cx, cy - int(round(self.backup_roi[3]/4))), (cx, cy + int(round(self.backup_roi[3]/4))), color, 1)
            except: pass

            self.img_name = "Select object"
            cv2.namedWindow(self.img_name)
            cv2.setMouseCallback(self.img_name, self.draw_circle)
            stop = False
            # while stop == False:

            while stop == False:

                cv2.imshow(self.img_name, self.img)
                #maybe this func_it's overwriting on the list!!
                key = cv2.waitKey(1)

                if key == 13: #if press enter
                    self.support_manual.append(((self.support1[n][2][0],self.support1[n][2][1]),
                                                (self.support1[n][2][0]+0.01,self.support1[n][2][1]+0.01)))
                    stop = True
                    print("accepted previous position")
                elif key == 32: #if press space
                    stop = True
                    print("new position saved")
                else:
                    # 0.33 s for delay to change image after the previous position selection
                    #print("nothing pressed")
                    pass


        cv2.destroyAllWindows()
        if visualization == True:
            ret = self.plot_result()
        else:
            ret = self.support_manual
        return ret

    def display_tracking(self, images, tracking_dict, method, beam_size_diff = None):
        if method == "debug":
            method = "KF"

        data = tracking_dict[str(method)]
        self.series_support_display = []
        if type(images) == list and len(images) > 0:
            for frame in images:
                self.img = cv2.imread(frame)
                self.img = cv2.normalize(self.img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.img = cv2.merge([self.img, self.img, self.img])  ###
                self.series_support_display.append(self.img)
        elif type(images) == np.ndarray:
            for frame in images:
                self.img = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                self.img = cv2.merge([self.img, self.img, self.img])  ###
                self.series_support_display.append(self.img)

        self.support_display = []
        self.i = 0

        for n, frame in enumerate(self.series_support_display):
            index_ = "img %s / %s" %(str(n+1), str(len(self.series_support_display)))
            print(index_)
            self.img = frame
            #r = roi return [int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            # Define the parameters for the rectangle
            x1 = int(round(data[n][0] - self.backup_roi[2] / 2))
            y1 = int(round(data[n][1] - self.backup_roi[3] / 2))
            x2 = int(round(data[n][0] + self.backup_roi[2] / 2))
            y2 = int(round(data[n][1] + self.backup_roi[3] / 2))

            # Calculate the center of the rectangle
            cx = int(round(data[n][0]))
            cy = int(round(data[n][1]))

            # Define the color for the crosshair (BGR format)
            color = (0, 0, 255)  # Red in BGR
            # Draw the rectangle on the image
            #cv2.rectangle(self.img, (x1, y1), (x2, y2), 255, 1)

            # Draw horizontal and vertical lines for the crosshair
            cv2.line(self.img, (cx - int(round(self.backup_roi[2]/4)), cy), (cx + int(round(self.backup_roi[2]/4)), cy), color, 2)
            cv2.line(self.img, (cx, cy - int(round(self.backup_roi[3]/4))), (cx, cy + int(round(self.backup_roi[3]/4))), color, 2)
            # put the help text
            anchor = self.img.shape[1]
            cv2.putText(self.img, index_, (int(anchor * 0.015), int(anchor * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(self.img, "enter: next image", (anchor - int(anchor*0.10), anchor - int(anchor*0.05)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            # put the beam size is present!
            if beam_size_diff != None:
                cv2.circle(self.img, (cx, cy), int(round(beam_size_diff/2)), color, 2)

            cv2.namedWindow("Tracking displayed: %s" %str(method))
            cv2.imshow("Tracking displayed: %s" %str(method), self.img)

            if self.exp_type == "stepwise":
                cv2.waitKey(500)
            elif self.exp_type == "continuous":
                cv2.waitKey(300)

        cv2.destroyAllWindows()

    def select_other_KF_model(self, KF_model = None, KF_from_list = None):
        """function to overwrite the KF model used. the KF should be a class with at least 2 functions predict
        and update to work properly. a custom KF model can be passed in the argument KF_model (a class to construct).
        otherwise you can use the argument KF_from_list to select an already implemented KF model.
        the function construct the class for you. as it's builded the experiment up to now the KF model should be using
        as measurement for the update step only the x,y position obtained from an object detection method."""

        try: from .KF_custom_models import ukf_2D, ukf_4D, ukf_6D
        except: from KF_custom_models import ukf_2D, ukf_4D, ukf_6D

        KF_models_available = {"ukf_2D": ukf_2D, "ukf_4D": ukf_4D, "ukf_6D": ukf_6D}

        if KF_model != None:
            self.KF = KF_model()
        elif KF_from_list != None:
            if KF_from_list == "ukf_2D":
                self.KF = KF_models_available[KF_from_list](dt=1, x_std_P=500, y_std_P=500, x_std_R=100, y_std_R=100, noise_std_Q=100, pos_x0=0, pos_y0=0)
                required_datapoints = 0 # necessary only the initial position x0, y0
                self.custom_model = [True, KF_from_list, required_datapoints]
                return self.KF
            elif KF_from_list == "ukf_4D":
                self.KF = KF_models_available[KF_from_list](dt=1, x_std_P=500, y_std_P=500, x_std_R=100, y_std_R=100, noise_std_Q=100, pos_x0=0, pos_y0=0, vel_x0=0., vel_y0=0.)
                required_datapoints = 1  # necessary only the initial position x0, y0, vel_x0, vel_y0
                self.custom_model = [True, KF_from_list, required_datapoints]
                return self.KF
            elif KF_from_list == "ukf_6D":
                self.KF = KF_models_available[KF_from_list](dt=1, x_std_P=500, y_std_P=500, x_std_R=100, y_std_R=100, noise_std_Q=100, pos_x0=0, pos_y0=0, vel_x0=0., vel_y0=0., acc_x0=0, acc_y0=0, vel_std_P=10., acc_std_P=10.)
                required_datapoints = 2  # necessary only the initial position x0, y0, vel_x0, vel_y0, acc_x0, acc_y0
                self.custom_model = [True, KF_from_list, required_datapoints]
                return self.KF



######################################
#workflow
#assign_images, init_the_class track = Tomography_tracker(series = list(paths), visualization=False)
#or track = Tomography_tracker(images = np.array(3D), visualization=False)

#run automatic_res = track.main(),
#plot result = track.plot_tracking() just last plot

#if you want the manual check
#manual_res = track.manual_tracking(list(paths), visualization = False)
######################################
if __name__ == "__main__":
    ### init data   ###
    # most easy case dataset 18!
    #path1 = r"L:\Marco\hardware_microscopes\TecnaiF30\sergi_track\Tracking\Tomography\Sequential\16\clean"
    path1 = r"L:\Marco\hardware_microscopes\TecnaiF30\sergi_track\Tracking\Tomography\Sequential\18\clean"
    #path1 = r"L:\Marco\hardware_microscopes\TecnaiF30\sergi_track\Tracking\Tomography\Sequential\21\clean"
    series1 = os.listdir(path1)
    series1.sort()
    series2 = []
    series2 = [path1+os.sep+name for name in series1]
    ######################################
    #main class instance
    manual_comparison = True
    track = Tomography_tracker(images = series2, visualization= True)
    # main code loop over the images
    result_to_check = track.main()
    # plot overall tracking path
    track.plot_tracking()
    # save tracking file automatic
    datapoints = []
    for _,_, data in track.support1: datapoints.append(tuple(data))
    track.save_tracking(datapoints)

    # manual tracking
    if manual_comparison == True:
        # automatic check
        automatic_res = track.support1.copy()
        # manual check
        manual_res = track.manual_tracking(images = series2, visualization = True)
        # plot manual

        # comparison
        img0 = cv2.normalize(cv2.imread(series2[0]), None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        img1 = cv2.normalize(cv2.imread(series2[int(len(series2)/2)]), None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)
        img2 = cv2.normalize(cv2.imread(series2[-1]), None, 0, 255, cv2.NORM_MINMAX, dtype = cv2.CV_8U)

        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        calibration = 0.011439 # nm/px
        plotter_x = []
        plotter_y = []
        plotter_y1= []
        plotter_xum = []
        plotter_yum = []
        plotter_y1um= []
        for _,_, (x,y) in automatic_res:
            plotter_x.append(x)
            plotter_y.append(y)
            plotter_y1.append(-y)
            plotter_xum.append(x*calibration)
            plotter_yum.append(y*calibration)
            plotter_y1um.append(-y*calibration)
        ax[0].imshow(img0)
        ax[0].plot(plotter_x, plotter_y, color = "b" ,label = "auto_tracking", linewidth = 0.5)
        ax[0].set_title("starting_img(max_tilt)")
        ax[1].imshow(img1)
        ax[1].plot(plotter_x, plotter_y, color = "b" ,label = "auto_tracking", linewidth = 0.5)
        ax[1].set_title("middle_img(0deg)")
        ax[2].imshow(img2)
        ax[2].plot(plotter_x, plotter_y, color = "b" ,label = "auto_tracking", linewidth = 0.5)
        ax[2].set_title("ending_img(max_tilt)")
        ax[3].plot(plotter_xum, plotter_y1um, color = "b" ,label = "auto_tracking")

        plotter_x = []
        plotter_y = []
        plotter_y1= []
        plotter_xum = []
        plotter_yum = []
        plotter_y1um= []
        # to save the manual tracking data
        datapoints = []

        for (x, y), _ in manual_res[0]:
            plotter_x.append(x)
            plotter_y.append(y)
            plotter_y1.append(-y)
            plotter_xum.append(x * calibration)
            plotter_yum.append(y * calibration)
            plotter_y1um.append(-y * calibration)

            datapoints.append((x,y))

        ax[0].plot(plotter_x, plotter_y, color = "r", label = "manual_tracking", linewidth = 0.5)

        ax[1].plot(plotter_x, plotter_y, color = "r", label = "manual_tracking", linewidth = 0.5)

        ax[2].plot(plotter_x, plotter_y, color = "r", label = "manual_tracking", linewidth = 0.5)
        ax[3].plot(plotter_xum, plotter_y1um, color = "r", label = "manual_tracking")
        ax[3].legend(loc = "upper right")
        fig.suptitle("auto vs manual tracking")
        plt.show()

        # save tracking file manual
        track.save_tracking(datapoints)

