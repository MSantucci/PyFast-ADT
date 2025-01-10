# test in-situ tracker

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
import datetime
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import threading
import imageio
from ast import literal_eval
from PIL import Image, ImageTk  # Required for displaying images with Tkinter
from .cross_correlation_kalman_filtered import Tomography_tracker
import random

class InSituTracker():
    def __init__(self, shift= None, tilt_step = 5, sim_mode = False, tomo_tracker = None):
        self.a = []
        self.dt =  0.1
        if sim_mode == True:
            directory = r"L:\Marco\datasets\pyfastadt_tracking_test\philipp_lapo4\06_08_2024\experiment_6\tracking_images"
            os.chdir(directory)
            self.tracking_images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f)) and f.endswith('.tif')]
            self.visualization = False
            self.shift = shift # shift in pixels
        self.tilt_step = tilt_step  # every x frame perform the in-situ correction
        self.updated_positions = []
        self.updated_positions_2 = []
        self.second_iteration = False
        self.tomo_tracker = tomo_tracker

    def main(self):
        self.tomo_tracker = Tomography_tracker(images=self.tracking_images, visualization=self.visualization, dt = self.dt)
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
        return self.track_result

    def shift_image(self, image_array, shift):
        # Get the dimensions of the image
        height, width = image_array.shape[:2]

        # Create a new black image of the same size
        new_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Unpack the shift tuple
        shift_x, shift_y = shift
        # Generate random values for the shift in x and y

        shift_x = random.randint(int(-shift_x/5), int(+shift_x/3))
        shift_y = random.randint(int(-shift_y/5), int(+shift_y/3))

        # Handle horizontal shift (shift_x)
        if shift_x > 0:  # Shift right
            new_image[:, shift_x:] = image_array[:, :-shift_x]
        elif shift_x < 0:  # Shift left
            new_image[:, :width + shift_x] = image_array[:, -shift_x:]

        # Create another new black image to handle vertical shift separately
        final_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Handle vertical shift (shift_y) on the horizontally shifted image
        if shift_y > 0:  # Shift down
            final_image[shift_y:, :] = new_image[:-shift_y, :]
        elif shift_y < 0:  # Shift up
            final_image[:height + shift_y, :] = new_image[-shift_y:, :]

        return final_image

    def insitu_obj_detection_sim(self):
        image_to_use_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        # image_to_use_list = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]
        # image_to_use = 5

        for ii, image_to_use in enumerate(image_to_use_list):


            self.shifted = self.shift_image(self.tomo_tracker.series_support[image_to_use], self.shift)
            self.template = self.tomo_tracker.list_templates[image_to_use]
            original_position = self.track_result["patchworkCC"][image_to_use]

            # Match template to find object in shifted image
            match = cv2.matchTemplate(self.shifted, self.template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

            # Get the top-left corner of the rectangle in the shifted image
            x1, y1 = max_loc
            # Get the width and height of the template
            h, w = self.template.shape[:2]
            # copy the images for the final plots
            self.original_image = self.tomo_tracker.series_support[image_to_use].copy()
            self.shifted_image = self.shifted.copy()
            # Draw a rectangle around the matched region in shifted image
            shifted_with_rect = self.shifted.copy()
            cv2.rectangle(shifted_with_rect, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 1)

            # Calculate the center of the matched region in shifted image
            cx_shifted = x1 + w // 2
            cy_shifted = y1 + h // 2

            # Calculate the center of the original position
            cx_original = original_position[0]
            cy_original = original_position[1]

            cx_found = max_loc[0] + w // 2
            cy_found = max_loc[1] + h // 2

            # Calculate displacement vector from original to new position
            self.displacement_vector = (cx_found - cx_original, cy_found - cy_original)
            self.magnitude = math.sqrt(self.displacement_vector[0] ** 2 + self.displacement_vector[1] ** 2)
            print("shift:", self.displacement_vector)
            print("Magnitude of Displacement:", self.magnitude)

            # Create a subplot layout with 2 images side by side
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Display the original image in the first subplot
            axes[0].imshow(self.original_image, cmap='gray')
            axes[0].set_title('Original Image')

            # Draw the original position on the first image with an arrow
            axes[0].scatter(cx_original, cy_original, color='red', label='Original Position')
            axes[0].arrow(cx_original, cy_original, self.displacement_vector[0], self.displacement_vector[1], head_width=5,
                          head_length=10, fc='blue', ec='blue', label='Displacement')
            axes[0].plot([x[0] for x in self.track_result["patchworkCC"]],
                         [x[1] for x in self.track_result["patchworkCC"]], linewidth=2, linestyle="dotted",
                         color='red', label='patchworkCC original')

            # Display the shifted image in the second subplot
            axes[1].imshow(self.shifted_image, cmap='gray')
            axes[1].set_title('Shifted Image')

            # Draw the new position on the second image with an arrow
            axes[1].scatter(cx_shifted, cy_shifted, color='green', label='New Position')
            axes[1].arrow(cx_shifted, cy_shifted, -self.displacement_vector[0], -self.displacement_vector[1], head_width=5,
                          head_length=10, fc='blue', ec='blue', label='Displacement')

            for i, (x, y) in enumerate(self.track_result["patchworkCC"][image_to_use:image_to_use+5]):
                x += self.displacement_vector[0]
                y += self.displacement_vector[1]
                self.updated_positions.append((x, y))

            self.updated_positions_2.append((image_to_use, self.displacement_vector))

            axes[1].scatter([x[0] for x in self.updated_positions], [y[1] for y in self.updated_positions], color='green', label='patchworkCC shifted', s=1.5)
            axes[1].plot([x[0] for x in self.track_result["patchworkCC"]], [x[1] for x in self.track_result["patchworkCC"]], linewidth=2, linestyle="dotted", color='red', label='patchworkCC original')

            # Add a legend if desired
            axes[0].legend()
            axes[1].legend()

            # Show the subplot with both images
            plt.tight_layout()
            plt.show()

        # final result to display how the tracking file changed due to the in-situ correction
        support1 = []
        support2 = []
        plt.plot([x[0] for x in self.updated_positions], [-x[1] for x in self.updated_positions], linewidth=1, color='red', label='insitu path')
        plt.plot([x[0] for x in self.track_result["patchworkCC"]], [-x[1] for x in self.track_result["patchworkCC"]], linewidth=2, linestyle="dotted", color='green', label='a-priori path')
        for i, a in enumerate(self.updated_positions, start = 1):
            if i % 5 == 0:
                support1.append(a)
            else:
                support2.append(a)
        plt.scatter([x[0] for x in support1], [-x[1] for x in support1], s = 5, color='black')
        plt.scatter([x[0] for x in support2], [-x[1] for x in support2], s = 3, color='gray')
        plt.legend()
        plt.show()

    def in_situ_run(self, image, frame_number):
        self.shifted = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # conversion to 8 bit
        self.shifted_image = self.shifted.copy()
        self.shifted = cv2.bilateralFilter(self.shifted, 9, 150, 150)

        self.template = self.tomo_tracker.list_templates[frame_number]
        original_position = self.track_result["patchworkCC"][frame_number]

        # Match template to find object in shifted image
        match = cv2.matchTemplate(self.shifted, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

        # Get the top-left corner of the rectangle in the shifted image
        x1, y1 = max_loc
        # Get the width and height of the template
        h, w = self.template.shape[:2]
        # copy the images for the final plots
        self.original_image = self.tomo_tracker.series_support[frame_number].copy()
        # Draw a rectangle around the matched region in shifted image
        shifted_with_rect = self.shifted.copy()
        cv2.rectangle(shifted_with_rect, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 1)

        # Calculate the center of the matched region in shifted image
        cx_shifted = x1 + w // 2
        cy_shifted = y1 + h // 2

        # Calculate the center of the original position
        cx_original = original_position[0]
        cy_original = original_position[1]

        cx_found = max_loc[0] + w // 2
        cy_found = max_loc[1] + h // 2

        # Calculate displacement vector from original to new position
        self.displacement_vector = (cx_found - cx_original, cy_found - cy_original)
        self.magnitude = math.sqrt(self.displacement_vector[0] ** 2 + self.displacement_vector[1] ** 2)
        print("shift:", self.displacement_vector)
        print("Magnitude of Displacement:", self.magnitude)

        # Create a subplot layout with 2 images side by side
        plt.clf()
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Display the original image in the first subplot
        axes[0].imshow(self.original_image, cmap='gray')
        axes[0].set_title('Original Image')

        # Draw the original position on the first image with an arrow
        axes[0].scatter(cx_original, cy_original, color='red', label='Original Position')
        axes[0].arrow(cx_original, cy_original, self.displacement_vector[0], self.displacement_vector[1], head_width=5,
                      head_length=10, fc='blue', ec='blue', label='Displacement')
        axes[0].plot([x[0] for x in self.track_result["patchworkCC"]],
                     [x[1] for x in self.track_result["patchworkCC"]], linewidth=2, linestyle="dotted",
                     color='red', label='patchworkCC original')

        # Display the shifted image in the second subplot
        axes[1].imshow(self.shifted_image, cmap='gray')
        axes[1].set_title('Shifted Image')

        # Draw the new position on the second image with an arrow
        axes[1].scatter(cx_shifted, cy_shifted, color='green', label='New Position')
        axes[1].arrow(cx_shifted, cy_shifted, -self.displacement_vector[0], -self.displacement_vector[1], head_width=5,
                      head_length=10, fc='blue', ec='blue', label='Displacement')
        # Add a legend if desired
        axes[0].legend()
        axes[1].legend()
        # Show the subplot with both images
        plt.tight_layout()
        plt.show()

        if frame_number != 0 and self.second_iteration == False:
            self.second_iteration = True
            for i, (x, y) in enumerate(self.track_result["patchworkCC"][0:frame_number]):
                self.updated_positions.append((x, y))

        for i, (x, y) in enumerate(self.track_result["patchworkCC"][frame_number:frame_number + self.tilt_step]):
            x += self.displacement_vector[0]
            y += self.displacement_vector[1]
            self.updated_positions.append((x, y))

        self.updated_positions_2.append((frame_number, self.displacement_vector))
        return self.displacement_vector

    def plot_insitu_result(self):
        # final result to display how the tracking file changed due to the in-situ correction
        plt.clf()
        support1 = []
        support2 = []
        plt.plot([x[0] for x in self.updated_positions], [-x[1] for x in self.updated_positions], linewidth=1, color='red', label='insitu path')
        plt.plot([x[0] for x in self.track_result["patchworkCC"]], [-x[1] for x in self.track_result["patchworkCC"]], linewidth=2, linestyle="dotted", color='green', label='a-priori path')
        for i, a in enumerate(self.updated_positions, start=1):
            if i % self.tilt_step == 0:
                support1.append(a)
            else:
                support2.append(a)
        plt.scatter([x[0] for x in support1], [-x[1] for x in support1], s=5, color='black')
        plt.scatter([x[0] for x in support2], [-x[1] for x in support2], s=3, color='gray')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # """simulation mode here"""
    # # Assuming your image_to_use and other variables are already defined
    # shift = (0, 0)
    # insitu_tracker = InSituTracker(shift = shift, tilt_step = 5, sim_mode = True)
    # track_result = insitu_tracker.main() # normal a priori tracking
    # insitu_tracker.insitu_obj_detection_sim() # thsi is evaluating the shift for the current image/s
    # self = insitu_tracker
    """real experiment here"""
    insitu_tracker = InSituTracker(tilt_step=5)
    # pipe the result of patchworkCC to the in-situ tracker
    ############### insitu_tracker.track_result = track_result ##### thsi should be uncommented in the real exp

    track_result = insitu_tracker.main()  # normal a priori tracking
    image_to_use_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]

    for ii, image_to_use in enumerate(image_to_use_list):
        image = insitu_tracker.shift_image(insitu_tracker.tomo_tracker.series_support[image_to_use], (50,50))

        corrected_pos = insitu_tracker.in_situ_run(image, frame_number=image_to_use)  # this is evaluating the shift for the current image/s

    insitu_tracker.plot_insitu_result()
    self = insitu_tracker