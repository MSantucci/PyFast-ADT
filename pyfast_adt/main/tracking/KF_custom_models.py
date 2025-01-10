# set of custom KF models that can be used in pyfast-ADT
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver
from filterpy.stats import plot_covariance
import math
import matplotlib.pyplot as plt
import numpy as np


class ukf_2D:
    def __init__(self, dt = 1, x_std_P = 500, y_std_P = 500, x_std_R = 100, y_std_R = 100, noise_std_Q = 100, pos_x0 = 0, pos_y0 = 0):
        self.dt = float(dt)
        # variance for the process (P)
        self.x_std_P = float(x_std_P)  # pixels
        self.y_std_P = float(y_std_P)  # pixels

        # variance for the measurement (R)
        self.x_std_R = float(x_std_R)  # pixels
        self.y_std_R = float(y_std_R)  # pixels

        # variance for the noise (Q)
        self.noise_std_Q = float(noise_std_Q)

        self.pos_x0 = float(pos_x0)
        self.pos_y0 = float(pos_y0)

        self.points = MerweScaledSigmaPoints(n=2, alpha=.1, beta=2., kappa=1.)
        self.kf = UKF(dim_x=2, dim_z=2, dt=self.dt, fx=self.f_track, hx=self.h_track, points=self.points)
        saver = Saver(self.kf)  # save data for kf filter

        # state vector [x, dx, y, dy] 1x4 we input the initial x, y pos from obj detection and velocities 0
        self.kf.x = np.array([self.pos_x0, self.pos_y0], dtype=float)
        # state covariance matrix P 4x4
        self.kf.P = np.diag([self.x_std_P ** 2, self.y_std_P ** 2])
        # measurement noise matrix R 2x2
        self.kf.R = np.diag([self.x_std_R ** 2, self.y_std_R ** 2])
        # process noise matrix Q 4x4
        self.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=self.dt, var=self.noise_std_Q**2)

    # create the f(x) and h(x) functions
    def f_track(self, x, dt):
        """ state transition function for nonlinear kf
        x is a vector as [x, dx, y, dy]"""
        F = np.array([[1, 0],
                      [0, 1]], dtype=float)
        return F @ x

    def h_track(self, z):
        """ measurement function for nonlinear kf."""
        return z[[0, 1]]

    def predict(self):
        self.kf.predict()
        return self.kf.x[[0,1]]

    def update(self, z):
        self.kf.update(z=z)
        return self.kf.x[[0,1]]

    def run(self, path):
        """this function run the model on a custom dataset provided by the path argument where a path of a txt file
        containing the X,Y coordinates of the tracked object"""
        # Read the tracking file containing the x,y positions in pixels
        with open(path, 'r') as file:
            file_contents = file.read()

        values = {}
        lines = file_contents.split('\n')

        data_start = False
        exp_points = []
        for i, line in enumerate(lines):
            if 'X, Y position in pixels' in line:
                data_start = True
                continue
            if data_start:
                try:
                    x, y = line.split(",")
                    exp_points.append([float(x), float(y)])
                except:
                    break
            else:
                pass
        exp_points = np.array(exp_points)
        exp_points[:, 1] += -1023
        exp_points[:, 1] = abs(exp_points[:, 1])

        # print(x_point, y_point)

        # set up the ukf for tracking
        dt = 1.
        # variance for the process
        x_std_P = 500.  # pixels
        y_std_P = 500.  # pixels

        # variance for the measurement
        x_std_R = 100  # pixels
        y_std_R = 100  # pixels

        points = MerweScaledSigmaPoints(n=2, alpha=.1, beta=2., kappa=1.)
        kf = UKF(dim_x=2, dim_z=2, dt=dt, fx=self.f_track, hx=self.h_track, points=points)
        saver = Saver(kf)  # save data for kf filter

        # state vector [x, dx, y, dy] 1x4 we input the initial x, y pos from obj detection and velocities 0
        kf.x = np.array([exp_points[0][0], exp_points[0][1]], dtype=float)
        # state covariance matrix P 4x4
        kf.P = np.diag([x_std_P ** 2, y_std_P ** 2])
        # measurement noise matrix R 2x2
        kf.R = np.diag([x_std_R ** 2, y_std_R ** 2])
        # process noise matrix Q 4x4
        kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=100)

        xs = []

        for z in exp_points:
            kf.predict()
            kf.update([z[0], z[1]])
            xs.append(kf.x.copy())
            saver.save()

        xs = np.array(xs)
        # Plot the tracking path and CC points
        plt.plot(np.array(saver.x)[:, 0], np.array(saver.x)[:, 1], color='blue', label='Tracking Path')
        plt.plot(np.array(saver.z)[:, 0], np.array(saver.z)[:, 1], color='red', label='CC points')

        # Plot the covariance ellipses
        for i, z in enumerate(exp_points):
            if i % 3 == 0:
                a = saver.P[i].copy()
                cov = np.array([[a[0, 0], a[1, 0]],
                                [a[0, 1], a[1, 1]]])
                mean = (saver.x[i][0], saver.x[i][1])
                # plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.2, axis_equal=False, xlim=(min_x, max_x), ylim=(min_y, max_y))
                plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.2)

        # Set labels and legend
        plt.xlabel("X pixels")
        plt.ylabel("Y pixels")
        plt.legend()

        # Show the plot
        plt.show()

        plt.plot(saver.likelihood, label='likelihood')
        plt.title("likelihood")
        plt.xlabel("frame number")
        plt.legend()
        plt.show()

        plt.plot(saver.log_likelihood, label='log likelihood')
        plt.title("log likelihood")
        plt.xlabel("frame number")
        plt.legend()
        plt.show()

        # Create a range of frame numbers
        frame_numbers = np.arange(len(saver.x))

        # # Plot velocity x vs frame number
        # plt.plot(frame_numbers, abs(np.array(saver.x)[:, 1]), color='blue', label='Velocity X')
        # # Plot velocity y vs frame number
        # plt.plot(frame_numbers, abs(np.array(saver.x)[:, 3]), color='red', label='Velocity Y')
        # # Set labels and legend
        # plt.title("accelleration plot: abs velocity vs time")
        # plt.xlabel("time")
        # plt.ylabel("abs Velocity")
        # plt.legend()
        # # Show the plot
        # plt.show()

        # Plot position x vs time
        plt.plot(frame_numbers, np.array(saver.x)[:, 0], color='blue', label='X filtered')
        plt.plot(frame_numbers, exp_points[:, 0], color='blue', label='X experimental', linestyle='--')
        # Plot position y vs time
        plt.plot(frame_numbers, np.array(saver.x)[:, 1], color='red', label='Y filtered')
        plt.plot(frame_numbers, exp_points[:, 1], color='red', label='Y experimental', linestyle='--')
        # Set labels and legend
        plt.title("velocity plot: position vs time")
        plt.xlabel("time")
        plt.ylabel("position")
        plt.legend()
        # Show the plot
        plt.show()
        return saver

class ukf_4D:
    def __init__(self, dt=1, x_std_P=500, y_std_P=500, x_std_R=100, y_std_R=100, noise_std_Q=100, pos_x0=0, pos_y0=0, vel_x0 = 0., vel_y0 = 0., vel_std_P = 10.):
        self.dt = float(dt)
        # variance for the process (P)
        self.x_std_P = float(x_std_P)  # pixels
        self.y_std_P = float(y_std_P)  # pixels

        self.vel_std_P = float(vel_std_P)

        # variance for the measurement (R)
        self.x_std_R = float(x_std_R)  # pixels
        self.y_std_R = float(y_std_R)  # pixels

        # variance for the noise (Q)
        self.noise_std_Q = float(noise_std_Q)

        self.pos_x0 = float(pos_x0)
        self.pos_y0 = float(pos_y0)

        self.vel_x0 = float(vel_x0)
        self.vel_y0 = float(vel_y0)

        self.points = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=-1.)
        self.kf = UKF(dim_x=4, dim_z=2, dt=self.dt, fx=self.f_track, hx=self.h_track, points=self.points)
        saver = Saver(self.kf)  # save data for kf filter

        # state vector [x, dx, y, dy] 1x4 we input the initial x, y pos from obj detection and velocities 0
        self.kf.x = np.array([self.pos_x0, self.pos_y0, self.vel_x0, self.vel_y0], dtype=float)
        # state covariance matrix P 4x4
        self.kf.P = np.diag([self.x_std_P ** 2, self.vel_std_P ** 2, self.y_std_P ** 2, self.vel_std_P ** 2])
        # measurement noise matrix R 2x2
        self.kf.R = np.diag([self.x_std_R ** 2, self.y_std_R ** 2])
        # process noise matrix Q 4x4
        self.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=self.dt, var=self.noise_std_Q**2)
        self.kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=self.dt, var=self.noise_std_Q**2)


    # create the f(x) and h(x) functions
    def f_track(self, x, dt):
        """ state transition function for nonlinear kf
        x is a vector as [x, dx, y, dy]"""
        F = np.array([[1, dt, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, dt],
                      [0, 0, 0, 1]], dtype=float)
        return F @ x

    def h_track(self, z):
        """ measurement function for nonlinear kf."""
        return z[[0, 2]]

    def predict(self):
        self.kf.predict()
        return self.kf.x[[0,2]]

    def update(self, z):
        self.kf.update(z=z)
        return self.kf.x[[0,2]]

    def run(self, path):
        # Read the tracking file containing the x,y positions in pixels
        with open(path, 'r') as file:
            file_contents = file.read()

        values = {}
        lines = file_contents.split('\n')

        data_start = False
        exp_points = []
        for i, line in enumerate(lines):
            if 'X, Y position in pixels' in line:
                data_start = True
                continue
            if data_start:
                try:
                    x, y = line.split(",")
                    exp_points.append([float(x), float(y)])
                except:
                    break
            else:
                pass
        exp_points = np.array(exp_points)
        exp_points[:, 1] += -1023
        exp_points[:, 1] = abs(exp_points[:, 1])

        # print(x_point, y_point)

        dt = 1.
        # variance for the process
        x_std_P = 3.  # pixels
        y_std_P = 3.  # pixels
        vel_std_P = 10.
        # variance for the measurement
        x_std_R = 10.  # pixels
        y_std_R = 10.  # pixels

        points = MerweScaledSigmaPoints(n=4, alpha=.1, beta=2., kappa=-1.)
        kf = UKF(dim_x=4, dim_z=2, dt=dt, fx=self.f_track, hx=self.h_track, points=points)
        saver = Saver(kf)  # save data for kf filter

        # state vector [x, dx, y, dy] 1x4 we input the initial x, y pos from obj detection and velocities 0
        kf.x = np.array([exp_points[0][0], exp_points[1][0] - exp_points[0][0], exp_points[0][1],
                         exp_points[1][1] - exp_points[0][1]], dtype=float)
        # state covariance matrix P 4x4
        kf.P = np.diag([x_std_P ** 2, vel_std_P ** 2, y_std_P ** 2, vel_std_P ** 2])
        # measurement noise matrix R 2x2
        kf.R = np.diag([x_std_R ** 2, y_std_R ** 2])
        # process noise matrix Q 4x4
        kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0.1)
        kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=0.1)

        xs = []

        for z in exp_points:
            kf.predict()
            kf.update([z[0], z[1]])
            xs.append(kf.x.copy())
            saver.save()

        xs = np.array(xs)
        # Plot the tracking path and CC points
        plt.plot(np.array(saver.x)[:, 0], np.array(saver.x)[:, 2], color='blue', label='Tracking Path')
        plt.plot(np.array(saver.z)[:, 0], np.array(saver.z)[:, 1], color='red', label='CC points')

        # # Calculate and set xlim and ylim
        # min_x, max_x = float(exp_points[:, 0].min() - 30), float(exp_points[:, 0].max() + 30)
        # min_y, max_y = float(exp_points[:, 1].min() - 30), float(exp_points[:, 1].max() + 30)

        # Plot the covariance ellipses
        for i, z in enumerate(exp_points):
            if i % 3 == 0:
                a = saver.P[i].copy()
                cov = np.array([[a[0, 0], a[2, 0]],
                                [a[0, 2], a[2, 2]]])
                mean = (saver.x[i][0], saver.x[i][2])
                # plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.2, axis_equal=False, xlim=(min_x, max_x), ylim=(min_y, max_y))
                plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.2)

        # Set labels and legend
        plt.xlabel("X pixels")
        plt.ylabel("Y pixels")
        plt.legend()

        # Show the plot
        plt.show()

        plt.plot(saver.likelihood, label='likelihood')
        plt.title("likelihood")
        plt.xlabel("frame number")
        plt.legend()
        plt.show()

        plt.plot(saver.log_likelihood, label='log likelihood')
        plt.title("log likelihood")
        plt.xlabel("frame number")
        plt.legend()
        plt.show()

        # Create a range of frame numbers
        frame_numbers = np.arange(len(saver.x))

        # Plot velocity x vs frame number
        plt.plot(frame_numbers, abs(np.array(saver.x)[:, 1]), color='blue', label='Velocity X')
        # Plot velocity y vs frame number
        plt.plot(frame_numbers, abs(np.array(saver.x)[:, 3]), color='red', label='Velocity Y')
        # Set labels and legend
        plt.title("accelleration plot: abs velocity vs time")
        plt.xlabel("time")
        plt.ylabel("abs Velocity")
        plt.legend()
        # Show the plot
        plt.show()

        # # Plot velocity x vs frame number
        # plt.plot(np.array(saver.x)[:, 1], np.array(saver.x)[:, 0], color='blue', label='X')
        # # Plot velocity y vs frame number
        # plt.plot(np.array(saver.x)[:, 3], np.array(saver.x)[:, 2], color='red', label='Y')
        # # Set labels and legend
        # plt.title("velocity vs position")
        # plt.xlabel("Position")
        # plt.ylabel("Velocity")
        # plt.legend()
        # # Show the plot
        # plt.show()

        # Plot position x vs time
        plt.plot(frame_numbers, np.array(saver.x)[:, 0], color='blue', label='X filtered')
        plt.plot(frame_numbers, exp_points[:, 0], color='blue', label='X experimental', linestyle='--')
        # Plot position y vs time
        plt.plot(frame_numbers, np.array(saver.x)[:, 2], color='red', label='Y filtered')
        plt.plot(frame_numbers, exp_points[:, 1], color='red', label='Y experimental', linestyle='--')
        # Set labels and legend
        plt.title("velocity plot: position vs time")
        plt.xlabel("time")
        plt.ylabel("position")
        plt.legend()
        # Show the plot
        plt.show()
        return saver


class ukf_6D:
    def __init__(self, dt=1, x_std_P=500, y_std_P=500, x_std_R=100, y_std_R=100, noise_std_Q=100, pos_x0=0, pos_y0=0, vel_x0=0., vel_y0=0., acc_x0 = 0, acc_y0 = 0, vel_std_P = 10., acc_std_P = 10.):
        self.dt = float(dt)
        # variance for the process (P)
        self.x_std_P = float(x_std_P)  # pixels
        self.y_std_P = float(y_std_P)  # pixels
        self.vel_std_P = float(vel_std_P)
        self.acc_std_P = float(acc_std_P)

        # variance for the measurement (R)
        self.x_std_R = float(x_std_R)  # pixels
        self.y_std_R = float(y_std_R)  # pixels

        # variance for the noise (Q)
        self.noise_std_Q = float(noise_std_Q)

        self.pos_x0 = float(pos_x0)
        self.pos_y0 = float(pos_y0)

        self.vel_x0 = float(vel_x0)
        self.vel_y0 = float(vel_y0)

        self.acc_x0 = float(acc_x0)
        self.acc_y0 = float(acc_y0)

        self.points = MerweScaledSigmaPoints(n=6, alpha=.1, beta=2., kappa=-3.)
        self.kf = UKF(dim_x=6, dim_z=2, dt=self.dt, fx=self.f_track, hx=self.h_track, points=self.points)
        saver = Saver(self.kf)  # save data for kf filter

        # state vector [x, dx, y, dy] 1x4 we input the initial x, y pos from obj detection and velocities 0
        self.kf.x = np.array([self.pos_x0, self.vel_x0, self.acc_x0, self.pos_y0, self.vel_y0, self.acc_y0], dtype=float)
        # state covariance matrix P 6x6
        self.kf.P = np.diag([self.x_std_P ** 2, self.vel_std_P ** 2, self.acc_std_P ** 2, self.y_std_P ** 2, self.vel_std_P ** 2, self.acc_std_P ** 2])
        # measurement noise matrix R 2x2
        self.kf.R = np.diag([self.x_std_R ** 2, self.y_std_R ** 2])
        # process noise matrix Q 4x4
        self.kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=self.dt, var=self.noise_std_Q**2)
        self.kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=self.dt, var=self.noise_std_Q**2)
        self.kf.Q[4:6, 4:6] = Q_discrete_white_noise(2, dt=self.dt, var=self.noise_std_Q**2)

    # create the f(x) and h(x) functions
    def f_track(self, x, dt):
        """ state transition function for nonlinear kf
        x is a vector as [x, dx, y, dy]"""
        v = dt
        a = 0.5*(dt**2)
        F = np.array([[1, v, a, 0, 0, 0],
                      [0, 1, v, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, v, a],
                      [0, 0, 0, 0, 1, v],
                      [0, 0, 0, 0, 0, 1]], dtype=float)
        return F @ x

    def h_track(self, z):
        """ measurement function for nonlinear kf."""
        return z[[0, 3]]

    def predict(self):
        self.kf.predict()
        return self.kf.x[[0,3]]

    def update(self, z):
        self.kf.update(z=z)
        return self.kf.x[[0,3]]

    def run(self, path):
        # Read the tracking file containing the x,y positions in pixels
        with open(path, 'r') as file:
            file_contents = file.read()

        values = {}
        lines = file_contents.split('\n')

        data_start = False
        exp_points = []
        for i, line in enumerate(lines):
            if 'X, Y position in pixels' in line:
                data_start = True
                continue
            if data_start:
                try:
                    x, y = line.split(",")
                    exp_points.append([float(x), float(y)])
                except:
                    break
            else:
                pass
        exp_points = np.array(exp_points)
        exp_points[:, 1] += -1023
        exp_points[:, 1] = abs(exp_points[:, 1])

        # print(x_point, y_point)
        # set up the ukf for tracking
        dt = 1.
        # variance for the process
        x_std_P = 2. # pixels
        y_std_P = 2. # pixels
        vel_std_P = 5.
        acc_std_P = 10.
        # variance for the measurement
        x_std_R = 3. # pixels
        y_std_R = 3. # pixels



        points = MerweScaledSigmaPoints(n=6, alpha=.1, beta=2., kappa=-3.)
        kf = UKF(dim_x= 6, dim_z= 2, dt= dt, fx=self.f_track, hx=self.h_track, points=points)
        saver = Saver(kf) # save data for kf filter
        pos_x0 = exp_points[0][0]
        pos_y0 = exp_points[0][1]

        vel_x0 = exp_points[1][0]-exp_points[0][0]
        vel_y0 = exp_points[1][1]-exp_points[0][1]

        acc_x0 = (exp_points[1][0]-exp_points[0][0]) - (exp_points[2][0]-exp_points[1][0])
        acc_y0 = (exp_points[1][1]-exp_points[0][1]) - (exp_points[2][1]-exp_points[1][1])


        # state vector [x, dx, y, dy] 1x4 we input the initial x, y pos from obj detection and velocities 0
        kf.x = np.array([pos_x0, vel_x0, acc_x0, pos_y0, vel_y0, acc_y0], dtype=float)
        # state covariance matrix P 6x6
        kf.P = np.diag([x_std_P**2, vel_std_P**2, acc_std_P**2, y_std_P**2, vel_std_P**2, acc_std_P**2])
        # measurement noise matrix R 2x2
        kf.R = np.diag([x_std_R**2, y_std_R**2])
        # process noise matrix Q 4x4
        kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=1)
        kf.Q[2:4, 2:4] = Q_discrete_white_noise(2, dt=dt, var=1)
        kf.Q[4:6, 4:6] = Q_discrete_white_noise(2, dt=dt, var=1)

        xs = []

        for z in exp_points:
            kf.predict()
            kf.update([z[0], z[1]])
            xs.append(kf.x.copy())
            saver.save()

        xs = np.array(xs)
        # Plot the tracking path and CC points
        plt.plot(np.array(saver.x)[:, 0], np.array(saver.x)[:, 3], color='blue', label='Tracking Path')
        plt.plot(np.array(saver.z)[:, 0], np.array(saver.z)[:, 1], color='red', label='CC points')

        # # Calculate and set xlim and ylim
        # min_x, max_x = float(exp_points[:, 0].min() - 30), float(exp_points[:, 0].max() + 30)
        # min_y, max_y = float(exp_points[:, 1].min() - 30), float(exp_points[:, 1].max() + 30)

        # Plot the covariance ellipses
        for i, z in enumerate(exp_points):
            if i % 3 == 0:
                a = saver.P[i].copy()
                cov = np.array([[a[0, 0], a[3, 0]],
                                [a[0, 3], a[3, 3]]])
                mean = (saver.x[i][0], saver.x[i][3])
                # plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.2, axis_equal=False, xlim=(min_x, max_x), ylim=(min_y, max_y))
                plot_covariance(mean, cov=cov, fc='g', std=3, alpha=0.2)

        # Set labels and legend
        plt.xlabel("X pixels")
        plt.ylabel("Y pixels")
        plt.legend()

        # Show the plot
        plt.show()

        plt.plot(saver.likelihood, label = 'likelihood')
        plt.title("likelihood")
        plt.xlabel("frame number")
        plt.legend()
        plt.show()

        plt.plot(saver.log_likelihood, label = 'log likelihood')
        plt.title("log likelihood")
        plt.xlabel("frame number")
        plt.legend()
        plt.show()

        # Create a range of frame numbers
        frame_numbers = np.arange(len(saver.x))


        # Plot velocity x vs frame number
        # plt.plot(frame_numbers, abs(np.array(saver.x)[:, 1]), color='blue', label='Velocity X')
        plt.errorbar(frame_numbers, abs(np.array(saver.x)[:, 1]),yerr= np.sqrt(np.array(saver.P)[:,1,1])*3, color='blue', label='Velocity X')
        # Plot velocity y vs frame number
        # plt.plot(frame_numbers, abs(np.array(saver.x)[:, 4]), color='red', label='Velocity Y')
        plt.errorbar(frame_numbers, abs(np.array(saver.x)[:, 4]), yerr= np.sqrt(np.array(saver.P)[:,4,4])*3, color='red', label='Velocity Y')
        # Set labels and legend
        plt.title("accelleration plot: abs velocity vs time")
        plt.xlabel("time")
        plt.ylabel("abs Velocity")
        plt.legend()
        # Show the plot
        plt.show()


        # Plot acc x vs time
        # plt.plot(frame_numbers, np.array(saver.x)[:, 2], color='blue', label='acc X filtered')
        plt.errorbar(frame_numbers, np.array(saver.x)[:, 2], yerr= np.sqrt(np.array(saver.P)[:,2,2])*3,color='blue', label='acc X filtered')
        # Plot acc y vs time
        # plt.plot(frame_numbers, np.array(saver.x)[:, 5], color='red', label='acc Y filtered')
        plt.errorbar(frame_numbers, np.array(saver.x)[:, 5], yerr= np.sqrt(np.array(saver.P)[:,5,5])*3,color='red', label='acc Y filtered')
        # Set labels and legend
        plt.title("acc vs time")
        plt.xlabel("time")
        plt.ylabel("accelleration")
        plt.legend()
        # Show the plot
        plt.show()


        # Plot position x vs time
        plt.plot(frame_numbers, np.array(saver.x)[:, 0], color='blue', label='X filtered')
        plt.plot(frame_numbers, exp_points[:, 0], color='blue', label='X experimental', linestyle='--')
        # Plot position y vs time
        plt.plot(frame_numbers, np.array(saver.x)[:, 3], color='red', label='Y filtered')
        plt.plot(frame_numbers, exp_points[:, 1], color='red', label='Y experimental', linestyle='--')
        # Set labels and legend
        plt.title("velocity plot: position vs time")
        plt.xlabel("time")
        plt.ylabel("position")
        plt.legend()
        # Show the plot
        plt.show()
        return saver