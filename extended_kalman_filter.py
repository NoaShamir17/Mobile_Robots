
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))
import math
import matplotlib.pyplot as plt
import numpy as np
from utils import plot_covariance_ellipse

'''
estimation for the sensors accuracy 
'''
# Covariance for EKF simulation
R = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
Q = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance


#  Simulation parameter
INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
GPS_NOISE = np.diag([0.5, 0.5]) ** 2



DT = 0.1  # time tick [s]
SIM_TIME = 80.0  # simulation time [s]

show_animation = True


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def sim_step(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    # add noise to gps x-y
    z = observation_model(xTrue) + GPS_NOISE @ np.random.randn(2, 1)

    # add noise to input
    u_noise = u + INPUT_NOISE @ np.random.randn(2, 1)
    xd = motion_model(xd, u_noise)

    return xTrue, z, xd


def motion_model(x, u):
    """
    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    """
    A = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0],
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT],
                  [1.0, 0.0]])
    x = A @ x + B @ u
    return x


def observation_model(x):
    '''
    observation model
    h = [x_t, y_t]
    '''
    C = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
    return C @ x
    


def jacob_G(x, u):
    """
    Jacobian of Motion Model
    """
    yaw = x[2, 0]
    v = u[0, 0]
    G = np.array([[1.0, 0.0, -v * DT * math.sin(yaw), DT * math.cos(yaw)],
                  [0.0, 1.0, v * DT * math.cos(yaw), DT * math.sin(yaw)],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]])
    return G


def jacob_H():
    # Jacobian of Observation Model
    H = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0]])
    return H


def ekf_estimation(mean, cov, z, u):
    #  Predict
    G = jacob_G(mean, u)
    mean = motion_model(mean, u)
    cov = G @ cov @ G.T + R
    H = jacob_H()
    K = cov @ H.T @ np.linalg.inv(H @ cov @ H.T + Q)  # Kalman gain
    mean = mean + K @ (z - observation_model(mean))  # Update mean
    cov = (np.eye(len(mean)) - K @ H) @ cov  # Update covariance
    return mean, cov


def main():
    print(__file__ + " start!!")

    time = 0.0

    # State Vector [x y yaw v]'
    mean = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    cov = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hmean = mean
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))
    while SIM_TIME >= time:
        time += DT
        u = calc_input()
        xTrue, z, xDR = sim_step(xTrue, xDR, u)
        mean, cov = ekf_estimation(mean, cov, z, u)
        
        # store data history
        hmean = np.hstack((hmean, mean))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            # plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k")
            plt.plot(hmean[0, :].flatten(),
                     hmean[1, :].flatten(), "-r")
            plot_covariance_ellipse(mean[0, 0], mean[1, 0], cov)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)
    

if __name__ == '__main__':
    main()
