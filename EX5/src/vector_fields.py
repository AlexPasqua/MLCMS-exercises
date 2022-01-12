import pandas as pd
import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def read_vectorfield_data(dir_path="../data/", base_filename="linear_vectorfield_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the 2 files containing the vector field data
    :param dir_path: path of the directory containing the 2 files
    :param base_filename: common part of the name in the 2 files, then the suffix "_x0.txt" or "_x1.txt" is added
    :returns: the data contained in the 2 files in the form of 2 numpy ndarrays
    """
    x0 = pd.read_csv(dir_path + base_filename + "_x0.txt", sep=' ', header=None).to_numpy()
    x1 = pd.read_csv(dir_path + base_filename + "_x1.txt", sep=' ', header=None).to_numpy()
    return x0, x1


def estimate_vectors(delta_t: float, x0=None, x1=None) -> np.ndarray:
    """
    Estimates the vector field using the finite-difference formula
    :param delta_t: the time difference used as denominator of the time-difference formula
    :param x0: the data at beginning of time delta
    :param x1: the data at end of time delta
    :returns: an approximation of the vectors s.t. v(x0_k) = x1_k
    """
    # read the 2 files containing the vector field data (if data is not given)
    if x0 is None or x1 is None:
        x0, x1 = read_vectorfield_data()
    # estimate the vector field through the finite-difference formula
    vectors = (x1 - x0) / delta_t
    return vectors


def solve_trajectory(x0, x1, funct, end_time=0.1, plot=False):
    """
    Solves initial value point problem for a dataset, up to a certain moment in time
    :param x0: the data at time 0
    :param x1: the data at unknown time step after 0
    :param funct: to get derivative for next steps generation
    :param end_time: end time for the simulation
    :param plot: boolean to produce a scatter plot of the trajectory (orange) with the final x1 points in blue
    :returns: points at time end_time
    """
    x1_pred = []
    for i in range(len(x0)):
        sol = solve_ivp(funct, [0, end_time], x0[i], t_eval=np.linspace(0, 0.1, 100))
        x1_pred.append([sol.y[0,-1], sol.y[1,-1]])
        if plot:
            plt.scatter(x1[i,0], x1[i,1], c='blue', s=10)
            plt.scatter(sol.y[0,:],sol.y[1,:], c='orange', s=4)
    if plot:
        plt.show()
    return x1_pred

if __name__ == '__main__':
    vecs = estimate_vectors(delta_t=0.1)
