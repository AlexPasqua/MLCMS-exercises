import pandas as pd
import numpy as np
import math
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


def create_phase_portrait_matrix(A: np.ndarray, title_suffix: str, save_plots=False,
                                 save_path: str = None, display=True):
    """
    Plots the phase portrait of the linear system Ax, where A is a 2x2 matrix and x is a 2-dim vector
    :param A: system's 2x2 matrix
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    """
    w = 10  # width
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of A: ", eigenvalues)
    # linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(10, 10))
    plt.streamplot(X, Y, U, V, density=1.0)
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)


def solve_trajectory(x0, x1, funct, find_best_dt=False, end_time=0.1, plot=False):
    """
    Solves initial value point problem for a dataset, up to a certain moment in time
    :param x0: the data at time 0
    :param x1: the data at unknown time step after 0
    :param funct: to get derivative for next steps generation
    :param end_time: end time for the simulation
    :param plot: boolean to produce a scatter plot of the trajectory (orange) with the final x1 points in blue
    :returns: points at time end_time
    """
    best_dt = -1
    best_mse = math.inf
    x1_pred = []
    t_eval = np.linspace(0, end_time, 100)
    sols = []
    for i in range(len(x0)):
        sol = solve_ivp(funct, [0, end_time], x0[i], t_eval=t_eval)
        x1_pred.append([sol.y[0,-1], sol.y[1,-1]])
        if find_best_dt:
            sols.append(sol.y)
        if plot:
            plt.scatter(x1[i,0], x1[i,1], c='blue', s=10)
            plt.scatter(sol.y[0,:],sol.y[1,:], c='orange', s=4)
    if find_best_dt:
        for i in range(len(t_eval)):
            pred = [[sols[el][0][i], sols[el][1][i]] for el in range(len(sols))]
            mse = np.linalg.norm(pred - x1)**2 / x1.shape[0]
            if mse < best_mse:
                best_mse = mse
                best_dt = t_eval[i]
    if plot:
        plt.rcParams["figure.figsize"] = (14,14)
        plt.show()
    return x1_pred, best_dt, best_mse

def create_phase_portrait_derivative(system, alpha: float, title_suffix: str, save_plots=False,
                                     save_path: str = None, display=True, fig_size=10):
    """
    Plots the phase portrait of the given 'system', where 'system' is a 2 dimensional system given as couple of strings
    :param system: system ODEs
    :param alpha: system's parameter
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    :param fig_size: gives width and height of plotted figure
    """
    # setting up grid width/height
    w = 5
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # dynamic system parameter, responsible for the change in behaviour
    alpha = alpha
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]:
            res = system(0, [x1,x2])
            U.append(res[0])
            V.append(res[1])
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    plt.figure(figsize=(fig_size, fig_size))
    plt.streamplot(X, Y, U, V, density=2)
    plt.title(f"alpha: {alpha} - {title_suffix}")
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)

if __name__ == '__main__':
    vecs = estimate_vectors(delta_t=0.1)
