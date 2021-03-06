import pandas as pd
import numpy as np
import math
from typing import Tuple
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from function_approximation import rbf_approx, approx_nonlin_func


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
    Plots the phase portrait of the linear system Ax
    :param A: system's (2x2 matrix in our case)
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


def solve_trajectory(x0, x1, funct, args, find_best_dt=False, end_time=0.1, plot=False):
    """
    Solves initial value point problem for a whole dataset of points, up to a certain moment in time
    :param x0: the data at time 0
    :param x1: the data at unknown time step after 0
    :param funct: to get derivative for next steps generation
    :param find_best_dt: if True also the dt where we have lowest MSE is searched
    :param end_time: end time for the simulation
    :param plot: boolean to produce a scatter plot of the trajectory (orange) with the final x1 points in blue
    :returns: points at time end_time, best point in time (getting lowest MSE), lowest MSE
    """
    # initialize variables for find_best_dt procedure
    best_dt = -1
    best_mse = math.inf
    x1_pred = []
    # fixate some times where system must be evaluated
    t_eval = np.linspace(0, end_time, 100)
    sols = []
    for i in range(len(x0)):
        sol = solve_ivp(funct, [0, end_time], x0[i], args=args, t_eval=t_eval)  # solve initial value problem for a given point
        x1_pred.append([sol.y[0, -1], sol.y[1, -1]])  # save the final solution
        if find_best_dt:
            # to find best dt then all the different snapshots in time have to be saved
            sols.append(sol.y)
        # plot the trajectory (orange) and ground truth end point (blue)
        if plot:
            plt.scatter(x1[i, 0], x1[i, 1], c='blue', s=10)
            plt.scatter(sol.y[0, :], sol.y[1, :], c='orange', s=4)
    if find_best_dt:
        # try all the different moments in time, check if it is the best time
        for i in range(len(t_eval)):
            pred = [[sols[el][0][i], sols[el][1][i]] for el in range(len(sols))]
            mse = np.mean(np.linalg.norm(pred - x1, axis=1)**2)
            # if mse found is best yet, update the variables
            if mse < best_mse:
                best_mse = mse
                best_dt = t_eval[i]
    if plot:
        plt.rcParams["figure.figsize"] = (14,14)
        plt.show()
    return x1_pred, best_dt, best_mse


def find_best_rbf_configuration(x0, x1, dt=0.1, end_time=0.5):
    """
    grid search over various different eps and n_bases values, returning the whole configuration with lowest MSE
    :param x0: data at time 0
    :param x1: data after a certain unknown dt
    :param dt: dt to approximate the vector field between x0 and x1
    :param end_time: total time of solve_ivp system solving trajectory
    :return: best mse found with the configuration, including eps, n_bases, dt at which the mse was found, centers
    """
    final_best_mse, final_best_eps, final_best_n_bases, final_best_dt = math.inf, -1, -1, -1  # initialize variables
    n_bases_trials = [int(i) for i in np.linspace(100, 1001, 20)]  # define search space for n_bases
    for n_bases in n_bases_trials:
        centers = x0[np.random.choice(range(x0.shape[0]), replace=False, size=n_bases)]  # define centers
        for eps in (0.3, 0.5, 0.7, 1.0, 5.0, 10.0, 20.0):
            v = estimate_vectors(dt, x0, x1)  # estimate vector field
            C, res, _, _, _, eps, phi = approx_nonlin_func(data=(x0,v), n_bases=n_bases, eps=eps, centers=centers)
            x1_pred, best_dt, best_mse = solve_trajectory(x0, x1, rbf_approx, find_best_dt=True, args=[centers, eps, C], end_time=end_time, plot=False)
            if final_best_mse > best_mse:  # if new mse is better then update all return variables
                final_best_mse, final_best_eps, final_best_n_bases, final_best_dt, final_centers  = best_mse, eps, n_bases, best_dt, centers
    print(f"Printing best configuration: eps = {final_best_eps} - n_bases = {final_best_n_bases} - dt = {final_best_dt} giving MSE = {final_best_mse}")
    return final_best_mse, final_best_eps, final_best_n_bases, final_best_dt, final_centers


def create_phase_portrait_derivative(funct, args, title_suffix: str, save_plots=False,
                                     save_path: str = None, display=True, fig_size=10, w=4.5):
    """
    Plots the phase portrait given a 'funct' that gives the derivatives for a certain point
    :param funct: given a 2d point gives back the 2 derivatives
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    :param fig_size: gives width and height of plotted figure
    :param w: useful for defining range for setting Y and X
    """
    # setting up grid width/height
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # dynamic system parameter, responsible for the change in behaviour
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]:
            res = funct(0, np.array([x1, x2]), *args)
            U.append(res[0][0])
            V.append(res[0][1])
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    plt.figure(figsize=(fig_size, fig_size))
    plt.streamplot(X, Y, U, V, density=2)
    plt.title(f"{title_suffix}")
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)
