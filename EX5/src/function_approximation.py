import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Union, Iterable, Tuple
from scipy.spatial.distance import cdist


def get_points_and_targets(data: Union[str, Iterable[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Depending on the type of the parameter 'data', returns correctly the points and the targets
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays: points and targets
    :returns: points and targets
    """
    if isinstance(data, str):
        data_path = data
        # read data
        linear_func_data = pd.read_csv(data_path, sep=" ", header=None, dtype=np.float64)
        # divide data into auxiliary variables
        points, targets = linear_func_data.iloc[:, 0], linear_func_data.iloc[:, 1]
        points = np.expand_dims(points, 1)  # add 1 dimension, needed for np.linalg.lstsq
    else:
        if len(data) != 2:
            raise ValueError(f"Parameter data must be either a string or an Iterable of 2 numpy ndarrays, got {len(data)} elements")
        points, targets = data[0], data[1]
    return points, targets


def rbf(x, x_l, eps):
    """radial basic function
    Parameters
    ----------
    x: np.ndarray
        data
    x_l: np.ndarray
        random selected data
    eps: float
        epsilon
    Returns
    -------
    matrix contains radial basic function value
    """
    return np.exp(-cdist(x, x_l) ** 2 / eps ** 2)


def compute_bases(points: np.ndarray, eps: float, n_bases: int, centers: np.ndarray = None):
    """
    Compute the basis functions
    :param points: the points on which to calculate the basis functions
    :param centers: the center points to pick to compute the basis functions
    :param eps: epsilon param of the basis functions
    :param n_bases: number of basis functions to compute
    :returns: list of basis functions evaluated on every point in 'points'
    """
    if centers is None:
        # create n_bases basis functions' center points
        # centers = points[np.random.choice(points.ravel(), replace=False, size=n_bases)]
        centers = points[np.random.choice(range(points.shape[0]), replace=False, size=n_bases)]
    phi = rbf(points, centers, eps)
    return phi, centers


def approx_lin_func(data: Union[str, Iterable[np.ndarray]] = "../data/linear_function_data.txt") -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Approximate a linear function through least squares
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays: points and targets
    :returns: tuple (least squares solution, residuals, rank of coefficients matrix, singular values of coefficient matrix)
    """
    # get coefficients and targets from data
    points, targets = get_points_and_targets(data)
    # solve least square
    sol, residuals, rank, singvals = np.linalg.lstsq(a=points, b=targets, rcond=1e-5)
    return sol, residuals, rank, singvals


def approx_nonlin_func(data: Union[str, Iterable[np.ndarray]] = "../data/nonlinear_function_data.txt", n_bases: int = 5, eps: float = 0.1,
                       centers: np.ndarray = None):
    """
    Approximate a non-linear function through least squares
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays: points and targets
    :param n_bases: the number of basis functions to approximate the nonlinear function
    :param eps: bandwidth of the basis functions
    :param centers: list of center points to compute the basis functions
    :returns: tuple (least squares solution (transposed), residuals, rank of coefficients matrix, singular values of coefficient matrix, 
                    centers, eps and phi (list_of_basis))
    """
    # get coefficients and targets form the data
    points, targets = get_points_and_targets(data)

    # evaluate the basis functions on the whole data and putting each basis' result in an array
    list_of_bases, centers = compute_bases(points=points, centers=centers, eps=eps, n_bases=n_bases)

    # solve least square using the basis functions in place of the coefficients to use linear method with nonlinear function
    sol, residuals, rank, singvals = np.linalg.lstsq(a=list_of_bases, b=targets, rcond=1e-5)
    return sol, residuals, rank, singvals, centers, eps, list_of_bases


def plot_func_over_data(lstsqr_sol: np.ndarray, data: Union[str, Iterable[np.ndarray]], linear: bool, centers=None, eps=None, **kwargs):
    """
    Plot the approximated function over the actual data, given the solution of the least squares problem and the data
    :param lstsqr_sol: solution of the least squares problem
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays: points and targets
    :param linear: if True, plots the linear approximated function, otherwise the non-linear one
    :param centers: (optional) list of center points to compute the basis functions in case linear=False
    :param eps: (optional) epsilon parameter to compute the basis functions in case linear=False
    :param kwargs: (optional) can contain more data to include in the title of the plot, e.g. MSE of the approximation
    """
    plot_title = "Approximated function plotted over the actual data"

    # get the data's coefficients and targets
    points, targets = get_points_and_targets(data)

    # compute approximated function for every point on the x axis
    x = np.linspace(start=-5, stop=5, num=100)  # x axis
    if linear:
        y = lstsqr_sol * x  # y value for each x, used to plot the approximated data
    else:
        list_of_bases, centers = compute_bases(points=np.expand_dims(x, 1), centers=centers, eps=eps, n_bases=len(centers))
        y = np.sum(lstsqr_sol * list_of_bases, axis=1)  # '*' indicates and elementwise product (dimensions broadcast to common shape)
        plot_title += f"\nn_bases: {len(centers)}, eps: {eps}"

    # add eventual more data to the plot title
    for k, v in kwargs.items():
        plot_title += f", {k}: {v}"

    # plot approximated function over the actual data
    plt.figure(figsize=(5, 5))
    plt.scatter(points, targets, label="Data")
    plt.plot(x, y, color='r', label="Approximated function")
    plt.legend()
    plt.title(plot_title)
    plt.tight_layout()
    plt.show()


# Functions for solve_ivp

def rbf_approx(t, y, centers, eps, C):
    y = y.reshape(1, y.shape[-1])
    phi = np.exp(-cdist(y, centers) ** 2 / eps ** 2)
    return phi @ C


def linear_approx(t, y, A):
    return A @ y
