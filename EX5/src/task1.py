import pandas as pd
import numpy as np
from typing import Union, Iterable, Tuple


def get_coeffs_and_targets(data: Union[str, Iterable[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Depending on the type of the parameter 'data', returns correctly the coefficients and the targets
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays:
            1st: coefficients matrix
            2nd: targets for each point in the coefficients matrix.
    :returns: coefficients and targets
    """
    if isinstance(data, str):
        data_path = data
        # read data
        linear_func_data = pd.read_csv(data_path, sep=" ", header=None, dtype=np.float64)
        # divide data into auxiliary variables
        coeffs, targets = linear_func_data.iloc[:, 0], linear_func_data.iloc[:, 1]
        coeffs = np.expand_dims(coeffs, 1)  # add 1 dimension, needed for np.linalg.lstsq
    else:
        if len(data) != 2:
            raise ValueError(f"Parameter data must be either a string or an Iterable of 2 numpy ndarrays, got {len(data)} elements")
        coeffs, targets = data[0], data[1]
    return coeffs, targets


def approx_lin_func(data: Union[str, Iterable[np.ndarray]] = "../data/linear_function_data.txt") -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """
    Approximate a linear function through least squares
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays:
            1st: coefficients matrix
            2nd: targets for each point in the coefficients matrix.
    :returns: tuple (least squares solution, residuals, rank of coefficients matrix, singular values of coefficient matrix)
    """
    # get coefficients and targets from data
    coeffs, targets = get_coeffs_and_targets(data)
    # solve least square
    sol, residuals, rank, singvals = np.linalg.lstsq(a=coeffs, b=targets, rcond=None)
    return sol, residuals, rank, singvals


def approx_nonlin_func(data: Union[str, Iterable[np.ndarray]] = "../data/nonlinear_function_data.txt", n_bases: int = 5, eps: float = 0.1):
    """
    Approximate a non-linear function through least squares
    :param data:
        Either str: path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        Or Iterable containing 2 numpy ndarrays:
            1st: coefficients matrix
            2nd: targets for each point in the coefficients matrix.
    :param n_bases: the number of basis functions to approximate the nonlinear function
    :param eps: bandwidth of the basis functions
    :returns: tuple (least squares solution, residuals, rank of coefficients matrix, singular values of coefficient matrix)
    """
    # get coefficients and targets form the data
    coeffs, targets = get_coeffs_and_targets(data)

    # create n_bases basis functions' center points
    centers = np.random.choice(coeffs.ravel(), replace=False, size=n_bases)

    # evaluate the basis functions on the whole data and putting each basis' result in an array
    list_of_bases = np.empty(shape=(len(coeffs), n_bases))
    for i, center_point in enumerate(centers):
        # TODO: choose between alternative 1 and 2 in what follows. They are equivalent, it's a matter of aesthetic and "principle"

        # alternative 1
        subtraction = np.subtract(center_point, coeffs)     # note: center_point is a single point, coeffs are many points -> broadcasting
        norm = np.linalg.norm(subtraction, axis=1)
        basis = np.exp(-norm ** 2 / eps ** 2)
        list_of_bases[:, i] = basis

        # alternative 2
        list_of_bases[:, i] = np.exp(-np.linalg.norm(center_point - coeffs, axis=1) ** 2 / eps ** 2)

    # solve least square using the basis functions in place of the coefficients to use linear method with nonlinear function
    sol, residuals, rank, singvals = np.linalg.lstsq(a=list_of_bases, b=targets, rcond=None)
    return sol, residuals, rank, singvals


if __name__ == '__main__':
    approx_nonlin_func()
