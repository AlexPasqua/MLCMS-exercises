import pandas as pd
import numpy as np


def approx_lin_func(data_path: str = None, coeffs: np.ndarray = None, targets: np.ndarray = None):
    """
    Approximate a linear function through least squares
    :param data_path:
        Path to the file containing the data in the format Nx2, col 0 is the data, col 1 the targets.
        If None, then :param coeffs: and :param targets: must be not None.
    :param coeffs:
        Coefficients matrix.
        If None, then :param data_path: must be not None and the value of :param targets: is ignored
    :param targets:
        Targets for each point in the coefficients matrix.
        If None, then :param data_path: must be not None and the value of :param coeffs: is ignored
    :returns: tuple (least squares solution, residuals, rank of coefficients matrix, singular values of coefficient matrix)
    """
    # check consistency of passed arguments
    if data_path is None and (coeffs is None or targets is None):
        raise AttributeError("If :param data_path: is None, then both :param coeffs: and :param targets: must be not None")
    if data_path is not None and coeffs is not None and targets is not None:
        data_path = None    # useful for later
        raise Warning(":param coeffs: and :param targets: were passed, therefore the value of :param data_path: will be ignored")

    # if data_path is not None, read the data, otherwise it means that the data is already in coeffs and targets due to the previous checks
    if data_path is not None:
        # read data
        linear_func_data = pd.read_csv("../data/linear_function_data.txt", sep=" ", header=None, dtype=np.float64)
        # divide data into auxiliary variables
        coeffs, targets = linear_func_data.iloc[:, 0], linear_func_data.iloc[:, 1]
        coeffs = np.expand_dims(coeffs, 1)  # add 1 dimension, needed for np.linalg.lstsq

    # solve least square
    sol, residuals, rank, singvals = np.linalg.lstsq(a=coeffs, b=targets, rcond=None)
    return sol, residuals, rank, singvals


if __name__ == '__main__':
    pass
