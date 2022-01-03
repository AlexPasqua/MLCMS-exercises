import pandas as pd
import numpy as np
from typing import Tuple


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


def estimate_vectors():
    """

    """
    # read the 2 files
    read_vectorfield_data()


if __name__ == '__main__':
    estimate_vectors()
