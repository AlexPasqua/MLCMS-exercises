import numpy as np
import pandas as pd


def read_manifold_data(path="../data/takens_1.txt") -> np.ndarray:
    """
    Reads the file containing the manifold and returns the data
    :param path: path of the data file
    :returns: the data as numpy ndarray
    """
    data = pd.read_csv(path, header=None, sep=" ")
    return data.to_numpy()


if __name__ == '__main__':
    read_manifold_data()
