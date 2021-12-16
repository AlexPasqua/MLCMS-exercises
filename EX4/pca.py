from turtle import color

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.misc
import cv2
from typing import Union


def read_data(path="data/pca_dataset.txt"):
    """
    Read the data for the PCA task
    :param path: path to the dataset file
    :returns: the data in the form of a numpy.ndarray
    """
    return pd.read_csv(path, sep=" ", header=None).to_numpy()


def center_data(data: Union[np.ndarray, pd.DataFrame]):
    """
    Centers the data, i.e. removes the mean form it
    :param data: data to center
    :returns: the centered data in the form of a numpy.ndarray
    """
    return data - np.mean(data, axis=0)


def read_and_center_data(path="data/pca_dataset.txt"):
    """
    Read the data for the PCA task and centers it by removing the mean
    :param path: path to the dataset file
    :returns: the centered data in the form of a numpy.ndarray
    """
    data = read_data(path)
    centered_data = center_data(data)
    return centered_data


def svd(data: Union[np.ndarray, pd.DataFrame], center=False):
    """
    Compute the Singular Value Decomposition (SVD) of the "data"
    :param data: data to compute the SVD of
    :param center: if True, center the data before performing SVD
    :returns: the 3 matrices forming the SVD decomposition of "data"
    """
    # make the data a numpy ndarray (if it isn't already)
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    # center the data by removing the mean
    if center:
        data = center_data(data)

    # decompose the data through SVD decomposition
    U, singular_values, Vt = np.linalg.svd(data)  # note that V is already transpose
    # starting from a vector containing the singular values, create the S matrix
    S = np.vstack((
        np.diag(singular_values),
        np.zeros(shape=(data.shape[0] - len(singular_values), len(singular_values)))
    ))
    return U, S, Vt.T


def get_lines_along_principal_directions(pt1: np.ndarray, pt2: np.ndarray):
    m = (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])
    q = pt1[1] - m * pt1[0]
    return m, q


def load_racoon(display=False):
    img = scipy.misc.face(gray=True)
    img = cv2.resize(img, dsize=(249, 185))  # rescale image
    if display:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.show()
    return img


def show_2_images_grayscale(img1, img2, vmin=0, vmax=255, titles=("", "")):
    fig, ax = plt.subplots(1, 2, figsize=(10, 15))
    ax[0].imshow(img1, cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title(titles[0])
    ax[1].imshow(img2, cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title(titles[1])
    plt.show()


if __name__ == '__main__':
    load_racoon(True)
