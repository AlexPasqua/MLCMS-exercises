import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.metrics.pairwise import euclidean_distances


class DiffusionMap:
    """
    class implementing the main utilities of a DiffusionMap, including the algorithm for training and plotting
    """
    def __init__(self):
        # initialize all components of the algorithm
        self.D = None
        self.eps = None
        self.W = None
        self.P_inv = None
        self.K = None
        self.Q_inv_sqrt = None
        self.T = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.v_l = None
        self.a_l = None
        self.phi_l = None

    def execute_algorithm(self, data, L=6, eps=None):
        """
            This function will execute the algorithm to get the eigenfunctions.
            :param data: input data to reduce
            :param L: how many eigenfunctions are requested
            :param eps: possibly given, if not given is 5% of max value in self.D
        """
        # form a distance matrix
        self.D = euclidean_distances(data, data)
        print(self.D.shape)
        # set \eps to 5% of the diameter of the dataset if the parameter is not given
        if eps is None:
            self.eps = 0.05 * np.max(self.D)
        else:
            self.eps = eps
        # form the kernel matrix W
        self.W = np.exp((-self.D**2 / self.eps))
        # form the diagonal normalization matrix P
        r = np.sum(self.W, axis=0)
        self.P_inv = np.diag(r**-1)
        # normalize W to form the kernel matrix K
        self.K = self.P_inv @ self.W @ self.P_inv
        # form the diagonal normalization matrix Q
        r = np.sum(self.K, axis=0)
        self.Q_inv_sqrt = np.diag(r**-0.5)
        # form the symmetric matrix T
        self.T = self.Q_inv_sqrt @ self.K @ self.Q_inv_sqrt
        # find the L + 1 largest eigenvalues and  associated eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.T)
        self.a_l = self.eigenvalues[-L-1:][::-1]
        self.v_l = self.eigenvectors[:, -L-1:][:, ::-1]
        # compute the eigenvectors phi_l
        self.phi_l = self.Q_inv_sqrt @ self.v_l
        return self.phi_l, self.eps

    def plot_2D_diffusion_maps_task_one(self, d_map, d_map_idx, time, eps, lim=None):
        """
        plot 2D result of dimension reduction, eigenfunction with respect to time
        :param d_map: eigenfunction to plot
        :param d_map_idx: eigenfunction index for plot title
        :param time: time array to plot eigenfunction against
        :param eps: decided eps for created eigenfunction
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(x=time, y=d_map, c=time)
        ax.set_title(f"eigenfunct {d_map_idx} with eps={round(eps,5)}")
        # useful for eigenfunction 0
        if lim is not None:
            ax.set_ylim([-lim, lim])
        fig.show()

    def plot_2D_diffusion_maps_task_two(self, d_map_zero, d_map, d_map_idx, time, eps, lim=None):
        """
        plot 2D result of dimension reduction, eigenfunction with respect d_map_zero
        :param d_map_zero: eigenfunction to plot as comparison (usually is eigenfunction 1)
        :param d_map: eigenfunction to plot compared to d_map_zero
        :param d_map_idx: eigenfunction index for plot title
        :param time: time array for coloring
        :param eps: decided eps for created eigenfunction
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(x=d_map_zero, y=d_map, c=time)
        ax.set_title(f"eigenfunct {d_map_idx} with eps={round(eps,5)}")
        if lim is not None:
            ax.set_ylim([-lim, lim])
        fig.show()

    def plot_3D_diffusion_maps_task_three(self, d_map_zero, d_map_one, d_map_two, d_map_zero_idx, d_map_one_idx, d_map_two_idx, time, eps, lim=None):
        """
        plot 3D result of dimension reduction, plotting a subspace of three eigenfunctions
        :param d_map_zero: first eigenfunction
        :param d_map_one: second eigenfunction
        :param d_map_two: third eigenfunction
        :param d_map_zero_idx: idx of first
        :param d_map_one_idx: idx of second
        :param d_map_two_idx: idx of third
        :param time: given for coloring
        :param eps: decided eps for created eigenfunction
        :param lim: useful for giving plotting predefined limits (e.g. for phi0)
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(d_map_zero, d_map_one, d_map_two, c=time)
        ax.set_title(f"x=phi_{d_map_zero_idx}, y=phi_{d_map_one_idx}, z=phi_{d_map_two_idx} with eps={round(eps,5)}")
        ax.set_xlim([-lim, lim])
        fig.show()


def get_part_one_dataset(n=1000):
    """
    Function to create dataset desired in exercise_sheet_4, task2, part one
    :param n: number of samples to create
    :return: dataset with x1,x2,tk as coordinates
    """
    tk = 2 * math.pi * np.array(np.arange(0, n)) / (n+1)
    xk = np.array([np.cos(tk), np.sin(tk)])
    return xk.T, tk.T
