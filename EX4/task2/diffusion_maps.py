import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import make_swiss_roll, make_s_curve
from sklearn.metrics.pairwise import euclidean_distances

class DiffusionMap:
    def __init__(self):
        pass

    def execute_algorithm(self, data, L = 6, eps=None):
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

    def plot_2D_diffusion_maps_task_one(self, d_map, d_map_idx, time, eps):
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
        ax.scatter(x=d_map, y=time, c=time)
        ax.set_title(f"eigenfunct {d_map_idx} with eps={round(eps,5)}")
        fig.show()

    def plot_2D_diffusion_maps_task_two(self, d_map_zero, d_map, d_map_idx, time, eps):
        """
        plot 2D result of dimension reduction, eigenfunction with respect to time
        :param d_map_zero: eigenfunction constant to plot
        :param d_map: eigenfunction to plot
        :param d_map_idx: eigenfunction index for plot title
        :param time: time array to plot eigenfunction against
        :param eps: decided eps for created eigenfunction
        :return:
        """
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(x=d_map_zero, y=d_map, c=time)
        ax.set_title(f"eigenfunct {d_map_idx} with eps={round(eps,5)}")
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


def plot_fourier(x, t):
    Y_1 = np.fft.fft(x[:, 0])
    Y_2 = np.fft.fft(x[:, 1])
    freq_1 = np.fft.fftfreq(len(x[:, 0]), t[1] - t[0])
    freq_2 = np.fft.fftfreq(len(x[:, 1]), t[1] - t[0])
    plt.figure()
    plt.plot(freq_1, np.abs(Y_1), c='red')
    plt.plot(freq_2, np.abs(Y_2), c='blue')
    plt.show()

if __name__ == '__main__':
    # task2.1
    task = 2
    if task == 1:
        print("doing task2.1")
        x, t = get_part_one_dataset()
        visualize = True
        if visualize:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(x[:, 0], x[:, 1], t, c=t)
            plt.show()
        dm = DiffusionMap()
        phi_l, eps = dm.execute_algorithm(x)
        if visualize:
            for i in range(phi_l.shape[1]):
                dm.plot_2D_diffusion_maps_task_one(phi_l[:, i], i, t, eps)

    # task2.2
    if task == 2:
        print("doing task2.2")
        num_samples = 5000
        x, t = make_swiss_roll(num_samples)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=t)
        plt.show()

        dm = DiffusionMap()
        phi_l, eps = dm.execute_algorithm(x, L=10)
        print("algorithm executed")
        for i in range(phi_l.shape[1]):
            dm.plot_2D_diffusion_maps_task_two(phi_l[:, 1], phi_l[:, i], i, t, eps)
        input()

    # trying s-curve
    if task == 3:
        print("doing task s")
        nr_samples = 5000

        # reduce number of points for plotting
        nr_samples_plot = 1000
        idx_plot = np.random.permutation(nr_samples)[0:nr_samples_plot]

        # generate point cloud
        X, X_color = make_s_curve(nr_samples, random_state=3, noise=0)

        # plot
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            X[idx_plot, 0],
            X[idx_plot, 1],
            X[idx_plot, 2],
            c=X_color[idx_plot],
            cmap=plt.cm.Spectral,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title("point cloud on S-shaped manifold")
        ax.view_init(10, 70)
        fig.show()

        dm = DiffusionMap()
        phi_l, eps = dm.execute_algorithm(X, L=10)
        print("algorithm executed")
        for i in range(phi_l.shape[1]):
            dm.plot_2D_diffusion_maps_task_two(phi_l[:, 1], phi_l[:, i], i, X_color, eps)

