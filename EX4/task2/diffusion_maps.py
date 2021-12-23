import numpy as np
import matplotlib.pyplot as plt
import math

class DiffusionMap:
    def __init__(self):
        pass

    def power_matrix(self, mat, power):
        evalues, evectors = np.linalg.eig(mat)
        return evectors @ np.diag(np.power(evalues, power)) @ np.linalg.inv(evectors)

    def execute_algorithm(self, data, L = 10):
        # form a distance matrix
        self.D = np.zeros((data.shape[0], data.shape[0]))
        for i in range(self.D.shape[0]):
            for j in range(self.D.shape[0]):
                self.D[i][j] = np.linalg.norm(data[i]-data[j])

        # set \eps to 5% of the diameter of the dataset
        self.eps = 0.05 * np.max(self.D)

        # form the kernel matrix W
        self.W = np.exp((-self.D**2 / self.eps))

        # form the diagonal normalization matrix P
        self.P = np.zeros((self.W.shape))
        for i in range(data.shape[0]):
            self.P[i, i] = np.array([np.sum(self.W[i])])

        # normalize W to form the kernel matrix K
        self.K = np.linalg.inv(self.P) @ self.W @ np.linalg.inv(self.P)

        # form the diagonal normalization matrix Q
        self.Q = np.zeros((self.K.shape))
        for i in range(data.shape[0]):
            self.Q[i, i] = np.array([np.sum(self.K[i])])

        # form the symmetric matrix T
        self.T = self.power_matrix(np.linalg.inv(self.Q), 0.5) * self.K * self.power_matrix(np.linalg.inv(self.Q), 0.5)

        # find the L + 1 largest eigenvalues and  associated eigenvectors
        self.eigenvalues, self.eigenvectors = np.linalg.eig(self.T)
        max_L_eigenvalues_index = np.argpartition(self.eigenvalues, -L)[-L:]
        self.a_l = self.eigenvalues[max_L_eigenvalues_index]
        self.v_l = self.eigenvectors[:, max_L_eigenvalues_index]

        # compute the eigenvalues given the \eps root
        self.a_l_eps = np.power(self.a_l, 1 / self.eps)

        # compute the eigenvectors phi_l
        self.psi_l = self.power_matrix(np.linalg.inv(self.Q), 0.5) @ self.v_l
        return self.phi_l


def get_part_one_dataset(n=1000):
    """
    Function to create dataset desired in exercise_sheet_4, task2, part one
    :param n: number of samples to create
    :return: dataset with x1,x2,tk as coordinates
    """
    tk = 2 * math.pi * np.array(np.arange(0, n)) / (n+1)
    xk = np.array([np.cos(tk), np.sin(tk)])
    return xk.T, tk.T


if __name__ == '__main__':
    x, t = get_part_one_dataset()
    visualize = True
    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x[:, 0], x[:, 1], t)
        plt.show()
    dm = DiffusionMap()
    phi_l = dm.execute_algorithm(x)
