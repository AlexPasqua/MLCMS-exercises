import matplotlib.pyplot as plt
import numpy as np


def plot_traj(start_pos, ts=1e-3, iter_num=10000):
    """
    Function to plot a trajectory for a certain amount of iterations, given a starting point.
    @param start_pos: initial coordinates of point
    @param ts: step length
    @param iter_num: number of iterations to execute
    """
    X, Y = [], []  # lists to store positioning update
    x = start_pos
    for i in range(iter_num):
        x1, x2 = x[0], x[1]
        X.append(x1)
        Y.append(x2)
        # find vectors for next movement
        u = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
        v = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)
        # execute step
        x = x + ts * np.array([u, v])
    plt.figure(figsize=(10, 10))
    # plot trajectory
    plt.plot(X, Y)
    plt.scatter(X[0], Y[0])  # put the starting point in evidence
    plt.title(f"Trajectory [{start_pos[0]}, {start_pos[1]}]")
    plt.xlim(-w, w)
    plt.ylim(-w, w)
    plt.show()


if __name__ == '__main__':
    # TASK 3.1
    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    alpha = 0.5
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]:
            u = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
            v = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)
            U.append(u)
            V.append(v)
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    plt.figure(figsize=(10, 10))
    plt.streamplot(X, Y, U, V, density=2)
    plt.title("Phase portrait")
    plt.show()

    # TASK 3.2
    start = x = np.array([0, 2])
    ts = 1e-3
    U, V = [], []
    for i in range(1000):
        x1, x2 = x[0], x[1]
        u = alpha * x1 - x2 - x1 * (x1 ** 2 + x2 ** 2)
        v = x1 + alpha * x2 - x2 * (x1 ** 2 + x2 ** 2)
        x_new = x + ts * np.array([u, v])
        U.append(u)
        V.append(v)
        x = x_new
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    plt.figure(figsize=(10, 10))
    plt.streamplot(X, Y, U, V, density=2)
    plt.title("Orbit")
    plt.show()
