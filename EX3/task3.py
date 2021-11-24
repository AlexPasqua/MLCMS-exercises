import matplotlib.pyplot as plt
import numpy as np


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
