import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
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
