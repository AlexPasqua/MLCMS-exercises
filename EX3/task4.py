import numpy as np
import matplotlib.pyplot as plt

from utilities import logistic, plot_logistic_map_bifurcations

if __name__ == '__main__':
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    plot_logistic_map_bifurcations(2.5, .1, 10, ax=ax1)
    plot_logistic_map_bifurcations(3.5, .1, 10, ax=ax2)
    plt.show()

    # TASK 4.1
    n = 5000
    r = np.linspace(0, 2, n)
    iterations = 1000
    last = 100
    x = 1e-5 * np.ones(n)

    fig, ax = plt.subplots(1, 1, figsize=(8, 9), sharex=True)
    for i in range(iterations):
        x = logistic(r, x)
        # We display the bifurcation diagram.
        if i >= (iterations - last):
            ax.plot(r, x, ',k', alpha=.25)
    ax.set_title("Bifurcation diagram")
    plt.tight_layout()
    plt.show()

    # TASK 4.2
    n = 5000
    r = np.linspace(2, 4, n)
    iterations = 1000
    last = 100
    x = 1e-5 * np.ones(n)

    fig, ax = plt.subplots(1, 1, figsize=(8, 9), sharex=True)
    for i in range(iterations):
        x = logistic(r, x)
        # We display the bifurcation diagram.
        if i >= (iterations - last):
            ax.plot(r, x, ',k', alpha=.25)
    # ax.set_xlim(2.5, 4)
    ax.set_title("Bifurcation diagram")
    plt.tight_layout()
    plt.show()

    # TASK 4.2
    n = 5000
    r = np.linspace(0, 4, n)
    iterations = 1000
    last = 100
    x = 1e-5 * np.ones(n)

    fig, ax = plt.subplots(1, 1, figsize=(8, 9), sharex=True)
    for i in range(iterations):
        x = logistic(r, x)
        # We display the bifurcation diagram.
        if i >= (iterations - last):
            ax.plot(r, x, ',k', alpha=.25)
    ax.set_ylim(0, 1)
    ax.set_title("Bifurcation diagram")
    plt.tight_layout()
    plt.show()
