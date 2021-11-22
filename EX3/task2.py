import copy

import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol
from sympy.solvers import solve


if __name__ == '__main__':
    x = Symbol('x')
    alphas = np.arange(-1, 1.1, 0.1)
    xes = []
    for alpha in alphas:
        sol = solve(alpha - x**2, x)
        xes.append(sol)

    # postprocessing
    for i, sol in enumerate(xes):
        sol = copy.deepcopy(sol)
        for s in sol:
            if not isinstance(s, float):
                xes[i].remove(s)

    plt.scatter(alphas, xes)
    # plt.xlim(alphas[0], alphas[-1])
    plt.show()
