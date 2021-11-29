import copy

import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol, core
from sympy.solvers import solve


if __name__ == '__main__':
    x = Symbol('x')
    alphas = np.arange(-2, 2.1, 0.1)
    fixed_points = {}
    fixed_points_rel_alphas = {}
    for alpha in alphas:
        sol = solve(alpha - x**2, x)
        for i, single_sol in enumerate(sol):
            if i not in fixed_points:
                fixed_points[i] = [single_sol]
                fixed_points_rel_alphas[i] = [alpha]
            else:
                fixed_points[i].append(single_sol)
                fixed_points_rel_alphas[i].append(alpha)

    # postprocessing
    for i in sorted(fixed_points.keys()):
        for j in range(len(fixed_points[i])):
            # print(type(fixed_points[i][j]), fixed_points[i][j])
            if not isinstance(fixed_points[i][j], core.numbers.Float):
                fixed_points[i][j] = None
        plt.plot(fixed_points_rel_alphas[i], fixed_points[i])
    plt.xlim(alphas[0], alphas[-1])
    plt.show()
