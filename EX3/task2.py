import copy

import matplotlib.pyplot as plt
import numpy as np
from utilities import create_bifurcation_diagram_1D

from sympy import Symbol, core
from sympy.solvers import solve

if __name__ == '__main__':
    """
            Plots the bifurcation diagram of a given function to solve
            :param f_to_solve: string to represent the function to solve, based on having 'x' as variable
        """
    x = Symbol('x')
    alphas = np.arange(-2, 2.1, 0.1)
    alphas = [round(alpha, 2) for alpha in alphas]
    fixed_points = {}
    fixed_points_rel_alphas = {}
    for alpha in alphas:
        sol = solve(alpha - x**2, x)
        print(f"{alpha} - {sol}")
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
            if not isinstance(fixed_points[i][j], core.numbers.Float) and not isinstance(fixed_points[i][j], core.numbers.Integer):
                fixed_points[i][j] = None
        plt.scatter(fixed_points_rel_alphas[i], fixed_points[i])
    plt.xlim(alphas[0], alphas[-1])
    plt.show()