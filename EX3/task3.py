import matplotlib.pyplot as plt
import numpy as np
from sympy import Symbol, core, Float
from sympy.solvers import solve


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
    # setting up grid width/height
    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # dynamic system parameter, responsible for the change in behaviour
    alpha = -1
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
    # plotting a trajectory starting from point [0.5, 0]
    plot_traj(np.array([0.5, 0]))


    # TASK 3.2
    # plotting a trajectory starting from point [0, 2]
    plot_traj(np.array([0, 2]))

    # TASK 3.3
    # x = Symbol('x')
    # alphas_one = np.arange(-2, 2.1, 0.01)
    # alphas_two = np.arange(-2, 2.1, 0.01)
    # fixed_points = {}
    # fixed_points_rel_alphas_one = {}
    # fixed_points_rel_alphas_two = {}
    # for a1 in alphas_one:
    #     print(a1)
    #     for a2 in alphas_two:
    #         sol = solve(a1 + a2*x - x**3, x)
    #         for i, single_sol in enumerate(sol):
    #             if i not in fixed_points:
    #                 fixed_points[i] = [single_sol]
    #                 fixed_points_rel_alphas_one[i] = [Float(a1)]
    #                 fixed_points_rel_alphas_two[i] = [Float(a2)]
    #             else:
    #                 fixed_points[i].append(single_sol)
    #                 fixed_points_rel_alphas_one[i].append(Float(a1))
    #                 fixed_points_rel_alphas_two[i].append(Float(a2))
    #
    # # postprocessing
    # ax = plt.axes(projection='3d')
    # for i in sorted(fixed_points.keys()):
    #     for j in range(len(fixed_points[i])):
    #         # print(type(fixed_points[i][j]), fixed_points[i][j])
    #         if not isinstance(fixed_points[i][j], core.numbers.Float):
    #             fixed_points[i][j] = 0
    #     print(len(fixed_points[i]), len(fixed_points_rel_alphas_one[i]), len(fixed_points_rel_alphas_two[i]))
    #     ax.plot3D(fixed_points_rel_alphas_one[i], fixed_points_rel_alphas_two[i], fixed_points[i])
    # plt.show()

    # TASK 3.3
    # dx = a1 + a2*x - x**3
    alphas_two = np.arange(-5, 5.1, 0.1)
    xs = np.arange(-5, 5.1, 0.1)
    a1s = []
    a2s = []
    xss = []
    for a2 in alphas_two:
        print(a2)
        for x in xs:
            a2s.append(a2)
            xss.append(x)
            a1s.append(-a2 * x + x ** 3)
    ax = plt.axes(projection='3d')
    print(len(a1s), "\n\n", len(a2s), "\n\n", len(xss))
    ax.scatter(a1s, a2s, xss)
    plt.show()
