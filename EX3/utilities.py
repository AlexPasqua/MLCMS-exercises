import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, core
from typing import Tuple
from sympy.solvers import solve


def create_phase_portrait_matrix(A: np.ndarray, alpha: float, title_suffix: str, save_plots=False,
                                 save_path: str = None, display=True):
    """
    Plots the phase portrait of the linear system Ax, where A is a 2x2 matrix and x is a 2-dim vector
    :param A: system's 2x2 matrix
    :param alpha: system's parameter
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    """
    w = 3  # width
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of A: ", eigenvalues)
    # linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(10, 10))
    plt.streamplot(X, Y, U, V, density=.7)
    plt.title(f"alpha: {alpha}, lambda_1: {eigenvalues[0]:.4f}, lambda_2: {eigenvalues[1]:.4f} - {title_suffix}")
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)


def create_bifurcation_diagram_1D(f_to_solve: str, min_alpha=-2, max_alpha=2.1, alpha_step=0.1):
    """
    Plots the bifurcation diagram of a given function to solve
    :param f_to_solve: string to represent the function to solve, based on having 'x' as variable
    """
    x = Symbol('x')
    alphas = np.arange(min_alpha, max_alpha, alpha_step)
    alphas = [round(alpha, 7) for alpha in alphas]
    fixed_points = {}
    fixed_points_rel_alphas = {}
    for alpha in alphas:
        sol = solve(eval(f_to_solve), x)
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
            if not isinstance(fixed_points[i][j], core.numbers.Float) and not isinstance(fixed_points[i][j],
                                                                                         core.numbers.Integer):
                fixed_points[i][j] = None
        plt.scatter(fixed_points_rel_alphas[i], fixed_points[i])
    plt.xlim(alphas[0], alphas[-1])
    plt.title(f_to_solve)
    plt.show()


def create_phase_portrait_derivative(system: list, alpha: float, title_suffix: str, save_plots=False,
                                     save_path: str = None, display=True, fig_size=10):
    """
    Plots the phase portrait of the given 'system', where 'system' is a 2 dimensional system given as couple of strings
    :param system: system ODEs
    :param alpha: system's parameter
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    :param fig_size: gives width and height of plotted figure
    """
    # check if given parameters are acceptable
    if len(system) != 2:
        print("A 2 ODE system is required.")
        return
    # setting up grid width/height
    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # dynamic system parameter, responsible for the change in behaviour
    alpha = alpha
    U, V = [], []
    for x2 in X[0]:
        for x1 in Y[:, 0]:
            u = eval(system[0])
            v = eval(system[1])
            U.append(u)
            V.append(v)
    U = np.reshape(U, X.shape)
    V = np.reshape(V, X.shape)
    plt.figure(figsize=(fig_size, fig_size))
    plt.streamplot(X, Y, U, V, density=2)
    plt.title(f"alpha: {alpha} - {title_suffix}")
    if display:
        plt.show()
    if save_plots:
        plt.savefig(save_path)


def plot_traj(system: list, start_pos, alpha=1, ts=1e-3, iter_num=10000, fig_size=10):
    """
    Function to plot a trajectory for a certain amount of iterations, given a starting point and a ode system.
    @param system: 2 ode system to get movement vector
    @param start_pos: initial coordinates of point
    @param alpha: system's parameter
    @param ts: step length
    @param iter_num: number of iterations to execute
    @param fig_size: gives width and height of plotted figure
    """

    w = 3
    X, Y = [], []  # lists to store positioning update
    x = start_pos
    for i in range(int(iter_num)):
        x1, x2 = x[0], x[1]
        X.append(x1)
        Y.append(x2)
        # find vectors for next movement
        u = eval(system[0])
        v = eval(system[1])
        # execute step
        x = x + ts * np.array([u, v])
    plt.figure(figsize=(fig_size, fig_size))
    # plot trajectory
    plt.plot(X, Y)
    plt.scatter(X[0], Y[0])  # put the starting point in evidence
    plt.title(f"Trajectory [{start_pos[0]}, {start_pos[1]}]")
    plt.xlim(-w, w)
    plt.ylim(-w, w)
    plt.show()


def plot_cusp_bifurcation(alpha_two_limit: float, x_limit: float, n: int, fig_size=10):
    """
    Function to plot the cusp bifurcation, in 2D and 3D.
    @param alpha_two_limit: value to delimit the upper bound of alpha2 sampling (0, alpha_two_limit)
    @param x_limit: value to delimit the upper and lower bound of x sampling (-x_limit, x_limit)
    @param n: number of samples to get
    @param fig_size: gives width and height of plotted figure
    """
    # sample alpha_2 and x n times
    alphas_two_sampled = np.random.uniform(0, alpha_two_limit, n)
    x_values_sampled = np.random.uniform(-x_limit, x_limit, n)
    # create placeholders for final alphas_1, alphas_2 and x values and fill them
    alphas_two = []
    alphas_one = []
    x_values = []
    for a2, x in zip(alphas_two_sampled, x_values_sampled):
        # evaluate alphas_1 reversing the equation (dx = a1 + a2*x - x**3 = 0)
        alphas_one.append(-a2 * x + x ** 3)
        alphas_two.append(a2)
        x_values.append(x)
    # round the values otherwise the bifurcation is not evident because of floating point precision (this causes some little artifacts)
    rounding = 3
    alphas_one = [round(a1, rounding) for a1 in alphas_one]
    alphas_two = [round(a2, rounding) for a2 in alphas_two]
    # create dictionary to store how many times every alpha couple is encountered (if more than one than instead of a single steady point there are more!)
    alphas_by_num = {}
    for a1, a2 in zip(alphas_one, alphas_two):
        if (a1, a2) not in alphas_by_num:
            alphas_by_num[(a1, a2)] = 1
        else:
            alphas_by_num[(a1, a2)] += 1
    # divide the couples so to plot them differently, divided by steady point number
    a1_to_plot = {}
    a2_to_plot = {}
    for k, v in alphas_by_num.items():
        if v - 1 in a1_to_plot:
            a1_to_plot[v - 1].append(k[0])
            a2_to_plot[v - 1].append(k[1])
        else:
            a1_to_plot[v - 1] = [k[0]]
            a2_to_plot[v - 1] = [k[1]]
    # plot with blue color the couples with only one steady point, with orange the rest (they should only be two, but the rounding creates artifacts)
    for k in a1_to_plot.keys():
        c = 'blue' if k == 0 else 'orange'
        plt.scatter(a1_to_plot[k], a2_to_plot[k], s=0.5, c=c)
    plt.xlabel("alpha_1")
    plt.ylabel("alpha_2")
    plt.show()
    # also plot the 3D situation for completeness, keeping alpha_1 and alpha_2 at the basis, x on the third axis
    ax = plt.axes(projection='3d')
    ax.scatter(alphas_one, alphas_two, x_values, cmap='viridis', c=alphas_two)
    ax.set_title("3D cusp bifurcation - alpha_1 and alpha_2 at the base")
    plt.show()


def logistic(r, x):
    """ Computes the logistic function (for the system "logistic map") """
    return r * x * (1 - x)


def logistic_map_cobweb_plot(r, x0, n):
    # Plot the function and the
    # y=x diagonal line.
    t = np.linspace(0, 1)
    plt.plot(t, logistic(r, t), 'k', lw=2)
    plt.plot([0, 1], [0, 1], 'k', lw=2)

    # Recursively apply y=f(x) and plot two lines:
    # (x, x) -> (x, y)
    # (x, y) -> (y, y)
    x = x0
    for i in range(n):
        y = logistic(r, x)
        # Plot the two lines.
        plt.plot([x, x], [x, y], 'k', lw=1)
        plt.plot([x, y], [y, y], 'k', lw=1)
        # Plot the positions with increasing
        # opacity.
        plt.plot([x], [y], 'ok', ms=10, alpha=(i + 1) / n)
        x = y

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"$r={r:.1f}, \, x_0={x0:.1f}$")
    plt.show()


def plot_logistic_map_bifurcations(min_r: float = 0., max_r: float = 4.):
    """
    Plots the bifurcations of the logistic map system
    :param min_r: minimum r to test the system with
    :param max_r: maximum r to test the system with
    """
    n = 5000  # number of runs
    r = np.linspace(min_r, max_r, n)
    iterations = 1000  # number of steps per run
    last = 100  # number of steps, from the last one, to plot for each run
    x = 1e-5 * np.ones(n)  # starting x for each run

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True)
    for i in range(iterations):
        x = logistic(r, x)  # compute a step for each run
        # if we're in the last 'last' iterations, plot the points for each run
        if i >= (iterations - last):
            ax.plot(r, x, ',k', alpha=.25)
    ax.set_title("Logistic map bifurcation diagram")
    ax.set_xlabel("r")
    ax.set_ylabel("x")
    plt.tight_layout()
    plt.show()


def lorenz_step(start_pt: Tuple[int, int, int], s=10, b=2.667, r=28):
    """
    :param start_pt: starting point in 3-dimensional space
    :param s, r, b: parameters of the Lorenz attractor (usually indicated with the Greek letters sigma, ro, beta)
    :returns: x_dot, y_dot, z_dot: values of the lorenz attractor's partial derivatives at the point start_pt
    """
    # break down the starting point start_pt into 3 variables representing a dimension each, for simplicity
    x, y, z = start_pt[0], start_pt[1], start_pt[2]
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


def lorenz_traj(dt=.02, tot_sim_time=1000, start_x=(10, 10, 10), s=10, b=2.667, r=28, plot=True):
    # simulation steps necessary given the total time and the width of the time step (dt)
    steps = int(tot_sim_time // dt)

    # Need one more slot for the initial values
    xs = np.empty(steps + 1)
    ys = np.empty(steps + 1)
    zs = np.empty(steps + 1)

    # Set starting point
    xs[0], ys[0], zs[0] = start_x[0], start_x[1], start_x[2]

    # Step through "time", calculating the partial derivatives at the current point
    # and using them to estimate the next point
    for i in range(steps):
        x_dot, y_dot, z_dot = lorenz_step((xs[i], ys[i], zs[i]), s, b, r)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    # plot
    ax = plt.figure(figsize=(8, 8)).add_subplot(projection='3d')
    ax.plot(xs, ys, zs, lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()

    # create list of the 3D points along the trajectory
    traj_pts = [(xs[i], ys[i], zs[i]) for i in range(len(xs))]
    return np.array(traj_pts)


if __name__ == '__main__':
    lorenz_traj()
