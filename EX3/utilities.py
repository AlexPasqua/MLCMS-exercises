import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sympy import Symbol, core
from sympy.solvers import solve


def create_phase_portrait(A: np.ndarray, alpha: float, title_suffix: str, save_plots=False, save_path: str = None, display=True):
    """
    Plots the phase portrait of the linear system Ax, where A is a 2x2 matrix and x is a 2-dim vector
    :param A: system's 2x2 matrix
    :param alpha: system's parameter
    :param title_suffix: suffix to add to the title (after the value of alpha and A's eigenvalues
    :param save_plots: if True, saves the plots instead of displaying them
    :param save_path: path where to save the plots if save_plots is True
    :param display: if True, display the plots
    """
    w = 3   # width
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    eigenvalues = np.linalg.eigvals(A)
    print("Eigenvalues of A: ", eigenvalues)
    # linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(10, 10))
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    # ax0.set_title('Streamplot for linear vector field A*x')
    # ax0.set_aspect(1)
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
            if not isinstance(fixed_points[i][j], core.numbers.Float) and not isinstance(fixed_points[i][j], core.numbers.Integer):
                fixed_points[i][j] = None
        plt.scatter(fixed_points_rel_alphas[i], fixed_points[i])
    plt.xlim(alphas[0], alphas[-1])
    plt.show()

if __name__ == '__main__':
    # unstable focus
    alpha = 0.1
    A = np.array([
        [alpha, alpha],
        [-1/4, 0]
    ])
    create_phase_portrait(A, title_suffix="unstable focus")

    # unstable saddle
    alpha = -0.2
    A = np.array([
        [-alpha, alpha],
        [-1/4, 0]
    ])
    create_phase_portrait(A, title_suffix="unstable saddle")

    # unstable node
    alpha = 2
    A = np.array([
        [alpha, alpha],
        [-1/4, 0]
    ])
    create_phase_portrait(A, title_suffix="unstable node")

    # stable node
    alpha = -0.25
    A = np.array([
        [alpha, 0],
        [0, -1]
    ])
    create_phase_portrait(A, title_suffix="node stable")

    # focus stable
    alpha = 0.1
    A = np.array([
        [-alpha, alpha],
        [-1/4, 0]
    ])
    create_phase_portrait(A, title_suffix="stable focus")

