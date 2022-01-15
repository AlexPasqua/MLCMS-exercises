from vector_fields import read_vectorfield_data, estimate_vectors, solve_trajectory, create_phase_portrait_matrix
from function_approximation import approx_lin_func, approx_nonlin_func, plot_func_over_data, compute_bases
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import math

def solve_trajectory(x0, x1, funct, find_best_dt=False, end_time=0.1, plot=False):
    """
    Solves initial value point problem for a dataset, up to a certain moment in time
    :param x0: the data at time 0
    :param x1: the data at unknown time step after 0
    :param funct: to get derivative for next steps generation
    :param end_time: end time for the simulation
    :param plot: boolean to produce a scatter plot of the trajectory (orange) with the final x1 points in blue
    :returns: points at time end_time
    """
    best_dt = -1
    best_mse = math.inf
    x1_pred = []
    t_eval = np.linspace(0, end_time, 100)
    sols = []
    for i in range(len(x0)):
        sol = solve_ivp(funct, [0, end_time], x0[i], t_eval=t_eval)
        x1_pred.append([sol.y[0,-1], sol.y[1,-1]])
        if find_best_dt:
            sols.append(sol.y)
        if plot:
            plt.scatter(x1[i,0], x1[i,1], c='blue', s=10)
            plt.scatter(sol.y[0,:],sol.y[1,:], c='orange', s=4)
    if find_best_dt:
        for i in range(len(t_eval)):
            pred = [[sols[el][0][i], sols[el][1][i]] for el in range(len(sols))]
            mse = np.linalg.norm(pred - x1)**2 / x1.shape[0]
            if mse < best_mse:
                best_mse = mse
                best_dt = t_eval[i]
    if plot:
        plt.rcParams["figure.figsize"] = (14,14)
        plt.show()
    return x1_pred, best_dt, best_mse

def rbf_approx(t, y):
    list_of_bases = np.empty(shape=(n_bases))
    for i, center_point in enumerate(centers):
        subtraction = np.subtract(center_point, y)  # note: center_point is a single point, points are many points -> broadcasting
        norm = np.linalg.norm(subtraction)
        basis = np.exp(-norm ** 2 / eps ** 2)
        list_of_bases[i] = basis
#     print(C.shape, list_of_bases.shape, (list_of_bases @ C).shape)
    return list_of_bases @ C
    
if __name__ == "__main__":
    # read the vector field data
    x0, x1 = read_vectorfield_data(base_filename="nonlinear_vectorfield_data")
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.scatter(x0[:,0], x0[:,1], s=1)
    ax.scatter(x1[:,0], x1[:,1], s=1)
    plt.show()

    dt = 0.1
    end_time=0.2
    v = estimate_vectors(dt, x0, x1)
    #for eps in (0.1, 0.5, 1, 2, 5, 10):
    eps=0.1
    results = []
    for n_bases in (100, 200): #, 300, 500, 750, 1000):
        print(f"EXECUTING WITH EPS: {eps} AND N_BASES: {n_bases}")
        centers = np.random.choice(x0.ravel(), replace=False, size=n_bases)
        C, res, _, _, _, eps, phi = approx_nonlin_func(data=(x0,v), n_bases=n_bases, eps=eps, centers=centers)
        print("Residual error:",res)
        x1_pred, best_dt, best_mse = solve_trajectory(x0, x1, rbf_approx, find_best_dt=True, end_time=end_time, plot=False)
        results.append(f"Best MSE value is found at time {best_dt} with MSE: {best_mse}")
    print(results)