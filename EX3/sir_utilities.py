import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


def mu(b, I, mu0, mu1):
    """
    Recovery rate.
    Parameters:
    -----------
    b
        hospital beds per 10,000 persons
    I
        number of infected
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    """
    # recovery rate, depends on mu0, mu1, b
    mu = mu0 + (mu1 - mu0) * (b / (I + b))
    return mu


def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.
    Parameters:
    -----------
    beta
        average number of adequate contacts per unit time with infectious individuals
    d
        natural death rate
    nu
        disease induced death rate
    mu1
        Maximum recovery rate
    """
    return beta / (d + nu + mu1)


def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.
    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """
    c0 = b ** 2 * d * A
    c1 = b * ((mu0 - mu1 + 2 * d) * A + (beta - nu) * b * d)
    c2 = (mu1 - mu0) * b * nu + 2 * b * d * (beta - nu) + d * A
    c3 = d * (beta - nu)
    res = c0 + c1 * I + c2 * I ** 2 + c3 * I ** 3
    for single_res in res:
        if round(single_res,7) == 0:
            print("WTF", I)
    return res


def model(t, y, mu0, mu1, beta, A, d, nu, b):
    """
    SIR model including hospitalization and natural death.

    Parameters:
    -----------
    mu0
        Minimum recovery rate
    mu1
        Maximum recovery rate
    beta
        average number of adequate contacts per unit time with infectious individuals
    A
        recruitment rate of susceptibles (e.g. birth rate)
    d
        natural death rate
    nu
        disease induced death rate
    b
        hospital beds per 10,000 persons
    """
    S, I, R = y[:]
    m = mu(b, I, mu0, mu1)

    dSdt = A - d * S - (beta * S * I) / (S + I + R)
    dIdt = - (d + nu) * I - m * I + (beta * S * I) / (S + I + R)
    dRdt = m * I - d * R

    return [dSdt, dIdt, dRdt]


def plot_SIR_variables(sol, b, mu0, mu1, beta, A, d, nu):
    """
    Create three plots:
        first one shows the evolution of the S, I, R variables in the SIR model
        second one temporally compares the recovery rate mu with the number of infected I
        third one shows the bifurcation indicator function for this scenario
    :param sol: solution of the system obtained with scipy.integrate.solve_ivp
    :param b: number of beds per 10000 persons in the SIR model
    :param mu0: minimum recovery rate in the SIR model
    :param mu1: maximum recovery rate in the SIR model
    :param beta: average number of adequate contacts per unit time with infectious individuals in the SIR model
    :param A: birth rate in the SIR model
    :param d: per capita natural deaths in the SIR model
    :param nu: per capita disease-induced death rate in the SIR model
    """
    # plot evolution of S, I, R variables
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(sol.t, sol.y[0] - 0 * sol.y[0][0], label='1E0*susceptible')
    ax[0].plot(sol.t, 1e3 * sol.y[1] - 0 * sol.y[1][0], label='1E3*infective')
    ax[0].plot(sol.t, 1e1 * sol.y[2] - 0 * sol.y[2][0], label='1E1*removed')
    ax[0].set_xlim([0, 500])
    ax[0].legend()
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$S,I,R$")

    # plot comparison between recovery rate and number of infected
    ax[1].plot(sol.t, mu(b, sol.y[1], mu0, mu1), label='recovery rate')
    ax[1].plot(sol.t, 1e2 * sol.y[1], label='1E2*infective')
    ax[1].set_xlim([0, 500])
    ax[1].legend()
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\mu,I$")

    # plot indicator function
    I_h = np.linspace(-0.005, 0.025, 100)
    ax[2].plot(I_h, h(I_h, mu0, mu1, beta, A, d, nu, b))
    ax[2].plot(I_h, 0 * I_h, 'r:')
    ax[2].set_title("Indicator function h(I)")
    ax[2].set_xlabel("I")
    ax[2].set_ylabel("h(I)")
    plt.setp(ax[2].get_xticklabels(), rotation=30, horizontalalignment='right')
    fig.tight_layout()

def plot_SIR_trajectories_actuator(sol, ax, s, color, colors_marker, two_dim):
    """
    function called by plot_SIR_trajectories to handle matplotlib execution
    :param sol: solution of ode system, containing all screenshots of S I and R
    :param ax: where to plot
    :param s: determines how big the start and end markers will be per plot
    :param color: defining the trajectory color
    :param colors_marker: defining the marker color
    :param two_dim: determining if plot in 2d or 3d
    :return:
    """
    # differentiate and color the plot of a single point
    if not two_dim:
        ax.scatter(sol.y[0], sol.y[1], sol.y[2], s=1, c=color, zorder=-1)
        ax.scatter(sol.y[0][0], sol.y[1][0], sol.y[2][0], marker='x', s=s, c=colors_marker, zorder=0)
        ax.scatter(sol.y[0][-1], sol.y[1][-1], sol.y[2][-1], marker='D', s=s, c=colors_marker, zorder=1)
    else:
        ax.scatter(sol.y[0], sol.y[1], s=1, c=color, zorder=-1)
        ax.scatter(sol.y[0][0], sol.y[1][0], marker='x', s=s, c=colors_marker, zorder=0)
        ax.scatter(sol.y[0][-1], sol.y[1][-1], marker='D', s=s, c=colors_marker, zorder=1)

def plot_SIR_trajectories(t_0, b, mu0, mu1, beta, A, d, nu, rtol=1e-8, atol=1e-8, s=200, figsize=20, two_dim=False):
    """
    function to plot trajectories of the three initial points proposed by task5.3
    :param t_0: initial time
    :param b: number of beds per 10000 persons in the SIR model
    :param mu0: minimum recovery rate in the SIR model
    :param mu1: maximum recovery rate in the SIR model
    :param beta: average number of adequate contacts per unit time with infectious individuals in the SIR model
    :param A: birth rate in the SIR model
    :param d: per capita natural deaths in the SIR model
    :param nu: per capita disease-induced death rate in the SIR model
    :param rtol: tolerance to avoid qualitatively wrong results
    :param atol: tolerance to avoid qualitatively wrong results
    :param s: determines how big the start and end markers will be per plot
    :param figsize: figure size width/height
    :param two_dim: True will plot the trajectory on the S,I space - False will plot the trajectory on the S,I,R space
    :return:
    """
    # initialize the figure
    fig = plt.figure(figsize=(figsize, figsize))
    if two_dim:
        ax = fig.add_subplot(111)
    else:
        ax = fig.add_subplot(111, projection="3d")
    # sample time instants
    NT = 10000
    time = np.linspace(t_0, 50000, NT)

    # associate plots of the 3 points to colors
    # first plot is RED -> its markers are BLACK
    # first plot is GREEN -> its markers are #cc00ff
    # first plot is BLUE -> its markers are ORANGE
    colors_markers = ["black", "#cc00ff", "orange"]
    colors = ["red", "green", "blue"]

    # work on first point
    SIM0 = [195.3, 0.052, 4.4]
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b), method='DOP853', rtol=rtol, atol=atol)
    plot_SIR_trajectories_actuator(sol, ax, s, colors[0], colors_markers[0], two_dim)

    # work on second point
    SIM0 = [195.7, 0.03, 3.92]  
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b), method='DOP853', rtol=rtol, atol=atol)
    plot_SIR_trajectories_actuator(sol, ax, s, colors[1], colors_markers[1], two_dim)

    # work on third point
    SIM0 = [193, 0.08, 6.21]  
    sol = solve_ivp(model, t_span=[time[0], time[-1]], y0=SIM0, t_eval=time, args=(mu0, mu1, beta, A, d, nu, b), method='DOP853', rtol=rtol, atol=atol)
    plot_SIR_trajectories_actuator(sol, ax, s, colors[2], colors_markers[2], two_dim)

    # set dimension labels
    ax.set_xlabel("S")
    ax.set_ylabel("I")
    if not two_dim:
        ax.set_zlabel("R")

    # set title and plot
    ax.set_title(f"SIR trajectory b: {b}")
    fig.tight_layout()


