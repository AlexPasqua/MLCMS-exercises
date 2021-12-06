import matplotlib.pyplot as plt
import numpy as np


def mu(b, I, mu0, mu1):
    """Recovery rate.

    """
    # recovery rate, depends on mu0, mu1, b
    mu = mu0 + (mu1 - mu0) * (b / (I + b))
    return mu


def R0(beta, d, nu, mu1):
    """
    Basic reproduction number.
    """
    return beta / (d + nu + mu1)


def h(I, mu0, mu1, beta, A, d, nu, b):
    """
    Indicator function for bifurcations.
    """
    c0 = b ** 2 * d * A
    c1 = b * ((mu0 - mu1 + 2 * d) * A + (beta - nu) * b * d)
    c2 = (mu1 - mu0) * b * nu + 2 * b * d * (beta - nu) + d * A
    c3 = d * (beta - nu)
    res = c0 + c1 * I + c2 * I ** 2 + c3 * I ** 3
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


def plot_SIR_graphs(sol, b, mu0, mu1, beta, A, d, nu):
    """
    this functo
    @return:
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].plot(sol.t, sol.y[0] - 0 * sol.y[0][0], label='1E0*susceptible')
    ax[0].plot(sol.t, 1e3 * sol.y[1] - 0 * sol.y[1][0], label='1E3*infective')
    ax[0].plot(sol.t, 1e1 * sol.y[2] - 0 * sol.y[2][0], label='1E1*removed')
    ax[0].set_xlim([0, 500])
    ax[0].legend()
    ax[0].set_xlabel("time")
    ax[0].set_ylabel(r"$S,I,R$")

    ax[1].plot(sol.t, mu(b, sol.y[1], mu0, mu1), label='recovery rate')
    ax[1].plot(sol.t, 1e2 * sol.y[1], label='1E2*infective')
    ax[1].set_xlim([0, 500])
    ax[1].legend()
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"$\mu,I$")

    I_h = np.linspace(-0., 0.05, 100)
    ax[2].plot(I_h, h(I_h, mu0, mu1, beta, A, d, nu, b))
    ax[2].plot(I_h, 0 * I_h, 'r:')
    # ax[2].set_ylim([-0.1,0.05])
    ax[2].set_title("Indicator function h(I)")
    ax[2].set_xlabel("I")
    ax[2].set_ylabel("h(I)")

    fig.tight_layout()
