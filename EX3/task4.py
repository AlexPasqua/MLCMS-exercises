import numpy as np
import matplotlib.pyplot as plt

from utilities import lorenz, logistic_map_cobweb_plot

if __name__ == '__main__':
    # dt = 0.01
    # num_steps = 10000
    #
    # # Need one more for the initial values
    # xs = np.empty(num_steps + 1)
    # ys = np.empty(num_steps + 1)
    # zs = np.empty(num_steps + 1)
    #
    # # Set initial values
    # xs[0], ys[0], zs[0] = (0., 1., 1.05)
    #
    # # Step through "time", calculating the partial derivatives at the current point
    # # and using them to estimate the next point
    # for i in range(num_steps):
    #     x_dot, y_dot, z_dot = lorenz((xs[i], ys[i], zs[i]))
    #     xs[i + 1] = xs[i] + (x_dot * dt)
    #     ys[i + 1] = ys[i] + (y_dot * dt)
    #     zs[i + 1] = zs[i] + (z_dot * dt)
    #
    # # Plot
    # ax = plt.figure().add_subplot(projection='3d')
    #
    # ax.plot(xs, ys, zs, lw=0.5)
    # ax.set_xlabel("X Axis")
    # ax.set_ylabel("Y Axis")
    # ax.set_zlabel("Z Axis")
    # ax.set_title("Lorenz Attractor")
    # plt.show()
    # exit()

    # TODO: decidi cosa fare con questa sezione -> è quella cosa che fa vedere i punti che saltano di qua e di là
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    logistic_map_cobweb_plot(2.5, .1, 10, ax=ax1)
    logistic_map_cobweb_plot(3.5, .1, 10, ax=ax2)
    plt.show()
