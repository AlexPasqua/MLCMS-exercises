import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/plot_streamplot.html
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    w = 3
    Y, X = np.mgrid[-w:w:100j, -w:w:100j]
    # Y, X = np.mgrid[0:3, 0:5]

    # alpha is the system's parameter,
    # to get all the phase portraits in the picture from the book, change alpha and also the sign of the -1/4 element
    alpha = 0.1
    A = np.array([
        [alpha, alpha],
        [-1/4, 0]

    ])
    print("Eigenvalues of A: ", np.linalg.eigvals(A))
    # example linear vector field A*x
    UV = A @ np.row_stack([X.ravel(), Y.ravel()])
    print(X.ravel().shape, Y.ravel().shape, np.row_stack([X.ravel(), Y.ravel()]).shape)
    print(UV.shape)
    U = UV[0, :].reshape(X.shape)
    V = UV[1, :].reshape(X.shape)
    fig = plt.figure(figsize=(10, 10))
    # gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[1, 1, 2])
    # Varying density along a streamline
    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.streamplot(X, Y, U, V, density=[0.5, 1])
    # ax0.set_title('Streamplot for linear vector field A*x')
    # ax0.set_aspect(1)
    plt.streamplot(X, Y, U, V, density=.1)
    plt.title('Streamplot for linear vector field A*x')
    plt.show()
