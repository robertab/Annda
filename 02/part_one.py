from RBF import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


def main():
    """
    Assignment - Part 1,

    3.1
    Batch mode training using least squares - supervised
    learning of network weights
    """
    # Training data
    X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1, 1)
    sin_T_train = np.sin(2*X_train).reshape(-1, 1)
    square_T_train = signal.square(2*X_train).reshape(-1, 1)
    # Test data
    X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1, 1)
    sin_T_test = np.sin(2*X_test).reshape(-1, 1)
    square_T_test = signal.square(2*X_test).reshape(-1, 1)
    
    # Create a RBF network
    R = RBF()

    # train on the data
    nodes = 7
    learning_rule = 'least_squares'
    vec_mu = [0., .78, 2.35, 3.9, 5.5, 5.89, 6.2]
    # vec_sigma = [.51, .51, .51, .51]
    sigmas = np.arange(.48, .63, 0.001)
    delta = 0.01
    errors = []
    for sigma in sigmas:
        vec_sigma = [sigma] * nodes
        Y = R.train(X_train, sin_T_train, nodes, vec_mu, vec_sigma, learning_rule)
        error = R.error(Y, sin_T_train)
        errors.append(error)
        print(error, sigma)
        if error < delta:
            plt.title("RBF network vs. sin(2*x). Error: {}".format(round(error[0], 8)))
            plt.plot(X_train, sin_T_train, c='b')
            plt.plot(X_train, Y, c='r')
            plt.show()

    print(min(error))
    # plt.title("RBF network vs. sin(2*x). Error: {}".format(round(error[0], 3)))
    # plt.plot(X_train, sin_T_train, c='b')
    # plt.plot(X_train, Y, c='r')
    # plt.show()


if __name__ == '__main__':
    main()
