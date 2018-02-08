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
    nodes = 4
    learning_rule = 'least_squares'
    vec_mu = [.8, 2.2, 3.9, 5.4]
    vec_sigma = [.5, .5, .5, .5]
    Y = R.train(X_train, sin_T_train, nodes, vec_mu, vec_sigma, learning_rule)

    plt.plot(X_train, sin_T_train)
    plt.plot(X_train, Y)
    plt.show()







    print(R)
    


if __name__ == '__main__':
    main()
