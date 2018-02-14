from RBFN import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100)


def main():
    """
    Assignment - Part 1,

    3.1
    Batch mode training using least squares - supervised
    learning of network weights
    """
    # Training data
    X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1, 1)
    X_train = X_train + np.random.normal(0, 0.1, len(X_train)).reshape(-1,1)
    sin_T_train = np.sin(2*X_train).reshape(-1, 1)
    sin_T_train = sin_T_train + np.random.normal(0, 0.1, len(sin_T_train)).reshape(-1,1)
    
#     # Test dat
    X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1, 1)
    X_test = X_test + np.random.normal(0, 0.1, len(X_test)).reshape(-1,1)
    sin_T_test = np.sin(2*X_test).reshape(-1, 1)
    sin_T_test = sin_T_test  + np.random.normal(0, 0.1, len(sin_T_test)).reshape(-1,1)
 

    # NODES = list(range(10,20))
    NODES = list(range(63, 64))
    etas = np.arange(0,0.1,0.001)
    etas = np.array([0.001, 0.01, 0.05, 0.1])
    # learning_rule = 'least_squares'
    learning_rule = 'delta'    
    batch = False
    strategy = "k_means" #
    #variance of each basis function
    epochs = 500
    # Create a RBF network
    R = RBFN()
    # train the network
    # mat_res = np.ones((len(NODES), len(etas)))
    # for i, nodes in enumerate(NODES):
    #     for j, eta in enumerate(etas):
    #         vec_sigma = [0.7] * nodes
    #         R.train(X_train, sin_T_train, nodes, vec_sigma,
    #                 learning_rule, batch,
    #                 epochs, eta, strategy)
    #         Y = R.predict(X_test)
    #         mat_res[i, j] = R.error(Y, sin_T_test)
    # print(mat_res)
    for eta in etas:
        vec_sigma = [0.5] * 11        
        R.train(X_train, sin_T_train, 11, vec_sigma, learning_rule, batch, epochs, eta, strategy)
        fig = plt.figure()
        ax = plt.subplot(111)
        plt.title("error vs epoch. (sequential delta rule). Eta = "+str(eta))
        ax.plot(R.vec_errors, label = "error")
        plt.ylim((0,3))
        plt.legend()
        plt.show()

    #print error
#     fig = plt.figure()
#     ax = plt.subplot(111)
#     plt.title("learing rule: "+learning_rule + \
#                 ". Eta="+str(R.eta) + ". Nodes="+str(nodes) + \
#                 ". sigma="+str(R.vec_sigmas[0])+\
#                 ". Epochs="+str(epochs))
#     ax.plot(np.arange(0, 2*np.pi, 0.1), sin_T_test, label="true")
#     ax.plot(np.arange(0,2*np.pi, 0.1), Y, label="model")
#     plt.legend()
#     plt.show()

if __name__ == '__main__':
    main()
