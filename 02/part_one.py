from RBFN import *
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def main(eta):
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
    

    nodes = 6
#     learning_rule = 'least_squares'
    learning_rule = 'delta'    
    batch = False
    # mean of each basis function
    vec_mu = [0, 0.9, 2.25, 3.9, 5.5, 6.2]
    #variance of each basis function
    vec_sigma = [.5] * nodes
    epochs = 50
#     eta = 0.01
    # Create a RBF network
    R = RBFN(eta)
    # train the network
    R.train(X_train, sin_T_train, nodes, vec_mu, vec_sigma, learning_rule, batch, epochs)
    print(R.weights)
    Y = R.predict(X_train)
    
    # Use threshold for square(2x) function
    #Y = np.where(Y > 0, 1, -1)
    
    #print error
    print(R.error(Y, sin_T_train))

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.title("learing rule: "+learning_rule + \
                ". Eta="+str(R.eta) + ". Nodes="+str(nodes) + \
                ". sigma="+str(vec_sigma[0])+\
                ". Epochs="+str(epochs))
    ax.plot(X_train, sin_T_train, label="true")
    ax.plot(X_train, Y, label="model")
    plt.legend()
    plt.show()

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.title("error vs epoch. (sequential delta rule). Eta = "+str(R.eta))
    ax.plot(R.vec_errors, label = "error")
    plt.ylim((0,3))
    plt.legend()
    plt.show()
    
#     vec_mu = [0., .63, 2.35, 3.9, 5.5, 5.89, 6.2]
#     vec_mu = X_train
#     sigmas = np.arange(.1, 2, 0.001)
#     delta = 0.001
#     errors = []
#     for sigma in sigmas:
#         vec_sigma = [sigma] * nodes
#         Y = R.train(X_train, sin_T_train, nodes, vec_mu, vec_sigma, learning_rule)
#         error = R.error(Y, sin_T_train)
#         errors.append((error,sigma))
# #         if error < delta:
# #             plt.title("RBF network vs. sin(2*x). Error: {}".format(round(error[0], 8)))
# #             plt.plot(X_train, sin_T_train, c='b') 
# #             plt.plot(X_train, Y, c='r')
# #             plt.show()

#     err, sigma = min(errors)
#     vec_sigma = [sigma]*len(X_train)
#     print(err, sigma)
#     Y = R.train(X_train, sin_T_train, nodes, vec_mu, vec_sigma, learning_rule)
#     plt.title("RBF network vs. sin(2*x). Error: {}".format(round(err[0], 7)))

#     mpatches = "nodes: "+str(nodes) +' sigma: '+str(sigma)
#     true = plt.plot(X_train, sin_T_train, c='b')
#     pred = plt.plot(X_train, Y, c='r', label="nodes: "+str(nodes) +' sigma: '+str(sigma))
#     plt.show()


# if __name__ == '__main__':
#     main()
