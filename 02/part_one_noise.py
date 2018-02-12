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
# 
#     plt.plot(sin_T_train, c='g')
#     plt.show()
#     plt.plot(sin_T_test, c='y')
#     plt.show()

    nodes = 6
    learning_rule = 'least_squares'
#     learning_rule = 'delta'    
    batch = True
    # mean of each basis function
    vec_mu = [0.05, 0.9, 2.35, 3.9, 5.5, 6.2]
    #variance of each basis function
    vec_sigma = [.5] * nodes
    epochs = 50
    eta = 0.01
    # Create a RBF network
    R = RBFN(eta)
    # train the network
    R.train(X_train, sin_T_train, nodes, vec_mu, vec_sigma, learning_rule, batch, epochs)
    print(R.weights)
    Y = R.predict(X_test)
    
    #print error
    print('error:', R.error(Y, sin_T_train))

    fig = plt.figure()
    ax = plt.subplot(111)
    plt.title("learing rule: "+learning_rule + \
                ". Eta="+str(R.eta) + ". Nodes="+str(nodes) + \
                ". sigma="+str(vec_sigma[0])+\
                ". Epochs="+str(epochs))
    ax.plot(np.arange(0, 2*np.pi, 0.1), sin_T_test, label="true")
    ax.plot(np.arange(0,2*np.pi, 0.1), Y, label="model")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
