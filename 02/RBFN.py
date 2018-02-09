import numpy as np


class RBFN:
    def __init__(self):
        pass

    def train(self, X, T, nodes, vec_mu, vec_sigma, learning_rule, batch):
        """
        INPUT:
        @X - numpy array: representing the input data
        @T - numpy array: representing the target values
        @nodes - integer: number of basis functions in the network
        @vec_mu - numpy array: the centroids of each basis function
        @vec_sigma - numpu array: the variance for each basis function
        @learning_rule - string: specifies the type of learning rule
                                 that is used
        @batch - boolean: Indicates whether it is using batch or
                          incremental learning

        OUTPUT:
        @weights - numpy array: The calculated weights for each
                                basis function
        """
        K = np.zeros((len(X), nodes)).reshape(len(X), nodes)
        for node in range(nodes):
            K[:, node] = (np.exp(-((X - vec_mu[node])**2)/(2*vec_sigma[node]**2))).reshape(len(X), )

        if learning_rule == 'least_squares':
            if batch:
                weights = self._least_squares(K, T)
                return K.dot(weights)
            else:
                pass

        if learning_rule == 'delta':
            if batch:
                pass
            else:
                pass

    def _least_squares(self, K, T):
        return np.dot(np.linalg.pinv(K), T)

    def error(self, Y, T):
        return sum(np.absolute(Y - T)) / len(Y)

    def __str__(self):
        return "Radial Basis Function network"

