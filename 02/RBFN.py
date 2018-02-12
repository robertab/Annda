import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
np.random.seed(100)

class RBFN:
    def __init__(self, eta):
        self.eta = eta
        self.weights = []
        self.vec_mu = []
        self.vec_sigmas = []
        self.mat_K = []
        self.nodes = 0

    def train(self, X, T, nodes, vec_mu, vec_sigma, learning_rule, batch, epochs):
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
        @epochs - integer: How many iterations

        OUTPUT:
        @self.weights - numpy array: The calculated weights for each
                                     basis function
        """
        self.vec_errors = np.zeros((epochs,1))
        self.vec_sigmas = vec_sigma
        self.vec_mu = vec_mu
        self.weights = np.array(vec_mu).reshape(nodes, 1)
        self.nodes = nodes
        for epoch in range(epochs):
            K = self._kernel(X)
#             K,T = shuffle(K,T)
            print(self.weights)
            if learning_rule == 'least_squares':
                if batch:
                    self._least_squares_batch(K, T)
                else:
                    pass
            if learning_rule == 'delta':
                if batch:
                    pass
                else:
                    self._delta_sequential(K, T)
                    Y = self.predict(X)
                    self.vec_errors[epoch] = self.error(Y,T)


    def _delta_sequential(self, K, T):
        for ki, ti in zip(K, T):
            ki = ki.reshape(-1, len(ki))
            self.weights += self.eta * (ti - np.dot(ki, self.weights)) * ki.T

    def _least_squares_batch(self, K, T):
        self.weights = np.dot(np.linalg.pinv(K), T)

    def _kernel(self, X):
        K = np.zeros((len(X), self.nodes)).reshape(len(X), self.nodes)
        for node in range(self.nodes):
            K[:, node] = (np.exp(-((X - self.weights[node])**2)/(2*self.vec_sigmas[node]**2))).reshape(len(X), )
        for node in range(self.nodes):
            K[:, node] = (np.exp(-((X - self.weights[node])**2)/(2*self.vec_sigmas[node]**2))).reshape(len(X), )
        return K

    def predict(self, data):
        K = self._kernel(data)
        return K.dot(self.weights)
        
    def error(self, Y, T):
        return sum(np.absolute(Y - T)) / len(Y)

    def __str__(self):
        return "Radial Basis Function network"

