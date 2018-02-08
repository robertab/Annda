import numpy as np


class RBF:
    def __init__(self):
        pass

    def train(self, X, T, nodes, vec_mu, vec_sigma, learning_rule):
        K = np.zeros((len(X), nodes)).reshape(len(X), nodes)
        for node in range(nodes):
            K[:, node] = (np.exp(-((X - vec_mu[node])**2)/(2*vec_sigma[node]**2))).reshape(len(X), )
        print(K)

        if learning_rule == 'least_squares':
            weights = self._least_squares(K, T)
            return K.dot(weights)

    def _least_squares(self, K, T):
        return np.dot(np.linalg.pinv(K), T)

    def __str__(self):
        return "Radial Basis Function network"

