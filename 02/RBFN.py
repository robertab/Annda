import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
np.random.seed(100)

class RBFN:
    def __init__(self):
        self.weights = []
        self.vec_mu = []
        self.vec_sigmas = []
        self.mat_K = []
        self.nodes = 0

    def train(self, X, T, nodes, vec_sigma, learning_rule, batch, epochs, eta, strategy, normalize = False):
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
        self.eta = eta
        self.nodes = nodes
        self.vec_errors = np.zeros((epochs,1))
        self.vec_sigmas = vec_sigma
        self.vec_mu = self.init_weights(X,strategy)
        self.weights = np.array(self.vec_mu).reshape(nodes, 1)
        K = self._kernel(X, normalize)
        for epoch in range(epochs):
            K,T = shuffle(K,T)
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

    def _kernel(self, X, normalize=False):
        K = np.zeros((len(X), self.nodes)).reshape(len(X), self.nodes)
        for node in range(self.nodes):
            K[:, node] = (np.exp(-((X - self.vec_mu[node])**2) /
                             (2*self.vec_sigmas[node]**2))).reshape(len(X), )
        if normalize:
            normalizers = np.sqrt(np.sum(K**2,axis=1))
            print(normalizers.shape)
            print(K.shape)
            K = np.transpose(np.transpose(K)/normalizers)
            print(K.shape)
        return K

    def predict(self, data):
        K = self._kernel(data)
        return K.dot(self.weights)

#     def update_centers(self, min_index, centers, x):
#         # update the winner
#         centers[min_index] += alpha * (x - centers[min_index])
#         # how many units will get updated
#         neighbourhood = 1
#         for i in range(neighbourhood)
#         return centers
# 
#     def h(self, centers):


    def init_weights(self, X, strategy):
        centers = []
        if strategy == "random_init":
            indices = list(range(len(X)))
            np.random.shuffle(indices)
            for i in range(self.nodes):
                centers.append(X[indices[i]])
        
        elif strategy == "k_means":
            kmeans = KMeans(n_clusters=self.nodes).fit(X)
#             print(kmeans.cluster_centers_)
            centers = kmeans.cluster_centers_
            
        elif strategy == "competitive":
#             indices = list(range(len(X)))
#             np.random.shuffle(indices)
#             for i in range(self.nodes):
#                 centers.append(X[indices[i]][0])

            # random init weights
            centers = np.random.normal(0,1,self.nodes)
            old_centers = np.copy(centers)
            #normalize weights
            centers = centers / np.linalg.norm(centers)
            print(centers)
            alpha = 0.2
            dist = []
            for i in range(100):
                for x in X:
                    for c in centers:
                        dist.append(np.sqrt((x-c)**2))
                    dc = dist
                    min_index = np.argmin(dist)
                    centers[min_index] += alpha * (x - centers[min_index])
                    # leaky learning
                    for n in range(1):
                        dc.pop(min_index)
                        min_2_index = np.argmin(dc)
                        centers[min_2_index] += alpha * (x - centers[min_2_index])
                        min_index = min_2_index
#                     centers = self.update_centers(min_index,centers,x)
#                     centers = centers / np.linalg.norm(centers)
                    dist = []
            plt.plot(old_centers, 'b+', label='start_centers')
            plt.plot(centers, 'r+', label='new_centers')
            plt.legend()
            plt.show()

            print(old_centers)
        print(centers) 
        return centers 
        
    def error(self, Y, T):
        return sum(np.absolute(Y - T)) / len(Y)

    def __str__(self):
        return "Radial Basis Function network"

