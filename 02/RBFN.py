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

    def train(self, X, T, nodes, vec_sigma, learning_rule, batch, epochs, eta, strategy=None, normalize = False):
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
        self.vec_errors = np.zeros((epochs,2))
        self.vec_sigmas = vec_sigma
        self.vec_mu = self.init_weights(X,strategy)
        self.weights = np.array(self.vec_mu).reshape(nodes, 2)
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
            for point in range(len(X)):
                K[point,node] = np.exp((-(np.linalg.norm(X[point, :]-self.vec_mu[node, :]))**2) \
                        / (2*(self.vec_sigmas[0]**2)))
        return K
        # if normalize:
        #     normalizers = np.sqrt(np.sum(K**2,axis=1))
        #     K = np.transpose(np.transpose(K)/normalizers)
        # return K

    def predict(self, data):
        K = self._kernel(data)
        return K.dot(self.weights)


    def init_weights(self, X, strategy):
        centers = np.zeros((self.nodes, X.shape[1]))
        if strategy == None:
            centers = [0, 0.9, 2.25, 3.9, 5.5, 6.2]
            plt.plot(centers, 'r+', label='centers')
            plt.legend()
            plt.show()
        if strategy == "random_init":
            indices = list(range(len(X)))
            np.random.shuffle(indices)
            for i in range(self.nodes):
                centers.append(X[indices[i]])
        
        elif strategy == "k_means":
            kmeans = KMeans(n_clusters=self.nodes).fit(X)
            centers = kmeans.cluster_centers_
            
        elif strategy == "competitive":
            # random init weights
            # centers = np.random.normal(0,1,self.nodes)
            centers[:, 0] = np.random.normal(0,1,self.nodes)
            centers[:, 1] = np.random.normal(0,1,self.nodes)
            #normalize weights
            centers = np.transpose(np.transpose(centers) / np.linalg.norm(centers))
            old_centers = np.copy(centers)
            alpha = 0.2
            dist = []
            for i in range(100):
                for j in range(len(X)):
                    vec_x = X[np.random.randint(0,len(X)),:]
                    for c in centers:
                        dist.append(np.sqrt((vec_x[0]-c)**2))
                    dc = dist
                    min_index = np.argmin(dist, axis=0)
                    centers[min_index] += alpha * (vec_x[0] - centers[min_index])
                    # leaky learning - multiple winners
                    multiple_winners = 0
                    for n in range(multiple_winners):
                        dc[min_index[0]][0] = 9999999
                        dc[min_index[1]][1] = 9999999
#                         dc.pop(min_index)
                        min_2_index = np.argmin(dc, axis=0)
                        centers[min_2_index] += alpha * (vec_x - centers[min_2_index])
                        min_index = min_2_index
#                     centers = self.update_centers(min_index,centers,x)
#                     centers = np.transpose(np.transpose(centers) / np.linalg.norm(centers))
                    dist = []
            #plot the weight change
#             plt.plot(old_centers[:,0], 'b+', label='start_centers_col0')
#             plt.plot(centers[:,0], 'r+', label='new_centers_col0')
#             plt.legend()
#             plt.show()
#             plt.plot(old_centers[:,1], 'g+', label='start_centers_col1')
#             plt.plot(centers[:,1], 'y+', label='new_centers_col1')
#             plt.legend()
#             plt.show()
        print(centers)

        return centers 
        
    def error(self, Y, T):
        return sum(np.absolute(Y - T)) / len(Y)

    def __str__(self):
        return "Radial Basis Function network"

