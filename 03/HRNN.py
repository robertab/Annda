import numpy as np

class HRNN:
    def __init__(self, X):
        self.X = X
        self.weights = X.T.dot(X)
        self.nneurons = len(X)
        self.activations = None

    def recall(self, patterns, epochs, synch=False):
        self.activations = patterns
        if synch:
            for epoch in range(epochs):
                for i,activation in enumerate(self.activations):
                    self.activations[i,:] = np.where(activation.dot(self.weights) >= 0, 1, -1)
            print(self.activations.shape)
            return self.activations
        
        else:
            order = np.arange(len(patterns[0])) 
            print(order)
            for epoch in range(epochs):
                for i in range(len(self.activations)):
#                     if self.random:
#                     np.random.shuffle(order) 
                    for j in order:
                        if np.sum(self.weights[j,:] * self.activations[i])>=0:
                            self.activations[i,j] = 1
                        else:
                            self.activations[i,j] = -1
            print(self.activations.shape)
            return self.activations

