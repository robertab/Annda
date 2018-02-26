import numpy as np
import matplotlib.pyplot as plt
class HRNN:
    def __init__(self, X):
        self.X = X
        self.weights = X.T.dot(X)
        self.nneurons = len(X)
        self.activations = None
        
    def plotter(self):   
        """
        Draw the pixels
        """
        output = np.copy(self.activations[0])
        new_grid = output.reshape(32, 32)
        plt.imshow(new_grid, extent=(0, 32, 0, 32),
                   interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
        plt.show()


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
#                         if (i % 100 == 0):
#                             self.plotter()
                        if np.sum(self.weights[j,:] * self.activations[i])>=0:
                            self.activations[i,j] = 1
                        else:
                            self.activations[i,j] = -1
            print(self.activations.shape)
            return self.activations

