import numpy as np

# Hebbian learning rule for hopfield network
class Hebbian:
    def __init__(self, nrNodes, samples):
        self.nrSamples = len(samples)
        self.weights = np.zeros((nrNodes, nrNodes))
        self.nrNodes = nrNodes
        self.samples = np.array(samples)

    def train(self):
        self.weights = np.dot(self.samples.T, self.samples)
        
        # set diagonal to 0, no unit has a connection with itself
        # for i in range(self.nrNodes):
        #     self.weights[i,i] = 0
