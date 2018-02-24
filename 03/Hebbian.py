import numpy as np

# Hebbian learning rule for hopfield network
class Hebbian:
    def __init__(self, nrNodes, samples):
        self.nrSamples = len(samples)
        self.weights = np.zeros((nrNodes, nrNodes))

        # init samples to matrix for matrix operations
        self.samples = np.zeros((self.nrSamples, nrNodes, nrNodes))
        for i in range(self.nrSamples):
            for j in range(nrNodes):
                sample = np.array(samples[i])
                self.samples[i,j] = sample

    def train(self):
        for i in range(self.nrSamples):
            self.weights = self.weights + self.samples[i] * self.samples[i].T
