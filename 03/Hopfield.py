import numpy as np
from Hebbian import *

class Hopfield:
    def __init__(self, nrNodes, samples):
        self.learningRule = Hebbian(nrNodes, samples)
        self.nrNodes = nrNodes

    def train(self):
        self.learningRule.train()

    def update(self, sample):
        pattern = np.array(sample).reshape(1,self.nrNodes)
        activation = np.dot(pattern, self.learningRule.weights).reshape(self.nrNodes)
        activation = np.where(activation >= 0, 1, -1)
        return activation

    def recall(self, sample, nrOfUpdates=1):
        for i in range(nrOfUpdates):
            sample = self.update(sample)
        return sample
