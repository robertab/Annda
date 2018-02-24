import numpy as np
from Hebbian import *

class Hopfield:
    def __init__(self, nrNodes, samples):
        self.learningRule = Hebbian(nrNodes, samples)

    def train(self):
        self.learningRule.train()

    def update(self, sample):
        pattern = np.array(sample)
        activation = np.dot(pattern, self.learningRule.weights)
        activation = np.where(activation >= 0, 1, -1)
        return activation

    def recall(self, sample):
        return self.update(sample)


x1=[-1,-1,1,-1,1,-1,-1,1]
x2=[-1,-1,-1,-1,-1,1,-1,-1]
x3=[-1,1,1,-1,-1,1,-1,1]
samples = []
samples.append(x1)
samples.append(x2)
samples.append(x3)
hop = Hopfield(8, samples)

hop.train()
print("recall")
print(hop.recall(x1))
print(np.array(x1))
print(hop.recall(x2))
print(np.array(x2))
print(hop.recall(x3))
print(np.array(x3))
print("")
print("weights")
print("")
print(hop.learningRule.weights)
