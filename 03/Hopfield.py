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

        for i in range(8):
            if(activation[i] > 0):
                activation[i] = 1
            elif(activation[i] < 0):
                activation[i] = -1
            else:
                activation[i] = 0
        return activation

    def recall(self, sample, nrOfUpdates=1):
        for i in range(nrOfUpdates):
            sample = self.update(sample)
        return sample


x1=[-1,-1,1,-1,1,-1,-1,1]
x2=[-1,-1,-1,-1,-1,1,-1,-1]
x3=[-1,1,1,-1,-1,1,-1,1]

samples = []
samples.append(x1)
samples.append(x2)
samples.append(x3)
nrNodes = 8
hop = Hopfield(nrNodes, samples)
hop.train()
count = 0

# create all possible patterns
combinations = []
i = [0,0,0,0,0,0,0,0]
for i[0] in range(2):
    for i[1] in range(2):
        for i[2] in range(2):
            for i[3] in range(2):
                for i[4] in range(2):
                    for i[5] in range(2):
                        for i[6] in range(2):
                            for i[7] in range(2):
                                x = [i[0],i[1],i[2],i[3],i[4], i[5], i[6], i[7]]
                                combinations.append(x)

# set zeros to -1
pattern = []
for i in range(256):
    x = []
    for j in range(8):
        if (combinations[i][j] == 0):
            xj = -1
        else:
            xj = 1
        x.append(xj)
    pattern.append(x)

# recall all
recall = []
for i in range(256):
    nrOfUpdates = 5
    x = hop.recall(pattern[i], nrOfUpdates)
    recall.append(x)
print(x)
print((x>x) - (x<x))
