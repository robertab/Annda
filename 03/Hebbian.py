import numpy as np

class Hebbian:
    def __init__(self, nrNodes, samples):
        self.nrSamples = len(samples)
        self.weights = np.zeros((nrNodes, nrNodes))

        # init samples to matrix
        self.samples = np.zeros((self.nrSamples, nrNodes, nrNodes))
        for i in range(self.nrSamples):
            for j in range(nrNodes):
                sample = np.array(samples[i])
                self.samples[i,j] = sample

    def train(self):
        for i in range(self.nrSamples):
            self.weights = self.weights + self.samples[i] * self.samples[i].T

class Hopfield:
    def __init__(self, nrNodes, samples):
        self.hebb = Hebbian(nrNodes, samples)

    def train(self):
        self.hebb.train()

    def recall(self, sample):
        print(len(sample))
        pattern = np.array(sample)
        activation = np.dot(pattern, self.hebb.weights)
        print(activation)
        activation = np.where(activation > 0, 1, -1)
        print(activation)
        #activation = np.zeros()


x1=[-1,-1,1,-1,1,-1,-1,1]
x2=[-1,-1,-1,-1,-1,1,-1,-1]
x3=[-1,1,1,-1,-1,1,-1,1]
samples = []
samples.append(x1)
samples.append(x2)
samples.append(x3)
print(len(samples))
hop = Hopfield(8, samples)
print(hop.hebb.weights.shape)
print(hop.hebb.samples.shape)
print(hop.hebb.samples[0])
print(hop.hebb.samples[1])
print(hop.hebb.samples[2])
hop.train()
print("recall")
hop.recall(x1)
print(x1)
print("")
print("weights")
print("")
print(hop.hebb.weights)
