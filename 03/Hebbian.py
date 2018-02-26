import numpy as np

# Hebbian learning rule for hopfield network
class Hebbian:
    def __init__(self, nrNodes, samples):
        self.nrSamples = len(samples)
        self.weights = np.zeros((nrNodes, nrNodes))
        self.nrNodes = nrNodes
        self.samples = np.array(samples)

    def train(self):
        # for i in range(self.nrSamples):
        #     sample = self.samples[i].reshape(self.nrNodes,1)
        #     self.weights = self.weights + sample * sample.T
        self.weights = np.dot(self.samples.T, self.samples)
            # print("weights")
            # print(self.weights)

        # set diagonal to 0, no unit has a connection with itself
        # for i in range(self.nrNodes):
        #     self.weights[i,i] = 0


# x1=[-1,-1,1,-1,1,-1,-1,1]
# x2=[-1,-1,-1,-1,-1,1,-1,-1]
# x3=[-1,1,1,-1,-1,1,-1,1]
# samples = []
# samples.append(x1)
# samples.append(x2)
# samples.append(x3)
# nrNodes = 8
# hebb = Hebbian(nrNodes, samples)
# hebb.train()
# print(hebb.weights)
