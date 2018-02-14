import numpy as np

class SOM:
    def __init__(self, nrOutputNodes, nrFetures, data, nrSamples):
        self.weights = np.random.random((nrOutputNodes,nrFetures))
        self.data = data
        self.nrOutputNodes = nrOutputNodes
        self.nrSamples = nrSamples

    def findMostSimularNode(self, sample):
        distances = np.zeros(self.nrOutputNodes)
        # Measure simularity
        for node in range(0,self.nrOutputNodes):
            differences = np.subtract(sample, np.copy(self.weights[node]))
            # calculate lengt of the difference (skipp sqr as it is relative)
            distances[node] = np.dot(differences.T, differences)
        # index of winnder node (node closest to the input)
        winnerNode = distances.argmin()
        return winnerNode

    def updateWeights(self, sample, winnerNode, neighbourhood):
        start = winnerNode - neighbourhood
        if start < 0:
            start = 0
        end = winnerNode + neighbourhood
        if end > 100:
            end = 100
        #print(str(start) + " " + str(end))
        for j in range(start, end):
            deltaW = np.subtract(sample, self.weights[j])
            self.weights[j] = self.weights[j] + (0.2) * deltaW

    def updateWeightsCircular(self, sample, winnerNode, neighbourhood):
        start = winnerNode - neighbourhood
        end = winnerNode + neighbourhood
        #print("     " +str(start) + " " + str(end))
        for j in range(start, end+1):
            #print(self.data[j%self.nrOutputNodes])
            # self.data[j%self.nrOutputNodes]
            deltaW = np.subtract(sample, self.weights[j%self.nrOutputNodes])
            self.weights[j%self.nrOutputNodes] = self.weights[j%self.nrOutputNodes] + (0.2) * deltaW

    def run(self, neighbourhood, circular=False, nrWinners=32):
        pos = np.zeros(nrWinners)  # winners
        count = 0
        for sample in self.data[:]:
            #print("run: sample " + str(sample))
            winnerNode = self.findMostSimularNode(sample)
            #print("winnerNode: " + str(winnerNode))
            pos[count] = winnerNode
            count = count + 1
            if circular:
                #print("weights: ")
                #print(self.weights)
                self.updateWeightsCircular(sample, winnerNode, neighbourhood)

                #print(self.weights)
            else:
                self.updateWeights(sample, winnerNode, neighbourhood)
            #print(" ")

        return pos
