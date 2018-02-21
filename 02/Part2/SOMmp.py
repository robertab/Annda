import numpy as np

class SOMmp:
    def __init__(self, nrNodesInDim, nrFetures, data, nrSamples):
        self.weights = np.random.random((nrNodesInDim,nrNodesInDim,nrFetures))
        #print(self.weights[4])
        self.data = data
        self.nrNodesInDim = nrNodesInDim
        self.nrSamples = nrSamples

    def findWinner(self, sample):
        distances = np.zeros(self.nrNodesInDim*self.nrNodesInDim)
        # Measure simularity
        dis = 10000000

        # traverse all nodes in output grid
        for i in range(0,self.nrNodesInDim):
            for j in range(0, self.nrNodesInDim):
                difference = np.subtract(sample, np.copy(self.weights[i][j]))
                node = i * 10 + j
                distance = np.dot(difference.T, difference)
                distances[node] = distance
                if dis > distance:
                    dis=distance
                    winI = i
                    winJ = j

        # index of winnder node (node closest to the input)
        #winnerNode = distances.argmin()
        # print("distances")
        # print(distances)
        # print(" ")
        # # print("winnerNode")
        # # print(winnerNode)
        # print("distance")
        # print(dis)
        # print(winI)
        # print(winJ)
        # difference = np.subtract(sample, np.copy(self.weights[winI][winJ]))
        # distance = np.dot(difference.T, difference)
        # print(distance)
        # print(" ")
        # print(" ")


        return winI, winJ

    def updateWeights(self, sample, i, j, neighbourhood):
        # i = int(winnerNode/10)  # get onties
        # j = winnerNode - (i*10) # get singulars
        #

        iStart = i - neighbourhood
        jStart = j - neighbourhood
        if iStart < 0:
            iStart = 0
        if jStart <0:
            jStart = 0

        iEnd = i + neighbourhood
        jEnd = j + neighbourhood
        if iEnd > 9:
            iEnd = 9
        if jEnd > 9:
            jEnd = 9
        # print("update")
        # print(winnerNode)
        # print(str(i) + str(j))
        # print(" ")
        # print("updateWeights")
        # print("i")
        # print(str(iStart) + " " + str(iEnd))
        # print("j")
        # print(str(jStart) + " " + str(jEnd))
        for i in range(iStart, iEnd+1):
            for j in range(jStart, jEnd+1):
                # print("in loop")
                # print(str(i) + str(j))
                # print('shape')
                # print(sample.shape)
                # print(self.weights[i][j].shape)
                # print(self.weights[j].shape)
                deltaW = np.subtract(sample, np.copy(self.weights[i][j]))
                self.weights[i][j] = self.weights[i][j] + (0.2) * deltaW


    def run(self, neighbourhood, circular=False, nrWinners=349):
        pos = np.zeros(self.nrSamples)  # winners
        count = 0
        for sample in self.data[:]:
            i, j = self.findWinner(sample)
            pos[count] = i * 10 + j
            count = count + 1
            self.updateWeights(sample, i, j ,neighbourhood)

        return pos
