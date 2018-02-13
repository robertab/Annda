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

    def run(self, neighbourhood):
        pos = np.zeros(32)  # winners
        count = 0
        for sample in self.data[:]:
            winnerNode = self.findMostSimularNode(sample)
            pos[count] = winnerNode
            count = count + 1
            self.updateWeights(sample, winnerNode, neighbourhood)

        return pos

def readAnimalInput(array, path):
    arr = np.copy(array)
    with open('Animal_Data/animals.dat','r') as f:
        animal_file = f.read()
        animal = 0
        attribute = 0

        for token in animal_file:
            if token != ',' and token != '\n':

                #print(token)
                arr[animal,attribute] = int(token)
                #print(arr)
                attribute = attribute +1
                # Done traversing all attributes for one animal
                if 0 == attribute % 84:
                    animal = animal +1
                    attribute = 0
    return arr

def topologyOrderingAnimils():
    outputNodes = 100
    fetures = 84
    animals = 32

    animals_props = np.zeros((animals,fetures), int)
    path = 'Animal_Data/animals.dat'
    animals_props = readAnimalInput(animals_props, path)

    selfOrgMap = SOM(outputNodes, fetures, animals_props, animals)
    #organizedMap = selfOrgMap.run()

    for epoch in range(0,21):
        neighbourhood = (50 - (epoch *2.5))
        neighbourhood = round(neighbourhood/2) # half right and half left in array
        organizedMap = selfOrgMap.run(neighbourhood)

    # show result
    with open('Animal_Data/animalnames.txt', 'r') as f:
        animals = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    animals = [x.strip() for x in animals]

    # copy list
    animals_sorted = animals[:]

    sorted_indices = np.argsort(organizedMap)
    for i in range(len(animals)):
        animals_sorted[i] = animals[sorted_indices[i]]
    print(animals_sorted)

topologyOrderingAnimils()
