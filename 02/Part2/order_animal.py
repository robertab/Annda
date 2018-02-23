from SOM import *
import numpy as np


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

def getResult(organizedMap):
    # show result
    with open('Animal_Data/animalnames.txt', 'r') as f:
        animals = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    animals = [x.strip() for x in animals]

    # copy list
    animals_sorted = animals[:]
#     print(organizedMap)

    sorted_indices = np.argsort(organizedMap)

#     print(sorted_indices)
    for i in range(len(animals)):
        animals_sorted[i] = animals[sorted_indices[i]]
    return animals_sorted

def topologyOrderingAnimals():
    outputNodes = 100
    fetures = 84
    animals = 32

    animals_props = np.zeros((animals,fetures), int)
    path = 'Animal_Data/animals.dat'
    animals_props = readAnimalInput(animals_props, path)
    # #bat
    # print(animals_props[2])
    # #elephant
    # print(animals_props[12])
    # #rabbit
    # print(animals_props[26])

    selfOrgMap = SOM(outputNodes, fetures, animals_props, animals)

    ephocs = 21
    startNeigh = 50
    for epoch in range(0,ephocs):
        neighbourhood = (startNeigh - (epoch * startNeigh/ (ephocs-1)))
        #print(neighbourhood)
        neighbourhood = round(neighbourhood/2) # half right and half left in array
        organizedMap = selfOrgMap.run(neighbourhood)

    animals_sorted = getResult(organizedMap)
    print(animals_sorted)

topologyOrderingAnimals()
