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

def part21():

    animals_props = np.zeros((32,84), int)
    path = 'Animal_Data/animals.dat'
    animals_props = readAnimalInput(animals_props, path)


    # SOM starts here

    # Init Weight matrix
    # Returns random floats in the half-open interval [0.0, 1.0).
    W = np.random.random((100,84))
    #print(W)

    # distance to each output node
    distance = np.zeros(100)


    #batch
    pos = np.zeros(32)
    for i in range(0,21):
        count = 0
        # for every animal
        for animal in animals_props[:]:

            # Measure simularity
            for node in range(0,100):
                # calculate difference vectors and their lengts
                differences = np.subtract(animal, np.copy(W[node]))
                # calculate lengt of the difference (skipp sqr as it is relative)
                distance[node] = np.dot(differences.T, differences)

            # index of winnder node
            #print(distance)
            winnerNode = distance.argmin()
            #if i == 20:
            pos[count] = winnerNode
            count = count + 1
            #print(winnerNode)
            # update weight
            neighbourhood = (50 - (i *2.5))
            neighbourhood = round(neighbourhood/2) # half right and half left in array
            #print(neighbourhood)
            start = winnerNode - neighbourhood
            if start < 0:
                start = 0
            end = winnerNode + neighbourhood
            if end > 100:
                end = 100
            #print(str(start) + " " + str(end))
            for j in range(start, end):
                deltaW = np.subtract(animal, W[j])
                W[j] = W[j] + (0.2) * deltaW

    with open('Animal_Data/animalnames.txt', 'r') as f:
        animals = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    animals = [x.strip() for x in animals]
    # copy list
    animals_sorted = animals[:]
    #print(animals_sorted)
    print("")
    #print(animals)
    #print(np.sort(pos))
    #print(pos)
    sorted_indices = np.argsort(pos)
    for i in range(len(animals)):
        animals_sorted[i] = animals[sorted_indices[i]]
    print(" ")
    print(animals_sorted)
    #print(np.sort(pos))
part21()
