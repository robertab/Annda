import numpy as np
from Hopfield import *
import itertools
import matplotlib.pyplot as plt


def Q1():
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

    x1d=[1,-1,1,-1,1,-1,-1,1]
    x2d=[1,1,-1,-1,-1,1,-1,-1]
    x3d=[1,1,1,-1,1,1,-1,1]

    print(np.array(x1))
    print(hop.recall(x1d))
    print(np.array(x1) - hop.recall(x1d))
    print("")
    print("")
    print("")


    pattern = hop.recall(x2d)
    print(np.array(x2))
    print(pattern)
    print(np.array(x2) - pattern)
    print("")
    for i in range(3):
        pattern = hop.recall(pattern)
        print(np.array(x2))
        print(pattern)
        print(np.array(x2) - pattern)
        print("")

    print("")
    print("")
    print("")

    pattern = hop.recall(x3d)
    print(np.array(x3))
    print(pattern)
    print(np.array(x3) - pattern)
    print("")
    for i in range(3):
        pattern = hop.recall(pattern)
        print(np.array(x3))
        print(pattern)
        print(np.array(x3) - pattern)
        print("")

def q2():

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
    activations = []
    counter = []
    for i in range(256):
        nrOfUpdates = 5
        x = hop.recall(pattern[i], nrOfUpdates)
        add = True
        for j in range(len(activations)):
            count = 0
            re = activations[j]
            diff = re - x
            for k in range(8):
                if diff[k] == 0:
                    count += 1
            if count == 8:
                add = False
                counter[j] += 1
                break;
        if(add or len(activations) == 0):
            activations.append(x)
            counter.append(1)

    print(len(activations))

    totCount = 0

    for i in range(len(activations)):
        print(counter[i])
        totCount += counter[i]
        print(activations[i])
    print(totCount)

def q3():
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
    nrOfUpdates = 5
    x1d =[1,1,-1,1,-1,1,1,1]
    x2d =[1,1,1,1,1,-1,-1,-1]
    x3d =[1,-1,-1,1,1,1,-1,1]
    print(hop.recall(x1d, nrOfUpdates))
    print(hop.recall(x2d, nrOfUpdates))
    print(hop.recall(x3d, nrOfUpdates))

def energy(W, x):
    return W * x * x.T


def energyPlot():

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
    lst = list(map(list, itertools.product([-1, 1], repeat=8)))
    #print(lst)
    count = 0
    activations = np.array(lst)
    energy = np.zeros(256)
    W = hop.learningRule.weights
    print(energy)
    for i, x in enumerate(activations):
        x = x.reshape(8,1)

        #energy[i] = np.sum(W.dot(x))
        Wx =W.dot(x)
        energy[i] = Wx.T.dot(x).reshape(1)

    print(energy)

    for k, x in enumerate(activations):
        x = x.reshape(8,1)

        summ = 0
        for i in range(8):
            for j in range(8):
                summ += W[i,j] + x[i] + x[j]
        energy[k] = -summ

    print(energy)
    print(energy[251])
    #   32 16 8 4 2 1   41
    # 0 0 1 0 1 0 0 1
    # x1=[-1,-1,1,-1,1,-1,-1,1]
    #
    #                     4
    # x2=[-1,-1,-1,-1,-1,1,-1,-1]
    #
    #        64    32       4    1          101
    # x3=[-1,1,1,-1,-1,1,-1,1]

    plt.plot(energy)
    plt.plot(41, 'r')
    plt.show()






energyPlot()
