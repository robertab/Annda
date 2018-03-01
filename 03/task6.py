import numpy as np
import matplotlib.pyplot as plt
import math
import random
from HRNN2 import HRNN2

def read_data(filename):
    data = np.zeros((9, 1024))
    patterns = data.shape[0]
    with open(filename, 'r') as infile:
        for row in infile:
            tmp_data = row.split(',')
        for pattern in range(patterns):
            data[pattern, :] = tmp_data[pattern*1024:(pattern+1)*1024]
    return data

def main():
    #data = read_data(FILENAME)
    # Select p1, p2, p3
    data = np.zeros((3, 8))
    data[0] = np.array([0,0,1,0,1,0,0,1])
    data[1] = np.array([0,0,0,0,0,1,0,0])
    data[2] = np.array([0,1,1,0,0,1,0,1])


    xd = np.zeros((3, 8))
    xd[0] = np.array([1,0,1,0,1,0,0,1])
    xd[1] = np.array([1,1,0,0,0,1,0,0])
    xd[2] = np.array([1,1,1,0,1,1,0,1])

    selected_patterns = data[0:3, :]

    Network = HRNN2(selected_patterns)
    patterns = xd
    epochs = 5
    synch=True
    bias = 0.5
    activations = Network.recall(patterns, epochs, synch, bias)
    print(activations)
    print("")
    print(activations - data)
if __name__ == '__main__':
    FILENAME = "pict.dat"
    main()
