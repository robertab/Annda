import numpy as np
import matplotlib.pyplot as plt
import math
import random
from HRNN import HRNN

FILENAME = "pict.dat"

def read_data(filename):
    data = np.zeros((9, 1024))
    patterns = data.shape[0]
    with open(filename, 'r') as infile:
        for row in infile:
            tmp_data = row.split(',')
        for pattern in range(patterns):
            data[pattern, :] = tmp_data[pattern*1024:(pattern+1)*1024]
    return data




def add_noise(patterns, amt_of_noise):
    amt_of_noise = int(amt_of_noise * patterns.shape[1])
    index = np.random.choice(patterns.shape[1], amt_of_noise, replace=False)
    patterns[:, index] = patterns[:, index]*(-1)
    return np.copy(patterns)


def main():
    print("This is task 3.4 - Distortion Resistance")
    data = read_data(FILENAME)
    selected_patterns = data[0:3, :]
    # print()
    noise = [.0001, .1, .2, .3, .4, .5, .6, .7, .80, .9, 1]
    Network = HRNN(selected_patterns)
    epochs = 1 # int(math.log2(len(selected_patterns)))
    f, axarr = plt.subplots(2, 11)
    for i, amt_of_noise in enumerate(noise):
        new_selected_patterns = add_noise(np.copy(selected_patterns), amt_of_noise)
        output = np.copy(new_selected_patterns[2])
        new_grid = output.reshape(32, 32)
        axarr[0, i].imshow(new_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")

        Network.recall(np.copy(new_selected_patterns[0:3, :]), epochs, True)

        output = np.copy(Network.activations[2])
        new_grid = output.reshape(32, 32)
        axarr[1, i].imshow(new_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    plt.show()

        # Network.plotter(Network.activations[0])



if __name__ == '__main__':
    main()
