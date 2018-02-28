import numpy as np
import matplotlib.pyplot as plt
import math
import random
from HRNN import HRNN
np.random.seed(1000)


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
    # Calculate the percentages
    amt_of_noise = int(amt_of_noise * patterns.shape[1])
    # Pick out a row vector of indices
    index = np.random.choice(patterns.shape[1], amt_of_noise, replace=False)
    # Add some noise to the patterns by flipping the randomly selected units
    patterns[:, index] = patterns[:, index]*(-1)
    return np.copy(patterns)


def main():
    print("This is task 3.4 - Distortion Resistance")
    data = read_data(FILENAME)
    # Select p1, p2, p3
    selected_patterns = data[0:3, :]
    noise = [.0001, .1, .2, .3, .4, .5, .6, .7, .80, .9, 1]
    # Initialize the Hopfield RNN with three selected patterns from the pict dataset
    Network = HRNN(selected_patterns)
    epochs = int(math.log2(len(selected_patterns)))
    # Initialize a subplot object
    f, axarr = plt.subplots(2, 11)
    for i, amt_of_noise in enumerate(noise):
        new_selected_patterns = add_noise(np.copy(selected_patterns), amt_of_noise)
        # Plot the noisy pattern 
        output = np.copy(new_selected_patterns[2])
        new_grid = output.reshape(32, 32)
        axarr[0, i].imshow(new_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
        # Recall on all the patterns
        Network.recall(np.copy(new_selected_patterns[0:3, :]), epochs, True)
        # Plot the predicted patterns
        output = np.copy(Network.activations[2])
        new_grid = output.reshape(32, 32)
        axarr[1, i].imshow(new_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    plt.show()



if __name__ == '__main__':
    FILENAME = "pict.dat"
    main()
