import numpy as np
import matplotlib.pyplot as plt
import math
import random
from HRNN import HRNN
np.random.seed(10000)
np.set_printoptions(threshold=np.nan)

def missmatch(pattern,activation):
    diff = np.abs(pattern-activation)
    return np.sum(np.where(np.sum(diff,axis = 1)>0,0,1)) 

def add_noise(patterns, amt_of_noise):
    # Calculate the percentages
    amt_of_noise = int(amt_of_noise * patterns.shape[1])
    # Pick out a row vector of indices
    index = np.random.choice(patterns.shape[1], amt_of_noise, replace=False)
    # Add some noise to the patterns by flipping the randomly selected units
    patterns[:, index] = patterns[:, index]*(-1)
    return np.copy(patterns)

def main():
#     random_data = np.where(np.random.randint(2,size=(300,100))==0,-1,1)
    random_data = np.where((np.random.rand(300,100) * 10) - 2.5 >=0,1,-1)
    # random_data = np.where(npv.random.rand(2(300,100))==0,-1,1)
    print(random_data)
    correct_classed_patterns = [None] * random_data.shape[0]
    for p in range(2,random_data.shape[0]):
        selected_patterns = random_data[0:p, :]
        # Initialize the Hopfield RNN with three selected patterns from the pict dataset
        Network = HRNN(selected_patterns)
        new_selected_patterns = add_noise(np.copy(selected_patterns), 0.0)
        epochs = 1 #int(math.log2(len(selected_patterns)))
        # Initialize a subplot object
        Network.recall(np.copy(new_selected_patterns[0:p, :]), epochs, True)
        correct_classed_patterns[p-2] = missmatch(Network.activations, np.copy(selected_patterns)) 
    print(correct_classed_patterns)
    plt.title("Noise: 10%")
    plt.plot(correct_classed_patterns, label=" percent of correct patterns")
    plt.xlabel("number of stored patterns")
    plt.legend()
    plt.show()

main()
