import numpy as np
import matplotlib.pyplot as plt
import math
import random
from HRNN2 import HRNN2
np.random.seed(10000)
np.set_printoptions(threshold=np.nan)

def missmatch(pattern,activation):
    diff = np.abs(pattern-activation)
    return np.sum(np.where(np.sum(diff,axis = 1)>0,0,1)) 

def add_noise(pattern, amt_of_noise):
    # Calculate the percentages
    amt_of_noise = int(amt_of_noise * 100)
    # Pick out a row vector of indices
    index = np.random.choice(100, amt_of_noise, replace=False)
    # Add some noise to the patterns by flipping the randomly selected units
    pattern[index] = pattern[index] + 1
    return np.copy(pattern)

def main():
    random_data = np.random.randint(1,size=(100,100))
    for i, pattern in enumerate(random_data):
        random_data[i, :] = add_noise(pattern, 0.10)
    correct_classed_patterns = [None] * random_data.shape[0]
    biases = [0, .001, .01, .1, 1, 10]
    for bias in biases:
        for p in range(2, 30):
            selected_patterns = random_data[0:p, :]
            print(selected_patterns)
            Network = HRNN2(selected_patterns)
            output = Network.recall(selected_patterns, 1, True, bias)
            correct_classed_patterns[p-2] = missmatch(Network.activations, np.copy(selected_patterns)) 
            print(correct_classed_patterns[p-2])
        plt.plot(correct_classed_patterns)
        plt.show()






    # random_data = np.where((np.random.rand(300,100) * 10) - 2.5 >=0,1,-1)
    # print(random_data)
    # random_data = np.where(npv.random.rand(2(300,100))==0,-1,1)
    # print(random_data)
#     for p in range(2,random_data.shape[0]):
#         selected_patterns = random_data[0:p, :]
#         # Initialize the Hopfield RNN with three selected patterns from the pict dataset
#         Network = HRNN2(selected_patterns)
#         new_selected_patterns = add_noise(np.copy(selected_patterns), 0.0)
#         epochs = 1 #int(math.log2(len(selected_patterns)))
#         # Initialize a subplot object
#         Network.recall(np.copy(new_selected_patterns[0:p, :]), epochs, True)
#         correct_classed_patterns[p-2] = missmatch(Network.activations, np.copy(selected_patterns)) 
# 
main()
