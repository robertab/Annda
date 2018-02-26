import numpy as np
from Hopfield import *

# memory patterns
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

print("weights")
print(hop.learningRule.weights)

print("")
print("recall")
print(hop.recall(x1))
print(np.array(x1))
print("")
print(hop.recall(x2))
print(np.array(x2))
print("")
print(hop.recall(x3))
print(np.array(x3))
