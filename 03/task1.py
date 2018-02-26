import numpy as np
from Hopfield import *

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
