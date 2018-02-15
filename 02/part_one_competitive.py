from RBFN import RBFN
from math import ceil, floor
import matplotlib.pyplot as plt
import numpy as np

X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1, 1)
# X_train = X_train + np.random.normal(0, 0.1, len(X_train)).reshape(-1, 1)
sin_T_train = np.sin(2*X_train).reshape(-1, 1)
sin_T_train = sin_T_train + \
              np.random.normal(0, 0.1, len(sin_T_train)).reshape(-1, 1)

# Test dat
X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1, 1)
# X_test = X_test + np.random.normal(0, 0.1, len(X_test)).reshape(-1, 1)
sin_T_test = np.sin(2*X_test).reshape(-1, 1)
sin_T_test = sin_T_test + \
             np.random.normal(0, 0.1, len(sin_T_test)).reshape(-1, 1)

# Parameter settings for MLP
input_dim = X_train.shape[1]
nodes = 11

# Parameter settings for RBFN
learning_rule = 'delta'
batch = True
epochs = 1500
eta = 0.01
strategy = 'competitive'
vec_sigma = [0.5] * nodes
normalize = False

R = RBFN()
R.train(X_train, sin_T_train, nodes, vec_sigma,
        learning_rule, batch,
        epochs, eta, strategy, normalize)


