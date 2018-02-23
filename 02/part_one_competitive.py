from RBFN import RBFN
from math import ceil, floor
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np

X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1, 1)
# X_train = X_train + np.random.normal(0, 0.1, len(X_train)).reshape(-1, 1)
sin_T_train = np.sin(2*X_train).reshape(-1, 1)
# sin_T_train = sin_T_train + \
#               np.random.normal(0, 0.1, len(sin_T_train)).reshape(-1, 1)

# Test data
X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1, 1)
# X_test = X_test + np.random.normal(0, 0.1, len(X_test)).reshape(-1, 1)
sin_T_test = np.sin(2*X_test).reshape(-1, 1)
# sin_T_test = sin_T_test + \
#              np.random.normal(0, 0.1, len(sin_T_test)).reshape(-1, 1)

# Parameter settings for MLP
input_dim = X_train.shape[1]
nodes = 15

# Parameter settings for RBFN
learning_rule = 'delta'
batch = False
epochs = 500
eta = 0.01
strategy = 'competitive'
vec_sigma = [0.5] * nodes
normalize = False

R = RBFN()
R.train(X_train, sin_T_train, nodes, vec_sigma,
        learning_rule, batch,
        epochs, eta, strategy, normalize)

# X_train = np.transpose(np.transpose(X_train)/np.linalg.norm(X_train))
# print(X_train)
Y = R.predict(X_train)
print(R.error(Y,sin_T_train))
plt.plot(np.arange(0,2*np.pi, 0.1), Y)
plt.plot(np.arange(0,2*np.pi, 0.1), sin_T_train)
plt.show()

# etas = [0.001, 0.01, 0.05, 0.1]
# for eta in etas:
#         vec_sigma = [0.5] * nodes
#         R.train(X_train, sin_T_train,
#                 nodes, vec_sigma, learning_rule,
#                 batch, epochs, eta, strategy, normalize)
# 
        # fig = plt.figure()
ax = plt.subplot(111)
plt.title("error vs epoch. (sequential delta rule). Eta = "+str(eta))
ax.plot(R.vec_errors,c='y', label="error")
plt.ylim((0, 3))
plt.legend()
plt.show()



