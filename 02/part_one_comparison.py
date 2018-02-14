from RBFN import RBFN
from math import ceil, floor
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import optimizers

import matplotlib.pyplot as plt
import numpy as np

X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1, 1)
X_train = X_train + np.random.normal(0, 0.1, len(X_train)).reshape(-1, 1)
sin_T_train = np.sin(2*X_train).reshape(-1, 1)
sin_T_train = sin_T_train + \
              np.random.normal(0, 0.1, len(sin_T_train)).reshape(-1, 1)

# Test dat
X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1, 1)
X_test = X_test + np.random.normal(0, 0.1, len(X_test)).reshape(-1, 1)
sin_T_test = np.sin(2*X_test).reshape(-1, 1)
sin_T_test = sin_T_test + \
             np.random.normal(0, 0.1, len(sin_T_test)).reshape(-1, 1)

# Parameter settings for MLP
input_dim = X_train.shape[1]
nodes = 60

# Parameter settings for RBFN
learning_rule = 'least_squares'
batch = True
epochs = 1500
eta = 0.01
strategy = 'k_means'
vec_sigma = [0.5] * nodes

R = RBFN()
R.train(X_train, sin_T_train, nodes, vec_sigma,
        learning_rule, batch,
        epochs, eta, strategy)

model = Sequential()
model.add(Dense(floor(nodes/2), input_dim=input_dim,
                activation='sigmoid',
                bias_initializer='ones'))
model.add(Dense(ceil(nodes/2), input_dim=floor(nodes/2),
                activation='sigmoid',
                bias_initializer='ones'))
model.add(Dense(1))

# optimizer = optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999,
#                               epsilon=0, decay=0.0)
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(X_train, sin_T_train, epochs=epochs, verbose=0)
Y_mlp = model.predict(X_train)
Y_rbfn = R.predict(X_train)

x_range = np.arange(0, 2*np.pi, 0.1)
plt.plot(x_range, sin_T_train, c="b")
plt.plot(x_range, Y_mlp, c="r")
plt.plot(x_range, Y_rbfn, c="g")
plt.show()

# mse = model.evaluate(val_X,val_Y,verbose=0)
