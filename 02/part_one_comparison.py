from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras import optimizers

X_train = np.arange(0, 2*np.pi, 0.1).reshape(-1, 1)
X_train = X_train + np.random.normal(0, 0.1, len(X_train)).reshape(-1,1)
sin_T_train = np.sin(2*X_train).reshape(-1, 1)
sin_T_train = sin_T_train + np.random.normal(0, 0.1, len(sin_T_train)).reshape(-1,1)

  # Test dat
X_test = np.arange(0.05, 2*np.pi, 0.1).reshape(-1, 1)
X_test = X_test + np.random.normal(0, 0.1, len(X_test)).reshape(-1,1)
sin_T_test = np.sin(2*X_test).reshape(-1, 1)
sin_T_test = sin_T_test  + np.random.normal(0, 0.1, len(sin_T_test)).reshape(-1,1)
 

model = Sequential()
model.add(Dense(neurons, 
                    input_dim = input_dim,
                    activation='sigmoid',
                    kernel_regularizer=regularizers.l2(reg),
                    bias_initializer='ones'))
model.add(Dense(1))
optimizer = optimizers.Adamax(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0, decay=0.0)
model.compile(loss="mean_squared_error", optimizer = "adam")
callbacks = [ EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)]

model.fit(train_X, train_Y, epochs=epochs,
            validation_data=(val_X,val_Y),
            callbacks=callbacks, verbose = 1)

mse = model.evaluate(val_X,val_Y,verbose=0)
