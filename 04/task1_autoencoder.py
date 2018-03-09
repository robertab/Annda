import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Dense
from keras.models import Model

DIR = 'data'
BINDIGIT_TRN = 'bindigit_trn.csv'
BINDIGIT_TST = 'bindigit_tst.csv'

TARGETDIGIT_TRN = 'targetdigit_trn.csv'
TARGETDIGIT_TST = 'targetdigit_tst.csv'

TRN_DIM = 28*28
TRGT_DIM = 1
NDATA = 8000


def file_to_data(filename, dim, ndata):
    data = np.empty((ndata, dim))
    with open(DIR + '/' + filename, "r") as infile:
        for i, line in enumerate(infile):
            data[i, :] = line.split(',')[:dim]
    return data


def main():
    X_train = file_to_data(BINDIGIT_TRN, TRN_DIM, NDATA)
    y_train = file_to_data(TARGETDIGIT_TRN, TRGT_DIM, NDATA)

    X_test = file_to_data(BINDIGIT_TST, TRN_DIM, NDATA)
    y_test = file_to_data(TARGETDIGIT_TST, TRGT_DIM, NDATA)

    EPOCHS = 30
    encoding_dim = 100
    input_img = Input(shape=(TRN_DIM,))
    # AUTOENCODER
    encoded = Dense(encoding_dim,activation='relu',)(input_img)
    decoded = Dense(TRN_DIM, activation='sigmoid')(encoded)
    autoencoder = Model(input_img,decoded)
    # ENCODER
    encoder = Model(input_img,encoded)
    encoded_input = Input(shape=(encoding_dim,))
    # DECODER LAYER
    decoder_layer = autoencoder.layers[-1]
    # DECODER MODEL
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    #COMPILE
    autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
    #FIT DATA
    autoencoder.fit( X_train, X_train,
                    epochs=EPOCHS,
                    batch_size=255,
                    shuffle=True)
    
    # PREDICT
    encoded_imgs = encoder.predict(X_train)
    decoded_imgs = decoder.predict(encoded_imgs)
    print(decoded_imgs)
    print(decoded_imgs.shape)
    f, axarr = plt.subplots(2, 10)
    classes = [11, 2, 8, 15, 7, 3, 0, 4, 31, 9]
    for j, cls in enumerate(classes):
        axarr[0, j].imshow(X_train[cls].reshape(28,28), extent=(0,32,0,32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
        plt.gray()
        axarr[1, j].imshow(decoded_imgs[cls].reshape(28,28), extent=(0,32,0,32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
        plt.gray()
    plt.show()
    #print("\n\nMean error over {} pictures (false classified pixels): {}".format(NDATA, np.mean(errors)))



if __name__=='__main__':
    main()
