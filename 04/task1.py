import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM

DIR = 'data'
BINDIGIT_TRN = 'bindigit_trn.csv'
TARGETDIGIT_TRN = 'targetdigit_trn.csv'

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


    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm.learning_rate = 0.06
    rbm.n_iter = 10
    rbm.n_components = 50

    X_test = rbm.fit_transform(X_train, y_train)
    # test_grid = np.copy(X_reconstructed.reshape(32, 32))
    train_grid = np.copy(X_test[0].reshape(28, 28))
    # 3plt.imshow(test_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    plt.imshow(train_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    plt.show()




if __name__=='__main__':
    main()
