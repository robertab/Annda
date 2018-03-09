import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

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

    rbm = BernoulliRBM(random_state=0, verbose=True)

    EPOCHS = 30
    errors = np.empty((1, EPOCHS))
    for epoch in range(EPOCHS):
        rbm.learning_rate = 0.01
        rbm.n_iter = epoch
        rbm.n_components = 150
        X_new = rbm.fit_transform(X_train, y_train)
        X_reconstructed = X_new.dot(rbm.components_)
        # Predict using a binary threshold. TODO: Logistic threshold
        X_reconstructed = np.where(X_reconstructed >= 0, 1, 0)
        errors[:, epoch] = np.sum(np.abs(X_reconstructed - X_train))
    # f, axarr = plt.subplots(2, 10)
    # classes = [11, 2, 8, 15, 7, 3, 0, 4, 31, 9]
    # for j, cls in enumerate(classes):
    #     original_grid = np.copy(X_train[cls].reshape(28,28))
    #     reconstructed_grid = np.copy(X_reconstructed[cls].reshape(28,28))
    #     axarr[0, j].imshow(original_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto") 
    #     axarr[1, j].imshow(reconstructed_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    plt.plot(errors[0, :])
    plt.show()
    # print("\n\nMean error over {} pictures (false classified pixels): {}".format(NDATA, np.mean(errors)))

    # # print()
    # # print("Logistic regression using RBM features:\n%s\n" % (
    # #         metrics.classification_report(
    # #                     y_test.reshape(NDATA,),
    # #                             classifier.predict(X_test))))
    # # # X_original = np.copy(X_train[10].reshape(28, 28))
    # original_grid = np.copy(X_train[11].reshape(28,28))
    # reconstructed_grid = np.copy(X_reconstructed[11].reshape(28,28))
    # plt.imshow(original_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    # plt.show()
    # plt.imshow(reconstructed_grid, extent=(0, 32, 0, 32), interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    # plt.show()




if __name__=='__main__':
    main()
