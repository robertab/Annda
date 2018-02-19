from RBFN import RBFN
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open('ballist.dat', 'rb') as data:
        ball_data = np.loadtxt(data, delimiter=" ", skiprows=0)
        X_train = ball_data[:, 0:2]
        Y_train = ball_data[:, 2:4]
        
    with open('balltest.dat', 'rb') as data:
        ball_test = np.loadtxt(data, delimiter=" ", skiprows=0)
        X_test = ball_data[:, 0:2]
        Y_test = ball_data[:, 2:4]


    R = RBFN()
    # Parameter settings for RBFN
    nodes = 25
    learning_rule = 'delta'
    batch = False
    epochs = 500
    eta = 0.01
    strategy = 'competitive'
    vec_sigma = [[0.5] * nodes]*2
    normalize = False
    R.train(X_train, Y_train, nodes, vec_sigma,
            learning_rule, batch,
            epochs, eta, strategy, normalize)
    Y = R.predict(X_test)

    plt.plot(Y_test[:, 1])
    plt.plot(Y[:, 1])
    # plt.plot()
    plt.show()


if __name__ == '__main__':
    main()
