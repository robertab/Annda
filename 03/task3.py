import numpy as np
import matplotlib.pyplot as plt
from HRNN import HRNN

FILENAME = "pict.dat"


def read_data(filename):
    data = np.zeros((9, 1024))
    patterns = data.shape[0]
    with open(filename, 'r') as infile:
        for row in infile:
            tmp_data = row.split(',')
        for pattern in range(patterns):
            data[pattern, :] = tmp_data[pattern*1024:(pattern+1)*1024]
    return data

def energy(W, x, nrUnits=1024):
    # energy = 0
    # for i in range(nrUnits):
    #     for j in range(nrUnits):
    #         energy += W[i,j] * x[i] * x[j]
    return - x.dot(W.dot(x.T))


def main():
    X = read_data(FILENAME)
    Network = HRNN(X[0:3, :])


    p10_slice = (X[0, :64]*(-1)).reshape(1, -1)
    p10 = np.append(p10_slice, X[0, 64:]).reshape(1, -1)
    p11 = np.concatenate((X[1, :512], X[2, 512:]), axis=0).reshape(1, -1)

    ener =[]
    mu = 1
    sigma = 4
    randomW = np.random.normal(mu, sigma, (1024,1024))
    # set diagonal to 0, no unit has a connection with itself
    for i in range(1024):
        randomW[i,i] = 0
    Network.weights = randomW
    activation = np.zeros(1024)
    for i in range(0,100):
        activation = Network.recall(activation , 1, synch=False)
        #print(energy(Network.weights, activation[0]))
        ener.append(energy(Network.weights, activation))
        #print(i)

    print(ener)
    plt.plot(ener)
    plt.show()

    ener =[]
    Network.weights = 0.5 * (randomW * randomW.T)
    activation = np.zeros(1024)
    print("ener")
    print(ener)
    for i in range(0,100):
        activation = Network.recall(activation , 1, synch=False)
        #print(energy(Network.weights, activation[0]))
        ener.append(energy(Network.weights, activation))
        #print(i)

    print(ener)
    plt.plot(ener)
    plt.show()

    print("")
    print(energy(Network.weights, X[0]))
    print(energy(Network.weights, X[1]))
    print(energy(Network.weights, X[2]))
    print(energy(Network.weights, p10[0]))
    print(energy(Network.weights, p11[0]))



if __name__ == '__main__':
    main()
