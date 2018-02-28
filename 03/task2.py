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


def main():
    X = read_data(FILENAME)
    Network = HRNN(X[0:3, :])


    """
    3.2 Sequential update

     - Check that the first three patterns are stable:
         p1, p2, p3:

         output = Network.recall(X[0:3, :], 1, synch=True)
         print(np.sum(np.abs(output - X[0:3, :])))

         DONE
     - Can the network complete a degraded pattern?:
         * Our p10 consists of the first 64 dimensions multiplied by -1 and
           then appended to the original p1 pattern

         * Our p11 simply consists of half the p2 and p3 concatenated together
         DONE
    """


    """
    start: Can the network complete a degraded pattern?:
    """
    # p10_slice = (X[0, :64]*(-1)).reshape(1, -1)
    # p10 = np.append(p10_slice, X[0, 64:]).reshape(1, -1)
    # p11 = np.concatenate((X[1, :512], X[2, 512:]), axis=0).reshape(1, -1)
    #
    # Network.plotter(p10)
    # activation = Network.recall(p10, 1, synch=True)
    # Network.plotter(activation)
    # Network.plotter(p11)
    # activation = Network.recall(p11, 1000, synch=True)
    # Network.plotter(activation)
    # print(np.sum(np.abs(activation - X[1])))
    # print(np.sum(np.abs(activation - X[2])))

    """
    end: Can the network complete a degraded pattern?:
    """



    """
    start: What happens if we select units randomly?
    """
    p10_slice = (X[0, :64]*(-1)).reshape(1, -1)
    p10 = np.append(p10_slice, X[0, 64:]).reshape(1, -1)
    p11 = np.concatenate((X[1, :512], X[2, 512:]), axis=0).reshape(1, -1)

    # Network.plotter(p10)
    # for i in range(0,10):
    #     activation = Network.recall(p10, 300, synch=False)
    #     Network.plotter(activation)

    Network.plotter(X[2])
    Network.plotter(p11)
    for i in range(0,6):
        activation = Network.recall(p11, 2, synch=False)
        Network.plotter(activation)

    print(np.sum(np.abs(activation - X[1])))
    print(np.sum(np.abs(activation - X[2])))

    """
    end: What happens if we select units randomly?
    """

if __name__ == '__main__':
    main()
