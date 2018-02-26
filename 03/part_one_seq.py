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
         DONE
     - Can the network complete a degraded pattern?:
         * Our p10 consists of the first 64 dimensions multiplied by -1 and
           then appended to the original p1 pattern

         * Our p11 simply consists of half the p2 and p3 concatenated together
         DONE
    """
    p10_slice = (X[0, :64]*(-1)).reshape(1, -1)
    p10 = np.append(p10_slice, X[0, 64:]).reshape(1, -1)
    p11 = np.concatenate((X[1, :512], X[2, 512:]), axis=0).reshape(1, -1)
    degraded_pattern = np.concatenate((p10, p11), axis=0)
    # print(p10.shape, p11.shape)
    output = Network.recall(degraded_pattern, 100, False)
    print(degraded_pattern)
    print(output)
    print(np.sum(np.abs(output - degraded_pattern)))
    """
    Draw the pixels
    """
    old_grid = p10.reshape(32, 32)
    new_grid = output[0, :].reshape(32, 32)
    plt.imshow(old_grid, extent=(0, 32, 0, 32),
               interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    plt.show()
    plt.imshow(new_grid, extent=(0, 32, 0, 32),
               interpolation='nearest', cmap=plt.cm.get_cmap('binary'), aspect="auto")
    plt.show()

    # data[16, 16] = [254, 0, 0]       # Makes the middle pixel red
    # data[16, 17] = [0, 0, 255]       # Makes the next pixel blue

    # img = Image.fromarray(data)       # Create a PIL image
    # img.show()


if __name__ == '__main__':
    main()
