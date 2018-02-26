import numpy as np


class HRNN:
    def __init__(self, X):
        self.X = X
        self.weights = X.T.dot(X)

    def recall(self, patterns):
        for i, pattern in enumerate(patterns):
            patterns[i, :] = np.where(self.weights.dot(pattern) >= 0, 1, -1)
        return patterns

