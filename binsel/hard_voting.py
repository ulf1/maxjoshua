import numpy as np


def hard_voting(X):
    # convert to numpy array
    if isinstance(X, (list, tuple)):
        X = np.array(X)

    # return vector directly
    if len(X.shape) == 1:
        return X > 0
    else:
        # majority vote
        n_votes = len(X[0])
        n_majority = int(n_votes / 2.0) + 1
        return np.count_nonzero(X, axis=1) >= n_majority
