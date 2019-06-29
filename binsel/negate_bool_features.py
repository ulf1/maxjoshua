import numpy as np


def negate_bool_features(Xin, negate):
    X = np.array(Xin)
    neg = np.array(negate)
    return (X * (1 - neg * 2) + neg).astype(bool)
