from itertools import product
from .negate_bool_features import negate_bool_features
from .hard_voting import hard_voting
from korr import confusion, confusion_to_mcc
import numpy as np


def enumerate_negations(X: np.array, y: np.array) -> (list, float):
    n_features = len(X[0])  # warning 15 => 32768 combinations!

    best_neg = None
    best_rho = 0.0
    for negate in product([0, 1], repeat=n_features):
        Xtmp = negate_bool_features(X, negate)
        Yvote = hard_voting(Xtmp)
        rho = confusion_to_mcc(confusion(y, Yvote))
        if rho > best_rho:
            best_neg = negate
            best_rho = rho

    return best_neg, best_rho
