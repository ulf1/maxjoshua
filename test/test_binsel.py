import maxjoshua as mh
import numpy as np


def test1():
    X = np.random.random((1000, 21)) > 0.5
    idx, neg, rho, results = mh.binsel(
        X=X[:, :20], y=X[:, -1], n_select=5, oob_score=True)
    idx, neg, rho
    assert len(idx) == 5
    assert len(neg) == 5
    assert isinstance(rho, float)
