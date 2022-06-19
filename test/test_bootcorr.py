import maxjoshua as mh
import numpy as np


def test1():
    def corr_fn(x):
        return np.corrcoef(x, rowvar=False)
    X = np.random.random((100, 3))
    rho3, _ = mh.bootcorr(X, corr_fn=corr_fn, n_draws=10)
    assert rho3.shape == (10, 3, 3)
