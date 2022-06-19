import maxjoshua as mh
import korr
import numpy as np


def korr_matthews_mcc(x):
    return korr.mcc(x)[0]


def test1():
    X = np.random.random((1000, 21)) > 0.5
    solutions, oob = mh.bootstrap_solutions_pre(
        X=X[:, :20], y=X[:, -1], corr_fn=korr_matthews_mcc,
        n_select=5, preselect=1.0, n_draws=20, subsample=0.3)
    assert len(solutions) == 20


def test2():
    X = np.random.random((1000, 21)) > 0.5
    solutions, oob = mh.bootstrap_solutions_pre(
        X=X[:, :20], y=X[:, -1], corr_fn=korr_matthews_mcc,
        n_select=5, preselect=0.6, n_draws=20, subsample=0.3)
    assert len(solutions) == 20


def test3():
    X = np.random.random((1000, 21)) > 0.5
    solutions, oob = mh.bootstrap_solutions_all(
        X, corr_fn=korr_matthews_mcc,
        n_select=5, n_draws=20, subsample=0.3)
    assert len(solutions) == 20
