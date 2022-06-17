from .bootstrap_solutions import bootstrap_solutions
import numpy as np
from korr import pearson


# https://github.com/ulf1/numpy-linreg/blob/main/numpy_linreg/ridge.py
def ridge_lu(y, X, lam=0.01):
    n_vars = X.shape[1]
    P = np.dot(X.T, X) + lam * np.eye(n_vars)
    return np.linalg.solve(P, np.dot(X.T, y))


# https://github.com/ulf1/numpy-linreg/blob/main/numpy_linreg/metrics.py
def mse(y, X, beta):
    return np.mean((y - np.dot(X, beta))**2)


def fltsel(X: np.array,
           y: np.array,
           n_select: int = 5,
           max_rho: float = 0.4,
           preselect: float = None,
           n_draws: int = 50,
           subsample: float = 0.7,
           replace: bool = False,
           random_state: int = 42,
           unique: bool = True,
           oob_score: bool = True,
           verbose: bool = False) -> (list, list, float, list):
    # bootstrap some possible solutions
    solutions, oob = bootstrap_solutions(
        X, y=y, n_select=n_select, max_rho=max_rho,
        preselect=preselect, n_draws=n_draws,
        subsample=subsample, replace=replace,
        random_state=random_state, unique=unique,
        verbose=verbose, corr_fn=pearson)

    # find best way to negate features and store evaluation
    results = []
    for i, indicies in enumerate(solutions):
        feat_cols = solutions[i]
        oob_rows = oob[i]
        # fit model
        beta = ridge_lu(y=y[oob_rows], X=X[:, feat_cols][oob_rows, :])
        # compute RMSE loss
        loss = mse(y=y[oob_rows], X=X[:, feat_cols][oob_rows, :], beta=beta)
        # save it
        results.append([tuple(indicies), loss])

    # order results
    idx_sorted = np.argsort([r[-1] for r in results])
    results = [results[i] for i in idx_sorted]

    # done
    idx, rho = results[0]
    return idx, rho, results
