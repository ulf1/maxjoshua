from .bootstrap_solutions import bootstrap_solutions
from .enumerate_negations import enumerate_negations
import numpy as np


def binsel_hardvote(X, y, n_select=5, max_rho=0.4, preselect=None,
                    n_draws=50, subsample=0.7, replace=False,
                    random_state=42, unique=True):
    # bootstrap some possible solutions
    solutions = bootstrap_solutions(
        X, y=y, n_select=n_select, max_rho=max_rho,
        preselect=preselect, n_draws=n_draws,
        subsample=subsample, replace=replace,
        random_state=random_state, unique=unique)

    # find best way to negate features and store evaluation
    results = []
    for indicies in solutions:
        negate, rho = enumerate_negations(X[:, indicies], y)
        results.append([tuple(indicies), negate, rho])

    # order results
    results = np.array(results)
    idx_sorted = np.flip(np.argsort(results[:, 2]))
    results = results[idx_sorted]

    # done
    results = results.tolist()
    idx, neg, rho = results[0]
    return idx, neg, rho, results
