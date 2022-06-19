from .bootstrap_solutions import bootstrap_solutions
import numpy as np
import numba
import numpy_linreg.ridge
import numpy_linreg.metrics


@numba.njit
def numba_corrcoef_pearson(x):
    return np.corrcoef(x, rowvar=False)


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
           verbose: bool = False,
           l2: float = 0.01) -> (list, list, float, list):
    # bootstrap some possible solutions
    solutions, oob = bootstrap_solutions(
        X, y=y,
        n_select=n_select,
        max_rho=max_rho,
        preselect=preselect,
        n_draws=n_draws,
        subsample=subsample,
        replace=replace,
        random_state=random_state,
        unique=unique,
        verbose=verbose,
        corr_fn=numba_corrcoef_pearson)

    # find best way to negate features and store evaluation
    results = []
    for i, indicies in enumerate(solutions):
        feat_cols = solutions[i]
        oob_rows = oob[i]
        # fit model
        beta = numpy_linreg.ridge.lu(
            y=y[oob_rows], X=X[:, feat_cols][oob_rows, :], lam=l2)
        # compute RMSE loss
        loss = numpy_linreg.metrics.rmse(
            y=y[oob_rows], X=X[:, feat_cols][oob_rows, :], beta=beta)
        # save it
        results.append([tuple(indicies), beta.tolist(), loss])

    # order results
    idx_sorted = np.argsort([r[-1] for r in results])
    results = [results[i] for i in idx_sorted]

    # done
    idx, beta, loss = results[0]
    return idx, beta, loss, results
