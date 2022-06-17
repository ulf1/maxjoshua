import numpy as np
from korr import bootcorr, mcc, mincorr


def bootstrap_solutions_all(X: np.array,
                            n_select: int = 5,
                            max_rho: float = 0.4,
                            n_draws: int = 50,
                            subsample: float = 0.7,
                            replace: bool = False,
                            random_state: int = 42,
                            unique: bool = True,
                            verbose: bool = False,
                            corr_fn=mcc) -> (np.array, list):
    # compute all Matthew's correlations between (Xi,Xj)
    rho3, _, oob = bootcorr(X, n_draws=n_draws, subsample=subsample,
                            replace=replace, random_state=random_state,
                            corr_fn=corr_fn)

    # for each draw find the lowest abs(corr(Xi,Xj))
    solutions = []   # store results in list
    for cmat in rho3:
        idxlo = mincorr(
            cmat, n_stop=n_select, max_rho=max_rho,
            verbose=verbose)
        solutions.append(sorted(idxlo))

    # done
    if unique:
        sol, idx = np.unique(solutions, axis=0, return_index=True)
        return sol, list(np.array(oob)[idx])
    else:
        return solutions, oob


def bootstrap_solutions_pre(X: np.array,
                            y: np.array,
                            n_select: int = 5,
                            max_rho: float = 0.4,
                            preselect: float = 0.8,
                            n_draws: int = 50,
                            subsample: float = 0.7,
                            replace: bool = False,
                            random_state: int = 42,
                            unique: bool = True,
                            verbose: bool = False,
                            corr_fn=mcc) -> (np.array, list):
    # convert to number of features to preselect
    n_features = len(X[0])
    if isinstance(preselect, float):
        n_pre = max(1, int(n_features * min(1.0, preselect)))
    elif isinstance(preselect, int):
        n_pre = max(1, min(n_features, preselect))
    else:
        n_pre = None  # all

    # compute all Matthew's correlations between (y,X) and (Xi,Xj)
    rho3, _, oob = bootcorr(np.c_[y, X], n_draws=n_draws, subsample=subsample,
                            replace=replace, random_state=random_state,
                            corr_fn=corr_fn)

    # for each draw find the lowest abs(corr(Xi,Xj))
    solutions = []   # store results in list
    for cmat in rho3:
        # a) sort abs corr(y,x), b) sort by biggest, c) add +1 to index bc y
        idxft = np.flip(np.argsort(np.abs(cmat[0, 1:]))) + 1
        idxft = idxft[:n_pre]
        # find the smallest corr(xi,xj) between selected features
        idxlo = mincorr(
            cmat[:, idxft][idxft, :],
            n_stop=n_select,
            max_rho=max_rho,
            verbose=verbose)
        # convert indicies back to 'X' matrix (not 'dat')
        idxlo = list(np.array(idxft[idxlo]) - 1)
        # save it
        solutions.append(sorted(idxlo))

    # done
    if unique:
        sol, idx = np.unique(solutions, axis=0, return_index=True)
        return sol, list(np.array(oob)[idx])
    else:
        return solutions, oob


def bootstrap_solutions(X: np.array,
                        y: np.array = None,
                        n_select: int = 5,
                        max_rho: float = 0.4,
                        preselect: float = None,
                        n_draws: int = 50,
                        subsample: float = 0.7,
                        replace: bool = False,
                        random_state: int = 42,
                        unique: bool = True,
                        verbose: bool = False,
                        corr_fn=mcc) -> (np.array, list):
    if preselect and y is not None:
        return bootstrap_solutions_pre(
            X, y, n_select=n_select, max_rho=max_rho, preselect=preselect,
            n_draws=n_draws, subsample=subsample, replace=replace,
            random_state=random_state, unique=unique, verbose=verbose,
            corr_fn=corr_fn)
    else:
        return bootstrap_solutions_all(
            X, n_select=n_select, max_rho=max_rho,
            n_draws=n_draws, subsample=subsample, replace=replace,
            random_state=random_state, unique=unique, verbose=verbose,
            corr_fn=corr_fn)
