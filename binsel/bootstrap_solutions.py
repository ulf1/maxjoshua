import numpy as np
from korr import bootcorr, mcc, mincorr


def bootstrap_solutions_all(X, n_select=5, max_rho=0.4, n_draws=50,
                            subsample=0.7, replace=False, random_state=42,
                            unique=True):
    # compute all Matthew's correlations between (Xi,Xj)
    rho3, _, oob = bootcorr(X, n_draws=n_draws, subsample=subsample,
                       replace=replace, random_state=random_state,
                       corr_fn=mcc)

    # for each draw find the lowest abs(corr(Xi,Xj))
    solutions = []   # store results in list
    for cmat in rho3:
        idxlo = mincorr(cmat, n_stop=n_select, max_rho=max_rho)
        solutions.append(sorted(idxlo))

    # done
    if unique:
        sol, idx = np.unique(solutions, axis=0, return_index=True)
        return sol, list(np.array(oob)[idx])
    else:
        return solutions, oob


def bootstrap_solutions_pre(X, y, n_select=5, max_rho=0.4, preselect=0.8,
                            n_draws=50, subsample=0.7, replace=False,
                            random_state=42, unique=True):
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
                       corr_fn=mcc)

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
            max_rho=max_rho)
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


def bootstrap_solutions(X, y=None, n_select=5, max_rho=0.4, preselect=None,
                        n_draws=50, subsample=0.7, replace=False,
                        random_state=42, unique=True):
    if preselect and y is not None:
        return bootstrap_solutions_pre(
            X, y, n_select=n_select, max_rho=max_rho, preselect=preselect,
            n_draws=n_draws, subsample=subsample, replace=replace,
            random_state=random_state, unique=unique)
    else:
        return bootstrap_solutions_all(
            X, n_select=n_select, max_rho=max_rho,
            n_draws=n_draws, subsample=subsample, replace=replace,
            random_state=random_state, unique=unique)
