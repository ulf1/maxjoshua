from sklearn.base import BaseEstimator, ClassifierMixin
from .binsel_hardvote import (
    binsel_hardvote, negate_bool_features, hard_voting)
import numpy as np


class BinSel(BaseEstimator, ClassifierMixin):
    def __init__(self, n_select=5, max_rho=0.4, preselect=None,
                 n_draws=50, subsample=0.7, replace=False,
                 random_state=42, unique=True, oob_score=True,
                 verbose=False):
        self.n_select = n_select
        self.max_rho = max_rho
        self.preselect = preselect
        self.n_draws = n_draws
        self.subsample = subsample
        self.replace = replace
        self.random_state = random_state
        self.unique = unique
        self.oob_score = oob_score
        self.verbose = verbose

    def fit(self, X, y, **fit_params):
        self.idx, self.neg, self.rho, self.res = binsel_hardvote(
            X, y,
            n_select=self.n_select,
            max_rho=self.max_rho,
            preselect=self.preselect,
            n_draws=self.n_draws,
            subsample=self.subsample,
            replace=self.replace,
            random_state=self.random_state,
            unique=self.unique,
            oob_score=self.oob_score,
            verbose=self.verbose)
        return self

    def predict_proba(self, X):
        """The average of all binary votes.
        Only makes sense for high n_select
        """
        X_tmp = negate_bool_features(X[:, self.idx], self.neg)
        y_proba = np.sum(X_tmp, axis=1) / self.n_select
        return np.c_[1 - y_proba, y_proba]

    def predict(self, X):
        X_tmp = negate_bool_features(X[:, self.idx], self.neg)
        y_vote = hard_voting(X_tmp)
        return y_vote

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y == y_pred)
