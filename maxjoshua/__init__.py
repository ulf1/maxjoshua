__version__ = '0.4.2'

from .negate_bool_features import negate_bool_features
from .hard_voting import hard_voting
from .enumerate_negations import enumerate_negations
from .bootstrap_solutions import (
    bootstrap_solutions_all, bootstrap_solutions_pre, bootstrap_solutions)
from .binsel import binsel
from .fltsel import fltsel
from .bootcorr import bootcorr
from .sparsenn import (
    pretrain_submodels, SparseLayerAsEnsemble, InverseTransformer)
