import maxjoshua as mh
import numpy as np
import numpy.testing as npt


def test1():
    A = [[1, 1], [0, 0], [1, 1]]
    negate = (0, 0)
    B = [[1, 1], [0, 0], [1, 1]]
    C = mh.negate_bool_features(A, negate)
    npt.assert_array_equal(B, C.astype(int))


def test2():
    A = [[1, 1], [0, 0]]
    negate = (1, 1)
    B = [[0, 0], [1, 1]]
    C = mh.negate_bool_features(A, negate)
    npt.assert_array_equal(B, C.astype(int))
