import maxjoshua as mh
import numpy as np
import numpy.testing as npt


def test1():
    A = np.array([1, 0, 2, 1])
    B = np.array([1, 0, 1, 1])
    C = mh.hard_voting(A)
    npt.assert_array_equal(B, C.astype(int))


def test2():
    A = [1, 0, 2, 1]
    B = np.array([1, 0, 1, 1])
    C = mh.hard_voting(A)
    npt.assert_array_equal(B, C.astype(int))
