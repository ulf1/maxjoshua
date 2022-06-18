import unittest
from maxjoshua import hard_voting
import numpy as np
import numpy.testing as npt


class Test_hard_voting(unittest.TestCase):
    def test1(self):
        A = np.array([1, 0, 2, 1])
        B = np.array([1, 0, 1, 1])
        C = hard_voting(A)
        npt.assert_array_equal(B, C.astype(int))

    def test2(self):
        A = [1, 0, 2, 1]
        B = np.array([1, 0, 1, 1])
        C = hard_voting(A)
        npt.assert_array_equal(B, C.astype(int))
