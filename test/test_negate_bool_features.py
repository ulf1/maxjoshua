import unittest
from binsel import negate_bool_features
import numpy as np
import numpy.testing as npt


class Test_negate_bool_features(unittest.TestCase):

    def test1(self):
        A = [[1, 1], [0, 0], [1, 1]]
        negate = (0, 0)
        B = [[1, 1], [0, 0], [1, 1]]
        C = negate_bool_features(A, negate)
        npt.assert_array_equal(B, C.astype(int))

    def test2(self):
        A = [[1, 1], [0, 0]]
        negate = (1, 1)
        B = [[0, 0], [1, 1]]
        C = negate_bool_features(A, negate)
        npt.assert_array_equal(B, C.astype(int))

# run
if __name__ == '__main__':
    unittest.main()
