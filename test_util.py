#!/usr/bin/env python3

import unittest

import DeepFried.util as u

import numpy as np
import numpy.testing as npt


class TestUtilFunctions(unittest.TestCase):


    def test_tuplize(self):
        self.assertEqual(u.tuplize(1), (1,))
        self.assertEqual(u.tuplize((1,)), (1,))
        self.assertEqual(u.tuplize((1,2)), (1,2))
        self.assertEqual(u.tuplize([1]), ([1],))
        self.assertEqual(u.tuplize("a"), ("a",))
        self.assertEqual(u.tuplize("ab"), ("ab",))


    def test_maybetuple(self):
        self.assertEqual(u.maybetuple((0,1)), (0,1))
        self.assertEqual(u.maybetuple((0,)), 0)
        self.assertEqual(u.maybetuple(range(2)), (0,1))
        self.assertEqual(u.maybetuple(range(1)), 0)
        with self.assertRaises(TypeError):
            u.maybetuple(3)


    def test_batched_1d(self):
        # 1D array with fitting size
        l = list(u.batched(3, np.arange(6)))
        self.assertEqual(len(l), 2)
        npt.assert_array_equal(l[0], [0,1,2])
        npt.assert_array_equal(l[1], [3,4,5])

        # 1D arrays with fitting size
        l = list(u.batched(3, np.arange(6), np.arange(6)))
        self.assertEqual(len(l), 2)
        npt.assert_array_equal(l[0][0], [0,1,2])
        npt.assert_array_equal(l[0][1], [0,1,2])
        npt.assert_array_equal(l[1][0], [3,4,5])
        npt.assert_array_equal(l[1][1], [3,4,5])

        # 1D array with leftover
        l = list(u.batched(3, np.arange(7)))
        self.assertEqual(len(l), 3)
        npt.assert_array_equal(l[0], [0,1,2])
        npt.assert_array_equal(l[1], [3,4,5])
        npt.assert_array_equal(l[2], [6])

        # 1D arrays with leftover
        l = list(u.batched(3, np.arange(7), np.arange(7)))
        self.assertEqual(len(l), 3)
        npt.assert_array_equal(l[0][0], [0,1,2])
        npt.assert_array_equal(l[0][1], [0,1,2])
        npt.assert_array_equal(l[1][0], [3,4,5])
        npt.assert_array_equal(l[1][1], [3,4,5])
        npt.assert_array_equal(l[2][0], [6])
        npt.assert_array_equal(l[2][1], [6])


    def test_batched_2d(self):
        # 2D array with fitting size
        l = list(u.batched(3, np.array([[2*i, 2*i+1] for i in range(6)])))
        self.assertEqual(len(l), 2)
        npt.assert_array_equal(l[0], [[0,1],[2,3],[4,5]])
        npt.assert_array_equal(l[1], [[6,7],[8,9],[10,11]])

        # 2D arrays with fitting size
        l = list(u.batched(3, np.array([[2*i, 2*i+1] for i in range(6)]), np.arange(6)))
        self.assertEqual(len(l), 2)
        npt.assert_array_equal(l[0][0], [[0,1],[2,3],[4,5]])
        npt.assert_array_equal(l[0][1], [0, 1, 2])
        npt.assert_array_equal(l[1][0], [[6,7],[8,9],[10,11]])
        npt.assert_array_equal(l[1][1], [3, 4, 5])

        # 2D array with leftover
        l = list(u.batched(3, np.array([[2*i, 2*i+1] for i in range(7)])))
        self.assertEqual(len(l), 3)
        npt.assert_array_equal(l[0], [[0,1],[2,3],[4,5]])
        npt.assert_array_equal(l[1], [[6,7],[8,9],[10,11]])
        npt.assert_array_equal(l[2], [[12,13]])


        # 2D arrays with leftover
        l = list(u.batched(3, np.array([[2*i, 2*i+1] for i in range(7)]), np.arange(7)))
        self.assertEqual(len(l), 3)
        npt.assert_array_equal(l[0][0], [[0,1],[2,3],[4,5]])
        npt.assert_array_equal(l[0][1], [0, 1, 2])
        npt.assert_array_equal(l[1][0], [[6,7],[8,9],[10,11]])
        npt.assert_array_equal(l[1][1], [3, 4, 5])
        npt.assert_array_equal(l[2][0], [[12,13]])
        npt.assert_array_equal(l[2][1], [6])

