#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.testing as npt
import theano as th
floatX = th.config.floatX

import DeepFried.containers as c
import DeepFried.test as t
import DeepFried.layers as l


class TestSequence(unittest.TestCase):


    def setUp(self):
        self.X = np.random.randn(200, 30).astype(floatX)


    def test_output_simple(self):
        foo = c.Sequence(
            t.Identity(),
            t.Identity(),
            t.Identity(),
        )

        npt.assert_array_equal(t.mk_train_output_fn(foo)(self.X), self.X)
        npt.assert_array_equal(t.mk_pred_output_fn(foo)(self.X), self.X)


    def test_initializers_implicit(self):
        foo = c.Sequence(
            l.FullyConnected(10, 10),
            t.ConstantInitializer(42, 12),
            l.FullyConnected(10, 10),
            l.FullyConnected(10, 10),
            l.FullyConnected(10, 10),
            l.FullyConnected(10, 10),
            t.ConstantInitializer(43, 13),
            l.FullyConnected(10, 10),
        )

        # First, they should all be initialized to NaNs.
        self.assertTrue(all(np.all(np.isnan(p.get_value())) for p in foo.params))

        foo.reinit(None)

        # Then, they should *all* be initialized by the next initializer.
        # This implies the last layer not to be initialized!
        self.assertTrue(np.all(foo.Ws[0].get_value() == 42))
        self.assertTrue(np.all(foo.bs[0].get_value() == 12))
        for i in range(1, 5):
            self.assertTrue(np.all(foo.Ws[i].get_value() == 43))
            self.assertTrue(np.all(foo.bs[i].get_value() == 13))
        self.assertTrue(np.all(np.isnan(foo.Ws[-1].get_value())))
        self.assertTrue(np.all(np.isnan(foo.bs[-1].get_value())))


    def test_initializers_explicit_npast(self):
        foo = c.Sequence()
        fc1 = foo.append(l.FullyConnected(10, 10))
        ci1 = foo.append(t.ConstantInitializer(42, 12), init_previous=False)
        fc2 = foo.append(l.FullyConnected(10, 10))
        fc3 = foo.append(l.FullyConnected(10, 10))
        fc4 = foo.append(l.FullyConnected(10, 10))
        fc5 = foo.append(l.FullyConnected(10, 10))
        ci2 = foo.append(t.ConstantInitializer(43, 13), init_previous=3)
        ci3 = foo.append(t.ConstantInitializer(44, 14), init_previous=3, initializes=fc2)

        # First, they should all be initialized to NaNs.
        self.assertTrue(all(np.all(np.isnan(p.get_value())) for p in foo.params))

        foo.reinit(None)

        # Then, 1 should still be NaNs, ...
        self.assertTrue(np.all(np.isnan(foo.Ws[0].get_value())))
        self.assertTrue(np.all(np.isnan(foo.bs[0].get_value())))
        # ... 3 should be 43/13 and ...
        self.assertTrue(np.all(foo.Ws[2].get_value() == 43))
        self.assertTrue(np.all(foo.bs[2].get_value() == 13))
        # ... 2,4,5 should be 44/14.
        for i in (1, 3, 4):
            self.assertTrue(np.all(foo.Ws[i].get_value() == 44))
            self.assertTrue(np.all(foo.bs[i].get_value() == 14))


class TestParallel(unittest.TestCase):


    def setUp(self):
        self.X = np.random.randn(200, 30).astype(floatX)


    def test_output_simple(self):
        foo = c.Parallel(
            t.Identity(),
            t.Identity(),
            t.Identity(),
        )

        outt = t.mk_train_output_fn(foo)(self.X)
        outp = t.mk_pred_output_fn(foo)(self.X)

        self.assertEqual(len(outt), 3)
        self.assertEqual(len(outp), 3)

        self.assertTrue(all(np.all(X == self.X) for X in outt))
        self.assertTrue(all(np.all(X == self.X) for X in outp))


class TestContainerCombos(unittest.TestCase):


    def setUp(self):
        self.X = np.random.randn(200, 30).astype(floatX)


    def test_nested_sequences(self):
        foo = c.Sequence(c.Sequence(t.Identity()))

        npt.assert_array_equal(t.mk_train_output_fn(foo)(self.X), self.X)
        npt.assert_array_equal(t.mk_pred_output_fn(foo)(self.X), self.X)


        foo = c.Sequence(c.Sequence(c.Sequence(t.Identity())))

        npt.assert_array_equal(t.mk_train_output_fn(foo)(self.X), self.X)
        npt.assert_array_equal(t.mk_pred_output_fn(foo)(self.X), self.X)


    def test_mix(self):
        foo = c.Parallel(c.Sequence(t.Identity()))

        npt.assert_array_equal(t.mk_train_output_fn(foo)(self.X), self.X)
        npt.assert_array_equal(t.mk_pred_output_fn(foo)(self.X), self.X)


        foo = c.Sequence(c.Parallel(c.Sequence(t.Identity())))

        npt.assert_array_equal(t.mk_train_output_fn(foo)(self.X), self.X)
        npt.assert_array_equal(t.mk_pred_output_fn(foo)(self.X), self.X)


    def test_deeper_mix(self):
        foo = c.Sequence(
            t.Identity(),
            c.Parallel(
                t.Identity(),
                c.Sequence(
                    t.Identity(),
                    t.Identity(),
                )
            )
        )

        outt = t.mk_train_output_fn(foo)(self.X)
        outp = t.mk_pred_output_fn(foo)(self.X)

        self.assertEqual(len(outt), 2)
        self.assertEqual(len(outp), 2)

        self.assertTrue(all(np.all(X == self.X) for X in outt))
        self.assertTrue(all(np.all(X == self.X) for X in outp))
