#!/usr/bin/env python3

import unittest

import numpy as np
import numpy.testing as npt
import theano as th
floatX = th.config.floatX

import DeepFried.containers as c
import DeepFried.test as t
import DeepFried.layers as l


class TestFC(unittest.TestCase):


    def test_feedforward(self):
        W = np.random.randn(10, 5).astype(floatX)
        b = np.random.randn(5).astype(floatX)
        fc1 = l.FullyConnected(W.shape[0], W.shape[1], W=W, b=b)
        fc2 = l.FullyConnected(W.shape[0], W.shape[1], bias=False, W=W)

        fn1t = t.mk_train_output_fn(fc1)
        fn1p = t.mk_train_output_fn(fc1)
        fn2t = t.mk_train_output_fn(fc2)
        fn2p = t.mk_train_output_fn(fc2)

        X = np.random.randn(100, 10).astype(floatX)
        npt.assert_allclose(np.dot(X, W) + b, fn1t(X))
        npt.assert_allclose(np.dot(X, W) + b, fn1p(X))
        npt.assert_allclose(np.dot(X, W)    , fn2t(X))
        npt.assert_allclose(np.dot(X, W)    , fn2p(X))


    def test_feedforward_reshaping(self):
        W = np.random.randn(6*6, 5*5).astype(floatX)
        b = np.random.randn(5*5).astype(floatX)
        fc1 = l.FullyConnected((6,6), (5,5), W=W, b=b)
        fc2 = l.FullyConnected((6,6), (5,5), bias=False, W=W)

        fn1t = t.mk_train_output_fn(fc1)
        fn1p = t.mk_train_output_fn(fc1)
        fn2t = t.mk_train_output_fn(fc2)
        fn2p = t.mk_train_output_fn(fc2)

        X = np.random.randn(100, 6, 6).astype(floatX)
        XW = np.dot(X.reshape(100,6*6), W)
        npt.assert_allclose((XW + b).reshape(100,5,5), fn1t(X))
        npt.assert_allclose((XW + b).reshape(100,5,5), fn1p(X))
        npt.assert_allclose((XW    ).reshape(100,5,5), fn2t(X))
        npt.assert_allclose((XW    ).reshape(100,5,5), fn2p(X))


class TestSoftmax(unittest.TestCase):


    def test_init_fc(self):
        foo = c.Sequence(
            l.FullyConnected(100, 100),
            l.Softmax(),
        )

        # First, they should all be initialized to NaNs.
        self.assertTrue(all(np.all(np.isnan(p.get_value())) for p in foo.params))

        foo.reinit(None)

        # Then, they should all be zero:
        for w, b in zip(foo.Ws, foo.bs):
            npt.assert_equal(w.get_value(), 0)
            npt.assert_equal(b.get_value(), 0)


class TestTanh(unittest.TestCase):


    def test_init_fc(self):
        fan = 2500

        foo = c.Sequence(
            l.FullyConnected(fan, fan),
            l.Tanh(init='Xavier'),
            l.FullyConnected(fan, fan),
            l.Tanh(init='XavierN'),
            l.FullyConnected(fan, fan),
            l.Tanh(init=0.5),
        )

        # Sanity check.
        for w in foo.Ws:
            self.assertEqual(w._df_init_kw['fan_in'], fan)
            self.assertEqual(w._df_init_kw['fan_out'], fan)

        # First, they should all be initialized to NaNs.
        self.assertTrue(all(np.all(np.isnan(p.get_value())) for p in foo.params))

        foo.reinit(None)

        # Then, they should all be close to zero-mean:
        for w, b in zip(foo.Ws, foo.bs):
            npt.assert_allclose(np.mean(w.get_value()), 0, atol=1e-3)
            npt.assert_equal(b.get_value(), 0)

        # And...

        # ... for Xavier, a min/max within [-√6/fan, √6/fan]
        x1 = np.sqrt(6/fan)
        npt.assert_allclose(np.min(foo.Ws[0].get_value()), -x1, rtol=2e-5)
        npt.assert_allclose(np.max(foo.Ws[0].get_value()),  x1, rtol=2e-5)

        # ... for XavierN, a std of 1/√fan
        x2 = np.sqrt(1/fan)
        npt.assert_allclose(np.std(foo.Ws[1].get_value()), x2, rtol=1e-3)

        # ... for fixed std, that fixed std
        npt.assert_allclose(np.std(foo.Ws[2].get_value()), 0.5, rtol=1e-3)


class TestReLU(unittest.TestCase):


    def test_init_fc(self):
        fan = 2500

        foo = c.Sequence(
            l.FullyConnected(fan, fan),
            l.ReLU(init='Xavier'),
            l.FullyConnected(fan, fan),
            l.ReLU(init='XavierN'),
            l.FullyConnected(fan, fan),
            l.ReLU(init='PReLU'),
            l.FullyConnected(fan, fan),
            l.ReLU(init=0.5),
        )

        # Sanity check.
        for w in foo.Ws:
            self.assertEqual(w._df_init_kw['fan_in'], fan)
            self.assertEqual(w._df_init_kw['fan_out'], fan)

        # First, they should all be initialized to NaNs.
        self.assertTrue(all(np.all(np.isnan(p.get_value())) for p in foo.params))

        foo.reinit(None)

        # Then, they should all be close to zero-mean:
        for w, b in zip(foo.Ws, foo.bs):
            npt.assert_allclose(np.mean(w.get_value()), 0, atol=1e-3)
            npt.assert_equal(b.get_value(), 0)

        # And...

        # ... for Xavier, a min/max within [-√6/fan, √6/fan]
        x1 = np.sqrt(6/fan)
        npt.assert_allclose(np.min(foo.Ws[0].get_value()), -x1, rtol=2e-5)
        npt.assert_allclose(np.max(foo.Ws[0].get_value()),  x1, rtol=2e-5)

        # ... for XavierN, a std of 1/√fan
        x2 = np.sqrt(1/fan)
        npt.assert_allclose(np.std(foo.Ws[1].get_value()), x2, rtol=1e-3)

        # ... for PReLU, a std of 2/√fan
        prelu = np.sqrt(2/fan)
        npt.assert_allclose(np.std(foo.Ws[2].get_value()), prelu, rtol=1e-3)

        # ... for fixed std, that fixed std
        npt.assert_allclose(np.std(foo.Ws[3].get_value()), 0.5, rtol=1e-3)
