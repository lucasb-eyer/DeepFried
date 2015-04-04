#!/usr/bin/env python3

import numpy as _np
import theano as _th

import DeepFried.util as _u
from DeepFried.layers import Layer


def mk_train_output_fn(model):
    """
    Creates a theano function that computes the output of a forward pass
    of the training version of given `model`.
    This is exactly what will be given to the cost function.
    Note that this doesn't minibatch at all.
    """
    Xin = _u.tuplize(model.make_inputs())
    Xout = model.train_expr(*Xin)
    return _th.function(inputs=Xin, outputs=_u.maybetuple(_u.tuplize(Xout)))


def mk_pred_output_fn(model):
    """
    Creates a theano function that computes the output of a forward pass
    of the prediction version of given `model`.
    Note that this doesn't minibatch at all.
    """
    Xin = _u.tuplize(model.make_inputs())
    Xout = model.pred_expr(*Xin)
    return _th.function(inputs=Xin, outputs=_u.maybetuple(_u.tuplize(Xout)))


class Identity(Layer):
    """
    A layer whose output is its input. Useful for unittesting.
    """

    def train_expr(self, *Xs, **kw):
        return Xs


class ConstantInitializer(Identity):

    def __init__(self, cW, cB):
        super(type(self), self).__init__()
        self.cW = cW
        self.cB = cB

    def weightinitializer(self):
        def init(shape, *a, **kw):
            return _np.full(shape, self.cW)
        return init


    def biasinitializer(self):
        def init(shape, *a, **kw):
            return _np.full(shape, self.cB)
        return init
