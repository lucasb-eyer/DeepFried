#!/usr/bin/env python3

from DeepFried.layers import Layer

import numpy as _np
import theano as _th
import theano.tensor as _T


class Softmax(Layer):
    """
    A softmax layer is commonly used as output layer in multi-class logistic
    regression, the default type of multi-class classification in current-gen
    nnets.

    This layer really should follow a fully-connected layer with the number of
    classes to be predicted as output.
    """


    def __init__(self):
        super(Softmax, self).__init__()


    def train_expr(self, X):
        """
        Returns an expression computing the output at training-time given a
        symbolic input `X`.
        """
        return _T.nnet.softmax(X)


    def weightinitializer(self):
        """ See the documentation of `Layer`. """
        def init(shape, rng, *a, **kw):
            return _np.zeros(shape)
        return init


    def biasinitializer(self):
        """ See the documentation of `Layer`. """
        def init(shape, rng, *a, **kw):
            return _np.zeros(shape)
        return init


