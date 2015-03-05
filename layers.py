#!/usr/bin/env python3

import numpy as _np
import theano as _th
import theano.tensor as _T

from DeepFried.util import tuplize as _tuplize


class Layer(object):
    """
    Abstract superclass of all layers with a dual purpose:

    1. Define common interface all layers should implement.
    2. Implement boilerplate code which is the same for any kind of layer.

    Note that only the second point is actually a valid one; introducing a
    hierarchy of classes for the first one would be unpythonic.
    """


    def __init__(self):
        # A layer which doesn't have any parameters should still have an empty
        # attribute for that to make generic code easier.
        self.Ws = []
        self.bs = []
        self.params = []


    def train_expr(self, X):
        raise NotImplementedError("You need to implement `train_expr`!")


    def pred_expr(self, *args, **kwargs):
        # Most layers act exactly the same for training and for testing; this
        # way they'll only need to build their expression once.
        return self.train_expr(*args, **kwargs)


    def newweight(self, *args, **kwargs):
        """
        Creates a new shared weight parameter variable called `name` and of
        given `shape` which will be learned by the optimizers.

        - `name`: The name the variable should have.
        - `shape`: The shape the weight should have.
        - `val`: Optionally a value (numpy array or shared array) to use as
                 initial value.
        - `empty`: function which should return a new value for the variable,
                   will only be called if `val` is None.
        """
        p = self._newparam(*args, **kwargs)
        self.Ws.append(p)
        self.params.append(p)
        return p


    def newbias(self, *args, **kwargs):
        """
        Creates a new shared bias parameter variable called `name` and of
        given `shape` which will be learned by the optimizers.

        - `name`: The name the variable should have.
        - `shape`: The shape the weight should have.
        - `val`: Optionally a value (numpy array or shared array) to use as
                 initial value.
        - `empty`: function which should return a new value for the variable,
                   will only be called if `val` is None.
        """
        p = self._newparam(*args, **kwargs)
        self.bs.append(p)
        self.params.append(p)
        return p


    def _newparam(self, name, shape, val=None, empty=None):
        name = "{}{}".format(name, 'x'.join(map(str, shape)))

        if val is None:
            if empty is None:
                val = _np.full(shape, _np.nan, dtype=_th.config.floatX)
            else:
                val = empty(shape, dtype=_th.config.floatX)
        if isinstance(val, _np.ndarray):
            val = _th.shared(val.astype(_th.config.floatX).reshape(shape), name=name)
        else:
            raise ValueError("Couldn't understand parameters for parameter creation.")

        return val


class FullyConnected(Layer):
    """
    Fully-connected layer is the typical "hidden" layer. Basically implements
    a GEMV or, because it's batched, a GEMM.
    """


    def __init__(self, inshape, outshape, bias=True, W=None, b=None):
        """
        Creates a fully-connected (i.e. linear, hidden) layer taking as input
        a minibatch of `inshape`-shaped elements and giving as output a
        minibatch of `outshape`-shaped elements.

        - `inshape`: The shape of the input, excluding the leading minibatch
                     dimension. Usually, this is just a number, but it may be a
                     tuple, in which case the input is first flattened.
        - `outshape`: The shape of the output, excluding the leading minibatch
                      dimension. Usually, this is just a number, but it may be
                      a tuple, in which case the output is reshaped to this.
        - `bias`: Whether to use a bias term or not.
                  For example, if a minibatch-normalization follows, a bias
                  term is useless.
        - `W`: Optional initial value for the weights.
        - `b`: Optional initial value for the bias.
        """
        super(FullyConnected, self).__init__()

        self.inshape = _tuplize(inshape)
        self.outshape = _tuplize(outshape)

        self.fan_in = _np.prod(self.inshape)
        self.fan_out = _np.prod(self.outshape)

        self.W_shape = (self.fan_in, self.fan_out)
        self.W = self.newweight("W_fc", self.W_shape, W)

        self.bias = bias
        if self.bias:
            self.b_shape = (self.fan_out,)
            self.b = self.newbias("b_fc", self.b_shape, b, _np.zeros)


    def train_expr(self, X):
        """
        Returns an expression computing the output at training-time given a
        symbolic input `X`.
        """
        batchsize = X.shape[0]

        # For non-1D inputs, add a flattening step for convenience.
        if len(self.inshape) > 1:
            # (Don't forget the first dimension is the minibatch!)
            X = X.flatten(2)

        out = _T.dot(X, self.W)

        if self.bias:
            out += self.b

        # And for non-1D outputs, add a reshaping step for convenience.
        if len(self.outshape) > 1:
            # (Again, don't forget the first dimension is the minibatch!)
            out.reshape((batchsize,) + self.outshape)

        return out


class ReLU(Layer):
    """
    Typical ReLU nonlinearity layer of newschool nnets.
    """


    def __init__(self, leak=0, cap=None, init='Xavier'):
        """
        Creates a ReLU layer which computes the elementwise application of the
        optionally capped ReLU or Leaky ReLU function:

            out = min(max(leak*X, X), cap)

        - `leak`: Number to use as slope for the "0" part. Defaults to 0, i.e.
                  standard ReLU.
        - `cap`: Number to use as max value of the linear part, as in Alex'
                 CIFAR Convolutional Deep Belief Nets.
                 If `None` (the dfeault), do not apply any cropping.
        - `init`: The initialization technique to use for initializing another
                  layer's weights. Currently available techniques are:
            - `'Xavier'`: Uniform-random Xavier initialization from [1].
            - `'XavierN'`: Normal-random [2] way of achieving Xavier
                           initialization.
            - `'PReLU'`: Normal-random initialization for PReLU[2].
            - A number: Standard deviation (sigma) of the normal distribution
                        to sample from.

        1: Understanding the difficulty of training deep feedforward neural networks.
        2: Delving Deep into Rectifiers.
        """
        super(ReLU, self).__init__()

        self.leak = leak
        self.cap = cap
        self.init = init


    def train_expr(self, X):
        """
        Returns an expression computing the output at training-time given a
        symbolic input `X`.
        """
        if self.cap is None:
            return _T.maximum(self.leak*X, X)
        else:
            return X.clip(self.leak*X, self.cap)


    def weightinitializer(self):
        """
        Returns a function which can be used to initialize weights of a
        preceding layer. Which function is returned depends on the value of
        the `init` flag passed in the constructor.

        The returned function's signature is:

        def init(rng, shape, fan_in, fan_out)

        It returns a numpy array with initial values for the weights.

        - `rng`: The random number generator to be used.
        - `shape`: The shape of the initial weights to return.
        - `fan_in`: The fan-in of the weights, used in some initializations.
        - `fan_out`: The fan-out of the weights, used in some initializations.
        """
        if self.init == 'Xavier':
            def init(rng, shape, fan_in, fan_out):
                fan_mean = (fan_in+fan_out)/2
                return rng.uniform(-_np.sqrt(6/fan_mean), _np.sqrt(6/fan_mean), shape)
            return init
        elif self.init == 'XavierN':
            def init(rng, shape, fan_in, fan_out):
                fan_mean = (fan_in+fan_out)/2
                return _np.sqrt(1/fan_mean)*rng.standard_normal(shape)
            return init
        elif self.init == 'PReLU':
            def init(rng, shape, fan_in, fan_out):
                fan_mean = (fan_in+fan_out)/2
                return _np.sqrt(2/fan_mean)*rng.standard_normal(shape)
            return init
        else:
            def init(rng, shape, fan_in, fan_out):
                return self.init*rng.standard_normal(shape)
            return init


    def biasinitializer(self):
        """
        See the documentation of `weightsinitializer`, the difference is that
        biases typically use a different (simper) initialization method.
        """
        def init(rng, shape, fan_in, fan_out):
            return _np.zeros(shape)
        return init
