#!/usr/bin/env python3

import numpy as _np
import theano as _th
import theano.tensor as _T

from DeepFried.util import tuplize as _tuplize
from DeepFried.util import check_random_state as _check_random_state


class Layer(object):
    """
    Abstract superclass of all layers with a dual purpose:

    1. Define common interface all layers should implement.
    2. Implement boilerplate code which is the same for any kind of layer.

    Note that only the second point is actually a valid one; introducing a
    hierarchy of classes for the first one would be unpythonic.

    Additional methods that *may* be implemented by some layers and (TODO)
    probably should be formalized and documented somewhere else:

    - `weightinitializer(self)`:
        Returns a function which can be used to initialize weights of a
        preceding layer. Which function is returned depends on the value of
        the `init` flag passed in the constructor.

        The returned function's signature is:

        def init(shape, rng, fan_in, fan_out)

        It returns a numpy array with initial values for the weights.

        - `shape`: The shape of the initial weights to return.
        - `rng`: The random number generator to be used.
        - `fan_in`: The fan-in of the weights, used in some initializations.
        - `fan_out`: The fan-out of the weights, used in some initializations.

    - `biasinitializer(self)`:
        See the documentation of `weightsinitializer`, the difference is that
        biases typically use a different (simper) initialization method.
    """


    def __init__(self):
        # A layer which doesn't have any parameters should still have an empty
        # attribute for that to make generic code easier.
        self.Ws = []
        self.bs = []
        self.params = []

        # This contains the initializers to be used for each parameter.
        self.inits = {}


    def train_expr(self, *Xs):
        """
        Returns an expression or a tuple of expressions computing the
        output(s) at training-time given the symbolic input(s) in `Xs`.
        """
        raise NotImplementedError("You need to implement `train_expr` or `train_exprs` for {}!".format(type(self).__name__))


    def pred_expr(self, *Xs):
        """
        Returns an expression or a tuple of expressions to be computed jointly
        during prediction given the symbolic input(s) in `Xs`.

        This defaults to calling `train_expr`, which is good enough for all
        layers which don't change between training and prediction.
        """
        return self.train_expr(*Xs)


    def make_inputs(self, name="Xin"):
        """
        Returns a Theano tensor or a tuple of Theano tensors with given `names`
        of the dimensions which this layer takes as input.

        **NOTE** that this needs to include a leading dimension for the
        minibatch.

        Defaults to returning a single matrix, i.e. each datapoint is a vector.
        """
        return _T.matrix(name)


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
        p, i = self._newparam(*args, **kwargs)
        self.Ws.append(p)
        self.params.append(p)
        self.inits[p] = i
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
        p, i = self._newparam(*args, **kwargs)
        self.bs.append(p)
        self.params.append(p)
        self.inits[p] = i
        return p


    def _newparam(self, name, shape, val=None, init=None):
        name = "{}{}".format(name, 'x'.join(map(str, shape)))

        if val is None:
            if init is None:
                init = lambda shape, *a, **kw: _np.full(shape, _np.nan, dtype=_th.config.floatX)
        elif isinstance(val, _np.ndarray):
            init = lambda *a, **kw: val
        elif isinstance(val, _T.TensorVariable):
            # When using an existing theano shared variable, don't store nay
            # initializer as "the origina" probably already has one.
            return val, None
        else:
            raise ValueError("Couldn't understand parameters for parameter creation.")

        return _th.shared(init(shape).astype(_th.config.floatX), name=name), init


    def reinit(self, rng):
        """
        Sets the value of each parameter to that returned by a call to the
        initializer which has been registered for it.

        If a parameter has a `_df_init_kw` attribute, it is supposed to be a
        dict and will be passed as additional keyword arguments to the
        initializer.
        """
        rng = _check_random_state(rng)

        for p in self.params:
            if p not in self.inits:
                raise RuntimeError("Want to reinit layer parameter '{}' but no initializer registered for it.".format(p.name))
            kw = {}
            if hasattr(p, "_df_init_kw"):
                kw = p._df_init_kw
            p.set_value(self.inits[p](p.get_value().shape, rng, **kw).astype(p.dtype))


    def batch_agg(self):
        """
        Returns a function which can be used for aggregating the outputs of
        minibatches in one epoch.

        If the layer has multiple outputs, it should return a tuple of
        aggregator functions.

        This default implementation just concatenates them, which is a
        sensible behaviour for almost all kinds of layers.
        """
        def agg(outputs):
            return _np.concatenate(outputs)
        return agg


    def ensembler(self):
        """
        Returns a function which, given a list of multiple outputs for a
        single minibatch, computes the output of ensembling them. This is
        useful e.g. for ensembling the predictions of multiple augmentations.

        If the layer has multiple outputs, it should return a tuple of
        ensembler functions.

        This default implementation computes the average of the outputs, whic
        is a sensible behaviour for most kinds of outputs.
        """
        def ens(outputs):
            return sum(outputs)/len(outputs)
        return ens


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

        fan_in = _np.prod(self.inshape)
        fan_out = _np.prod(self.outshape)

        self.W_shape = (fan_in, fan_out)
        self.W = self.newweight("W_fc", self.W_shape, W)
        self.W._df_init_kw = dict(fan_in=fan_in, fan_out=fan_out)

        if bias:
            self.b_shape = (fan_out,)
            self.b = self.newbias("b_fc", self.b_shape, b)


    def make_inputs(self, name="Xin"):
        return _T.TensorType(_th.config.floatX, (False,)*(1+len(self.inshape)))(name)


    def train_expr(self, X):
        batchsize = X.shape[0]

        # For non-1D inputs, add a flattening step for convenience.
        if len(self.inshape) > 1:
            # (Don't forget the first dimension is the minibatch!)
            X = X.flatten(2)

        out = _T.dot(X, self.W)

        if hasattr(self, "b"):
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
        if self.cap is None:
            return _T.maximum(self.leak*X, X)
        else:
            return X.clip(self.leak*X, self.cap)


    def weightinitializer(self):
        """ See the documentation of `Layer`. """
        if self.init == 'Xavier':
            def init(shape, rng, fan_in, fan_out):
                fan_mean = (fan_in+fan_out)/2
                return rng.uniform(-_np.sqrt(6/fan_mean), _np.sqrt(6/fan_mean), shape)
            return init
        elif self.init == 'XavierN':
            def init(shape, rng, fan_in, fan_out):
                fan_mean = (fan_in+fan_out)/2
                return _np.sqrt(1/fan_mean)*rng.standard_normal(shape)
            return init
        elif self.init == 'PReLU':
            def init(shape, rng, fan_in, fan_out):
                fan_mean = (fan_in+fan_out)/2
                return _np.sqrt(2/fan_mean)*rng.standard_normal(shape)
            return init
        else:
            def init(shape, rng, *a, **kw):
                return self.init*rng.standard_normal(shape)
            return init


    def biasinitializer(self):
        """ See the documentation of `Layer`. """
        def init(shape, *a, **kw):
            return _np.zeros(shape)
        return init


class Tanh(Layer):
    """
    Typical tanh nonlinearity layer of oldschool nnets.
    """


    def __init__(self, init='Xavier'):
        """
        Creates a Tanh layer which computes the elementwise application of the
        tanh function:

            out = tanh(X)

        - `init`: The initialization technique to use for initializing another
                  layer's weights. Currently available techniques are:
            - `'Xavier'`: Uniform-random Xavier initialization from [1].
            - `'XavierN'`: Normal-random [2] way of achieving Xavier
                           initialization.
            - A number: Standard deviation (sigma) of the normal distribution
                        to sample from.

        1: Understanding the difficulty of training deep feedforward neural networks.
        2: Delving Deep into Rectifiers.
        """
        super(Tanh, self).__init__()
        self.init = init


    def train_expr(self, X):
        return _T.tanh(X)


    def weightinitializer(self):
        """ See the documentation of `Layer`. """
        if self.init == 'Xavier':
            def init(shape, rng, fan_in, fan_out):
                fan_mean = (fan_in+fan_out)/2
                return rng.uniform(-_np.sqrt(6/fan_mean), _np.sqrt(6/fan_mean), shape)
            return init
        elif self.init == 'XavierN':
            def init(shape, rng, fan_in, fan_out):
                fan_mean = (fan_in+fan_out)/2
                return _np.sqrt(1/fan_mean)*rng.standard_normal(shape)
            return init
        else:
            def init(shape, rng, *a, **kw):
                return self.init*rng.standard_normal(shape)
            return init


    def biasinitializer(self):
        """ See the documentation of `Layer`. """
        def init(shape, *a, **kw):
            return _np.zeros(shape)
        return init
