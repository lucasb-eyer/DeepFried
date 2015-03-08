#!/usr/bin/env python3

import numpy as _np
import theano as _th
import theano.tensor as _T
import numbers

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


    def train_expr(self, *Xs, **kw):
        """
        Returns an expression or a tuple of expressions computing the
        output(s) at training-time given the symbolic input(s) in `Xs`.

        The following may be contained in `kw`:

        - `updates`: A list to which extra Theano updates which should be
            performed during training can be added.
            The layout is the same as for `theano.function`: `updates` is a
            list of pairs where the first element is the shared variable to be
            updated and the second element is the update expression.
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
        p = self._newparam(*args, **kwargs)
        self.Ws.append(p)
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
        return p


    def _newparam(self, name, shape, val=None, init=None):
        name = "{}{}".format(name, 'x'.join(map(str, shape)))

        if val is None and init is None:
            init = lambda shape, *a, **kw: _np.full(shape, _np.nan, dtype=_th.config.floatX)
        elif isinstance(val, _np.ndarray):
            init = lambda *a, **kw: val
        elif isinstance(val, numbers.Real):
            init = lambda *a, **kw: _np.full(shape, val, dtype=_th.config.floatX)
        elif isinstance(val, _T.TensorVariable):
            # When using an existing theano shared variable, don't store nay
            # initializer as "the origina" probably already has one.
            return val, None
        else:
            raise ValueError("Couldn't understand parameters for parameter creation.")

        p = _th.shared(init(shape).astype(_th.config.floatX), name=name)
        self.params.append(p)
        self.inits[p] = init
        return p


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


    # And a whole bunch of hooks that some weird layers may utilize.
    # (I'm thinking of you, BN!)


    def pre_epoch(self):
        pass


    def pre_minibatch(self):
        pass


    def post_minibatch(self):
        pass


    def post_epoch(self):
        pass


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


    def train_expr(self, X, **kw):
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


    def train_expr(self, X, **kw):
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


    def train_expr(self, X, **kw):
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


    def train_expr(self, X, **kw):
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


class Conv2D(Layer):
    """
    Your local neighborhood convolutional layer.

    If `*` is the convolution, it computes the following operation:

    out[b,k,:,:] = sum_d filt[k,d,:,:] * in[b,d,:,:] + b[k]

    `b` goes through the minibatch samples,
    `k` goes through the filterbank-kernels, and
    `d` sums across the input image dimensions.
    """

    def __init__(self, nconv, convshape, imdepth, imshape=None,
                 stride=(1,1), border_mode='valid',
                 bias=True, W=None, b=None):
        """
        Creates a 2D convolutional layer with the following properties:

        - `nconv`: The number of filters in the filterbank. This will also be
            the depth of the output image, since there's one layer per filter.
        - `convshape`: A number or a pair of numbers specifying the size (h,w)
            of the filters. If it's a single number, the filters are square.
        - `imdepth`: The number of layers the input image has. For RGB images,
            this would be three.
        - `imshape`: Optionally the shape (h,w) of the input image.
            If specified, the input will automatically be reshaped to this,
            taking `imdepth` and the batchsize into account.
            If `None`, the input could be images of any size,
            but needs to be shaped in the correct dimension, i.e. 4D.
        - `stride`: Factor by which the output is subsampled, *not* pooled.
            Note that this doesn't save computations as it's still doing the
            full convolution first. Isn't it almost pointless?
        - `border_mode`: From Theano's documentation:
            - `"valid"`: only apply filter to complete patches of the image.
                Output shape: image_shape - filter_shape + 1
            - `"full"`: zero-pads image to multiple of filter shape.
                Output shape: image_shape + filter_shape - 1
        - `bias`: Whether to use a bias term or not.
                  For example, if a minibatch-normalization follows,
                  a bias term is useless.
        - `W`: Optional initial value for the weights.
        - `b`: Optional initial value for the bias.
        """
        super(Conv2D, self).__init__()

        # Allow for specifying the conv shape as a single number if square.
        if isinstance(convshape, numbers.Integral):
            convshape = (convshape, convshape)

        self.border_mode = border_mode
        self.stride = stride
        self.imshape = imshape
        self.imdepth = imdepth

        fan_in = imdepth * _np.prod(convshape)
        fan_out = nconv * _np.prod(convshape)

        self.W_shape = (nconv, imdepth) + convshape
        self.W = self.newweight("W_conv", self.W_shape, W)
        self.W._df_init_kw = dict(fan_in=fan_in, fan_out=fan_out)

        if bias:
            self.b_shape = (nconv,)
            self.b = self.newbias("b_fc", self.b_shape, b)


    def make_inputs(self, name="Xin"):
        if self.imshape is None:
            return _T.tensor4(name)
        else:
            return _T.matrix(name)


    def train_expr(self, X, **kw):
        if self.imshape is not None:
            X = X.reshape((X.shape[0], self.imdepth) + self.imshape)

        out = _T.nnet.conv.conv2d(X, self.W,
            image_shape=(None, self.imdepth) + (self.imshape or (None, None)),
            filter_shape=self.W_shape,
            border_mode=self.border_mode,
            subsample=self.stride
        )

        if hasattr(self, "b"):
            out += self.b.dimshuffle('x', 0, 'x', 'x')

        return out


class SpatialMaxPool(Layer):
    """
    Your local neighborhood's pool. Quite full during summer.
    """

    def __init__(self, size, stride=None, ignore_border=False):
        """
        Creates a 2D max-pooling layer which pools over the 2 last dimensions.

        - `size`: The size (h,w) of the pooled patches.
        - `stride`: Not supported yet, will be in newer Theano version.
        - `ignore_border`: whether to make use of left-over border when sizes
            are not perfectly divisible, or just ignore it.
        """
        super(SpatialMaxPool, self).__init__()

        if isinstance(size, numbers.Integral):
            size = (size, size)
        self.size = size

        if isinstance(stride, numbers.Integral):
            stride = (stride, stride)
        self.stride = stride

        self.ignore_border = ignore_border


    def make_inputs(self, name="Xin"):
        # Actually, this could be anything > 2D.
        return _T.tensor4(name)


    def train_expr(self, X, **kw):
        return _T.signal.downsample.max_pool_2d(X, ds=self.size, ignore_border=self.ignore_border)
