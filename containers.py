#!/usr/bin/env python3

from DeepFried.layers import Layer
from DeepFried.util import collect, tuplize


def _mkproxy(self, fn):
    def proxy(*a, **kw):
        for l in self.layers:
            getattr(l, fn)(*a, **kw)
    return proxy


class Sequence(Layer):

    def __init__(self, *layers):
        """
        Constructs a layer composed of all passed `layers` stacked on top of
        eachother, as is typical for most feed-forward neural networks.

        This also implicitly uses each layer as an initializer for the previous
        one's weights. This means if you have multiple weight-layers following
        eachother without any initializing layer in-between, you'll have to
        resort to the `append` method.
        """
        super(Sequence, self).__init__()

        self.layers = []

        # These collect all contained layers' params
        self.Ws = []
        self.bs = []
        self.params = []

        for l in layers:
            self.append(l)

        # And forward a whole bunch of functions.
        self.reinit =_mkproxy(self, 'reinit')
        self.pre_epoch =_mkproxy(self, 'pre_epoch')
        self.pre_minibatch =_mkproxy(self, 'pre_minibatch')
        self.post_minibatch =_mkproxy(self, 'post_minibatch')
        self.post_epoch =_mkproxy(self, 'post_epoch')


    def append(self, layer, init_previous=True, initializes=None):
        """
        Appends a `layer` onto the sequence, making the output of the previous
        one be the input of this one. Additionally, `layer` will be used as
        initializer for the `init_previous` layers preceding it as well as for
        all the layers given in `initializes`.

        - `layer`: The `Layer` to append.
        - `init_previous`:
            - If `True`, initialize *all* of the preceding layers up to the
                next one which has an initializer method.
            - If a number n, unconditionally initialize the previons n layers.
                Note that passing `False` equals passing 0.
        - `initializes`: a `Layer` or list of `Layer`s which this one inits.

        Note that a layer will always try to initialize itself too.

        Returns `layer` to allow one-liners.
        """

        self.layers.append(layer)
        self.Ws += layer.Ws
        self.bs += layer.bs
        self.params += layer.params

        def registerinit(l):
            if hasattr(layer, "weightinitializer"):
                #print(layer, "inits W of", l)
                for W in l.Ws:
                    l.inits[W] = layer.weightinitializer()
            if hasattr(layer, "biasinitializer"):
                #print(layer, "inits b of", l)
                for b in l.bs:
                    l.inits[b] = layer.biasinitializer()

        # If it's explicitly stated which other layers this one should init,
        # just do it: trust the user!
        for l in tuplize(initializes, tuplize_none=True):
            registerinit(l)

        # Initialize all previous layers up to the next initializer.
        if init_previous is True:
            registerinit(layer)
            for l in reversed(self.layers[:-1]):
                if hasattr(l, "weightinitializer") or hasattr(l, "biasinitializer"):
                    break
                registerinit(l)
        # Unconditionally initialize all previous layers.
        else:
            # Also covers False and 0 thanks to the -1
            # and layer being already there.
            for l in self.layers[-init_previous-1:]:
                registerinit(l)

        return layer


    def make_inputs(self, *names):
        return self.layers[0].make_inputs(*names)


    def train_expr(self, *Xs, **kw):
        """
        Concatenates the training expression of all layers one to another
        and returns the training expression(s) of the last layer.
        """
        for l in self.layers:
            Xs = l.train_expr(*tuplize(Xs), **kw)
        return Xs


    def pred_expr(self, *Xs):
        """
        Same as `train_expr` but for prediction expressions.
        """
        for l in self.layers:
            Xs = l.pred_expr(*tuplize(Xs))
        return Xs


    def ensembler(self):
        """
        Uses the ensembler of the last (i.e. output) layer of the stack.
        """
        return self.layers[-1].ensembler()


    def batch_agg(self):
        """
        Uses the batch aggregator of the last (i.e. output) layer of the stack.
        """
        return self.layers[-1].batch_agg()


class Parallel(Layer):

    def __init__(self, *layers):
        """
        Constructs a layer composed of all passed `layers` using this layer's
        input (i.e. all the same input) but each having their own output.

        This is what you could use, for a multi-output at the top of a network,
        for instance.

        **NOTE** that this doesn't do any initialization wiring!
        """
        super(Parallel, self).__init__()

        self.layers = []

        # These collect all contained layers' params
        self.Ws = []
        self.bs = []
        self.params = []

        for l in layers:
            self.append(l)

        # And forward a whole bunch of functions.
        self.reinit =_mkproxy(self, 'reinit')
        self.pre_epoch =_mkproxy(self, 'pre_epoch')
        self.pre_minibatch =_mkproxy(self, 'pre_minibatch')
        self.post_minibatch =_mkproxy(self, 'post_minibatch')
        self.post_epoch =_mkproxy(self, 'post_epoch')


    def append(self, layer):
        """
        Appends a `layer` onto the stack, giving it the same input as all other
        layers get and adding its output to this layer's outputs.

        Returns `layer` to allow one-liners.
        """
        self.layers.append(layer)
        self.Ws += layer.Ws
        self.bs += layer.bs
        self.params += layer.params
        return layer


    def make_inputs(self, *names):
        """
        Since all contained layers will use the exact same input, this uses the
        first contained layer to create inputs, though it does check whether
        they'd all create similar inputs.
        """
        ins = tuplize(self.layers[0].make_inputs(*names))

        # A crude way of making sure all layers take the same input.
        for l in self.layers[1:]:
            lins = tuplize(l.make_inputs(*names))
            assert len(ins) == len(lins), "All parallel layers must require the same number of inputs!\n{} requires {}, but {} requires {}".format(self.out_layers[0], len(ins), l, len(lins))
            assert all(a.ndim == b.ndim for a, b in zip(ins, lins)), "All parallel layers inputs must have the same dimension!\nConflict between {} and {}".format(self.out_layers[0], l)

        return ins


    def train_expr(self, X, **kw):
        """
        Returns the training expressions of all contained layers, since each
        contained layer contributes to an output.
        """
        return collect(l.train_expr(X, **kw) for l in self.layers)


    def pred_expr(self, X):
        """
        Returns the prediction expressions of all contained layers, since each
        contained layer contributes to an output.
        """
        return collect(l.pred_expr(X) for l in self.layers)


    def ensembler(self):
        """
        Returns the ensemblers of all contained layers, since each contained
        layer contributes to an output.
        """
        return collect(l.ensembler() for l in self.layers)


    def batch_agg(self):
        """
        Returns the batch aggregators of all contained layers, since each
        contained layer contributes to an output.
        """
        return collect(l.batch_agg() for l in self.layers)
