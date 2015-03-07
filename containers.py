#!/usr/bin/env python3

from DeepFried.layers import Layer
from DeepFried.util import collect, tuplize


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


    def append(self, layer, init_previous=1, initializes=None):
        """
        Appends a `layer` onto the sequence, making the output of the previous
        one be the input of this one. Additionally, `layer` will be used as
        initializer for the `init_previous` layers preceding it as well as for
        all the layers given in `initializes`.

        - `layer`: The `Layer` to append.
        - `init_previous`: how many preceding weight layers this one inits.
        - `initializes`: a `Layer` or list of `Layer`s which this one inits.
        """

        self.layers.append(layer)
        self.Ws += layer.Ws
        self.bs += layer.bs
        self.params += layer.params

        def registerinit(l):
            for W in l.Ws:
                l.inits[W] = layer.weightinitializer()
            for b in l.bs:
                l.inits[b] = layer.biasinitializer()

        # If it's explicitly stated which other layers this one should
        # initialize, don't bother searching and also don't bother checking
        # whether this one actually can initialize.
        if isinstance(initializes, Layer):
            pass
        elif initializes is not None:
            for l in initializes:
                registerinit(l)

        # Initialize the previous `init_previous` layers otherwise, but only
        # if the layer actually can initialize.
        if hasattr(layer, "weightinitializer") and hasattr(layer, "biasinitializer"):
            for i in range(2, init_previous+2):
                registerinit(self.layers[-i])


    def make_inputs(self, *names):
        return self.layers[0].make_inputs(*names)


    def train_expr(self, *Xs):
        """
        Concatenates the training expression of all layers one to another
        and returns the training expression(s) of the last layer.
        """
        for l in self.layers:
            Xs = l.train_expr(*tuplize(Xs))
        return Xs


    def pred_expr(self, *Xs):
        """
        Same as `train_expr` but for prediction expressions.
        """
        for l in self.layers:
            Xs = l.pred_expr(*tuplize(Xs))
        return Xs


    def reinit(self, rng):
        """
        Re-initializes all contained layers. See `Layer` documentation for more.
        """
        for l in self.layers:
            l.reinit(rng)


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
