#!/usr/bin/env python3

from DeepFried.layers import Layer


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
        # Always use the last layer in the sequence as output layer.
        self.out_layers = [layer]

        self.layers.append(layer)
        self.Ws += layer.Ws
        self.bs += layer.bs
        self.params += layer.params

        def registerinit(l):
            print("{} initializes {}".format(layer, l))
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


    def make_input(self, *a, **kw):
        return self.layers[0].make_input(*a, **kw)


    def train_expr(self, X, *a, **kw):
        for l in self.layers:
            X = l.train_expr(X, *a, **kw)
        return X


    def pred_expr(self, X, *a, **kw):
        for l in self.layers:
            X = l.pred_expr(X, *a, **kw)
        return X


    def reinit(self, rng):
        """
        Re-initializes all contained layers. See `Layer` documentation for more.
        """
        for l in self.layers:
            l.reinit(rng)

