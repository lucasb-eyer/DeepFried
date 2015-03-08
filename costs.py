#!/usr/bin/env python3

from DeepFried.util import collect, tuplize

import numpy as _np
import theano.tensor as _T
from itertools import islice as _islice


class Cost(object):
    """
    Abstract superclass of all costs with a dual purpose:

    1. Define common interface all layers should implement.
    2. Implement boilerplate code which is the same for any kind of layer.

    Note that only the second point is actually a valid one; introducing a
    hierarchy of classes for the first one would be unpythonic.
    """


    def single_out_expr(self, model, y, t):
        """
        Returns a theano expression computing the cost for given `model`'s
        output `y` wrt. targets `t`.

        Implement this one if your cost works on a single output and a single
        target.
        """
        raise NotImplementedError("{} needs to either implement the single-output-single-target `single_out_expr` cost function, or the multi-output-multi-target `out_expr` cost function.".format(type(self).__name__))


    def out_expr(self, model, outputs, targets):
        """
        Returns a theano expression computing the cost for a given `model`'s
        `outputs` wrt. the `targets`.

        In the common case where there's just a single output and target,
        this will delegate to `single_out_expr`.
        """
        assert len(outputs) == 1, "{} can only handle a single output".format(type(self).__name__)
        assert len(targets) == 1, "{} can only handle a single target".format(type(self).__name__)

        return self.single_out_expr(model, outputs[0], targets[0])


    def make_target(self, *names):
        """
        Returns a Theano tensor or a tuple of Theano tensors with given `names`
        of the dimensions which this layer takes as targets.
        """
        raise NotImplementedError("{} needs to implement the `make_target` method.".format(type(self).__name__))


    def aggregate_batches(self, batchcosts):
        """
        Function used to aggregate the cost of multiple minibatches into an
        estimate of the full model's cost over the full dataset.

        By default, we aggregate by taking the mean.
        """
        return sum(batchcosts)/len(batchcosts)


class MultiCost(Cost):
    """
    This cost is a weighted linear combination of other costs. It comes in
    handy when the net has multiple outputs which should map to multiple
    targets.
    """

    def __init__(self, *args):
        super(MultiCost, self).__init__()

        self.costs = []
        self.weights = []

        for a in args:
            if isinstance(a, tuple):
                self.add(a[1], a[0])
            else:
                self.add(a)


    def add(self, cost, weight=1.0):
        self.costs.append(cost)
        self.weights.append(weight)


    def make_target(self, *names):
        if len(names) == 0:
            return collect(c.make_target() for c in self.costs)
        else:
            # TODO: How to distribute the names to the costs otherwise!?
            assert len(names) == len(self.costs), "For now, there can only be one explicitly named target per single cost in `{}`. Please file an issue with your use-case.".format(type(self).__name__)
            return collect(c.make_target(n) for c, n in zip(self.costs, names))


    def out_expr(self, model, outputs, targets):
        assert len(outputs) == len(targets), "Currently, there can only be exactly one output per target `{}`".format(type(self).__name__)

        outs = iter(outputs)
        tgts = iter(targets)

        tot = 0
        for w, c in zip(self.weights, self.costs):
            # Nasty trick to figure out how many targets this cost eats and only
            # eat that many. This allows for zero-target costs such as weight
            # decays to be added into the mix.
            n = len(tuplize(c.make_target(), tuplize_none=True))
            tot += w*c.out_expr(model, list(_islice(outs, n)), list(_islice(tgts, n)))

        return tot


class L2WeightRegCost(Cost):
    """
    Add this cost into the mix (e.g. using `MultiCost`) in order to add an L2
    weight-regularization term, also know as weight-decay.
    """


    def out_expr(self, model, outputs, targets):
        return sum((W**2).sum() for W in model.Ws)


    def make_target(self, *names):
        return None


class L1WeightRegCost(Cost):
    """
    Add this cost into the mix (e.g. using `MultiCost`) in order to add an L1
    weight-regularization term.
    """


    def out_expr(self, model, outputs, targets):
        return sum(abs(W).sum() for W in model.Ws)


    def make_target(self, *names):
        return None


class CategoricalCrossEntropy(Cost):
    """
    This is the typical cost used together with a softmax output layer.
    """

    def __init__(self, loclip=None, hiclip=None):
        """
        - `loclip`: Clips the predicted probability at this lower value in
                    order to avoid computing log(0).
        - `hiclip`: Clips the predicted probability at this upper value,
                    which doesn't really make sense to me but some criteria
                    require this.
        """
        super(CategoricalCrossEntropy, self).__init__()

        if loclip is None and hiclip is None:
            self.clip = None
        else:
            self.clip = (0.0 if loclip is None else loclip,
                         1.0 if hiclip is None else hiclip)


    def single_out_expr(self, model, p_t_given_x, t):
        """
        Creates an expression computing the mean of the negative log-likelihoods
        of the predictions `p_t_given_x` wrt. the targets `t`:

            -log p(t|x,params)

        - `p_t_given_x`: A 2d-tensor whose first dimension is the samples and
                         second dimension is the class probabilities of that
                         sample.
        - `t`: A one-hot encoded vector of labels whose length should be that
               of `p_t_given_x`'s first dimension.

        TODO: For now this doesn't actually make use of the clipping since
              that'd potentially break Theano's optimization of merging this
              with the softmax output and numerically stabilizing them.
              If I'll ever run into the issue of getting nans here, I'll add
              the clipping here too.
        """
        # TODO: Wait for https://github.com/Theano/Theano/issues/2464
        return -_T.mean(_T.log(p_t_given_x[_T.arange(t.shape[0]), t]))


    def np_cost(self, p_t_given_x, t):
        """
        Computes the cost immediately on numpy arrays instead of building an
        expression.

        This is currently necessary by design for augmented prediction.
        """
        pt = p_t_given_x[_np.arange(t.shape[0]), t]
        if self.clip is not None:
            pt = _np.clip(pt, *self.clip)
        return -_np.mean(_np.log(pt))


    def make_target(self, name="t"):
        """
        Return a theano variable of just the right type for holding the target.

        In this case, that means a vector of int32, each entry being the
        target class of the corresponding sample in the minibatch.

        - `name`: Name to give the variable, defaults to "t".
        """
        return _T.ivector(name)
