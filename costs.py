#!/usr/bin/env python3

import numpy as _np
import theano.tensor as _T


class CategoricalCrossEntropy(object):
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
        if loclip is None and hiclip is None:
            self.clip = None
        else:
            self.clip = (0.0 if loclip is None else loclip,
                         1.0 if hiclip is None else hiclip)


    def cost_expr(self, p_y_given_x, y):
        """
        Creates an expression computing the mean of the negative log-likelihoods
        of the predictions `p_y_given_x` wrt. the data `y`:

            -log p(y|x,params)

        - `p_y_given_x`: A 2d-tensor whose first dimension is the samples and
                         second dimension is the class probabilities of that
                         sample.
        - `y`: A one-hot encoded vector of labels whose length should be that
               of `p_y_given_x`'s first dimension.

        TODO: For now this doesn't actually make use of the clipping since
              that'd potentially break Theano's optimization of merging this
              with the softmax output and numerically stabilizing them.
              If I'll ever run into the issue of getting nans here, I'll add
              the clipping here too.
        """
        # TODO: Wait for https://github.com/Theano/Theano/issues/2464
        return -_T.mean(_T.log(p_y_given_x[_T.arange(y.shape[0]), y]))


    def cost(self, p_y_given_x, y):
        """
        Computes the cost immediately on numpy arrays instead of building an
        expression.

        This is currently necessary by design for augmented prediction.
        """
        py = p_y_given_x[_np.arange(y.shape[0]), y]
        if self.clip is not None:
            py = _np.clip(py, *self.clip)
        return -_np.mean(_np.log(py))


    def aggregate_batches(self, batchcosts):
        """
        Function used to aggregate the cost of multiple minibatches into an
        estimate of the full model's cost over the full dataset.

        In this case, we aggregate by taking the mean.
        """
        return sum(batchcosts)/len(batchcosts)


    def make_target(self, name="t"):
        """
        Return a theano variable of just the right type for holding the targets.

        In this case, that means a vector of int32, each entry being the target
        class of the corresponding sample in the minibatch.

        - `name`: Name to give the variable, defaults to "t".
        """
        return _T.ivector(name)
