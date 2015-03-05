#!/usr/bin/env python3

from DeepFried.util import batched, maybetuple

import theano as _th


class StreaMiniPredictor(object):
    """
    This is a predictor that works through minibatches of the dataset, each
    minibatch optionally first being augmented, then being uploaded to the
    GPU each time.
    """


    def __init__(self, batchsize, model, X=None):
        """
        - `batchsize`: The number of samples in a minibatch.
        - `model`: The model. This should be an object with at least:
            - `make_input(name='')`: a method which returns a symbolic
                variable of the correct dimensions for serving as input.
            - `out_layers`: a list of the output layers, where each output
                layer needs to be an object with at least:
                    - `ensemble(preds)`: A method which given a list of
                        separate predictions for a minibatch computes the
                        ensembled prediction for that minibatch.
                    - `aggregate_batches(preds)`: A method which given a list
                        of predictions for each batch returns the predictions
                        for the full training set. (Mostly just concatenate.)
            - `pred_exprs(X)`: a method which returns a list of symbolic
                outputs (the "predictions") of a model for a symbolic input
                minibatch `X`. Typical models just have a single prediction.
        """
        self.model = model
        self.batchsize = batchsize

        if X is None:
            self.X = self.model.make_input()
        elif isinstance(X, str):
            self.X = self.model.make_input(X)
        else:
            self.X = X

        self.outlayers = self.model.out_layers
        self.fn_pred = _th.function(
            inputs=[self.X],
            outputs=self.model.pred_exprs(self.X),
            name="StreaMiniPredictor"
        )


    def pred_epoch(self, X, aug=None, fast=False, batchsize=None):
        """
        Predicts the model's output for a full dataset `X` by iterating
        through minibatches if necessary.

        - `X`: A numpy array containing the data. The first dimension should be
               the datapoints, i.e. X.shape[0] == ndata, and any remaining
               dimensions should fit the model's expected input shape.
        - `aug`: An optional data augmentation pipeline that can transform each
                 sample in the minibatch individually.
        - `fast`: A flag passed on to `aug` which chooses how many
            augmentations should be used. `False` is slower but usually results
            in much better predictions.
        - `batchsize`: Optionally override the batchsize given at construction.
        """
        nout = len(self.outlayers)
        preds = [[] for _ in range(nout)]  # N.B. [[]]*nout won't work.
        bs = batchsize or self.batchsize

        # Go through the training in minibatches. Note that the last batch
        # may be smaller than the batchsize.
        for bx in batched(bs, X):
            # For prediction, augmentation makes a big difference:
            if aug:
                # With augmentation, the model will be evaluated on potentially
                # many augmented versions of each batch and we need to average
                # the output class-probabilities of all those runs.
                # See "Return of the Devil in the Details" for details.
                augpreds = [[] for _ in range(nout)]  # N.B. [[]]*nout won't work.
                for bx_aug in aug.augbatch_pred(bx, fast):
                    outs = self.fn_pred(bx_aug)
                    for p, o in zip(augpreds, outs):
                        p.append(o)
                # Now ensemble each of the predictions from the augmented
                # batches into single predictions for this batch and append
                # these to `preds`.
                for p, l, ap in zip(preds, self.outlayers, augpreds):
                    p.append(l.ensemble(ap))

            else:
                # While without augmentation, it's pretty straightforward.
                outs = self.fn_pred(bx)
                for p, o in zip(preds, outs):
                    p.append(o)

        # Now collect all predictions over the minibatches.
        # Predictions may be collected differently, e.g. errors are summed
        # while scores (e.g. neg-log-likelihood) are usually averaged.
        return maybetuple(l.aggregate_batches(p) for p, l in zip(preds, self.outlayers))
