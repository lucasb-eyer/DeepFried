#!/usr/bin/env python3

from DeepFried.util import tuplize, batched, maybetuple

import theano as _th


class StreaMiniPredictor(object):
    """
    This is a predictor that works through minibatches of the dataset, each
    minibatch optionally first being augmented, then being uploaded to the
    GPU each time.
    """


    def __init__(self, batchsize, model, Xnames=[]):
        """
        - `batchsize`: The number of samples in a minibatch.
        - `model`: The model. This should be an object with at least:
            - `make_inputs(*names='')`: a method which returns a symbolic
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

        self.Xs = tuplize(self.model.make_inputs(*Xnames))

        outs = tuplize(self.model.pred_expr(*self.Xs))
        self.batch_aggs = tuplize(self.model.batch_agg())
        self.ensemblers = tuplize(self.model.ensembler())

        # A few sanity checks before compiling the function.
        assert len(outs) == len(self.batch_aggs), "The amount of outputs ({}) differs from the amount of batch aggregators ({}). You probably hit a bug, please file an issue".format(len(outs), len(self.batch_aggs))
        assert len(outs) == len(self.ensemblers), "The amount of outputs ({}) differs from the amount of ensemblers ({}). You probably hit a bug, please file an issue".format(len(outs), len(self.ensemblers))

        self.fn_pred = _th.function(
            inputs=self.Xs,
            outputs=outs,
            name="StreaMiniPredictor pred"
        )


    def pred_epoch(self, X, aug=None, fast=False, batchsize=None, **kwargs):
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

        Any remaining arguments will be passed on to the prediction function.
        """
        nout = len(self.batch_aggs)

        # A list where each entry corresponds to an output and contains
        # all the values of this output for each minibatch.
        preds = [[] for _ in range(nout)]  # N.B. [[]]*nout won't work.

        # Sanitize inputs for more flexibility
        Xs = tuplize(X)
        bs = batchsize or self.batchsize

        assert all(X.shape[0] == Xs[0].shape[0] for X in Xs), "All inputs to pred_epoch should contain the same amount of datapoints."

        # Go through the training in minibatches. Note that the last batch
        # may be smaller than the batchsize.
        for bxs in batched(bs, *Xs):
            # Possibly need to re-tuplize them because `batched` tries to be
            # smart and not return a tuple if batching a single array.
            bxs = tuplize(bxs)

            # For prediction, augmentation makes a big difference:
            if aug is not None:
                # With augmentation, the model will be evaluated on potentially
                # many augmented versions of each batch and we need to average
                # the output class-probabilities of all those runs.
                # See "Return of the Devil in the Details" for details.
                augpreds = [[] for _ in range(nout)]  # N.B. [[]]*nout won't work.

                # Here, we assume that if we have multiple inputs, the
                # augmenter also takes multiple inputs. This is because
                # augmentation in the case of multiple inputs is domain-
                # specific knowledge.
                for bxs_aug in aug.augbatch_pred(*bxs, fast=fast):
                    bxs_aug = tuplize(bxs_aug)
                    outs = self.fn_pred(*bxs_aug, **kwargs)
                    for p, o in zip(augpreds, outs):
                        p.append(o)
                # Now ensemble each of the predictions from the augmented
                # batches into single predictions for this batch and append
                # these to `preds`.
                for p, ens, ap in zip(preds, self.ensemblers, augpreds):
                    p.append(ens(ap))

            else:
                # While without augmentation, it's pretty straightforward.
                outs = self.fn_pred(*bxs, **kwargs)
                for p, o in zip(preds, outs):
                    p.append(o)

        # Now collect all predictions over the minibatches.
        # Predictions may be collected differently, e.g. errors are summed
        # while scores (e.g. neg-log-likelihood) are usually averaged.
        return maybetuple(agg(p) for p, agg in zip(preds, self.batch_aggs))
