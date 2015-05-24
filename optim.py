#!/usr/bin/env python3

import DeepFried.util as _u

import numpy as _np
import theano as _th
import theano.tensor as _T


class StreaMiniOptimizer(object):
    """
    This is an optimizer that works through minibatches of the dataset, each
    minibatch being uploaded onto the GPU each time.

    This is slower than moving the whole dataset on the GPU once and addressing
    each slices of it, but it allows for larger datasets to fit on the GPU as
    well as "infinite" data augmentation.
    """


    def __init__(self, batchsize, model, cost, extra_outs=None, Xnames=[], tnames=[]):
        """
        Initializes the things that are common amongst all streaming minibatch
        optimizers.

        - `batchsize`: The number of samples in a minibatch.
        - `model`: The model. This should be an object with at least:
            - `make_inputs(basename='X')`: a function which returns a list of
                as many symbolic variables of the correct dimensions as the
                model takes as inputs. That's usually just one.
            - `train_exprs(*Xs)`: a function which returns the symbolic
                output(s) of the model, during training, given symbolic model
                input(s) `X`.
            - `params`: an iterable containing all trainable parameters.
        - `cost`: The cost. This should be an object with at least:
            - `make_target(name='')`: a function which returns a symbolic
                variable of the correct dimensions for serving as target.
            - `out_expr(Y, t)`: a function which returns the symbolic cost
                of the output `Y` wrt. the targets `t`.
            - `aggregate_batches(costs)`: a function which returns the
                aggregation of the `costs` of each minibatch.
        - `extra_outs`: A single or a list of extra outputs to compute along
            the way. Each such extra should be an object with both `out_expr`
            and `aggregate_batches` just like described for `cost` above.
        - `Xnames`: Optional list of names to use for input variables. Note
            that this must be exactly as many names as the model has inputs,
            then these names may be used as keyword arguments to `fit_epoch`.
        - `tnames`: The same as `Xnames`, but for target variables.
        """
        self.model = model
        self.cost = cost
        self.batchsize = batchsize

        self.Xs = _u.tuplize(self.model.make_inputs(*Xnames))
        self.targets = _u.tuplize(self.cost.make_target(*tnames))
        self.xtras = _u.tuplize(extra_outs, tuplize_none=True)

        # These two will collect any additional updates that layers may have,
        # for example batch-normalization's statistics collection.
        self.fwd_updates = []
        self.fin_updates = []

        train_expr = _u.tuplize(self.model.train_expr(*self.Xs, fwd_updates=self.fwd_updates, fin_updates=self.fin_updates))
        self.cost_expr = self.cost.out_expr(self.model, train_expr, self.targets)
        self.outs = (self.cost_expr,) + tuple(
            x.out_expr(self.model, train_expr, self.targets) for x in self.xtras
        )


    def _mk_train_fn(self, name, updates, extra_in=None, extra_out=None):
        """ To be used by specializations only. """
        self.fn_train = _th.function(
            inputs=self.Xs + self.targets + _u.tuplize(extra_in, tuplize_none=True),
            outputs=self.outs + _u.tuplize(extra_out, tuplize_none=True),
            updates=updates + self.fwd_updates,
            name=name
        )

        if len(self.fin_updates):
            # Because targets might or might not be used by the layers in the
            # extra update rules, we'll just allow for unused inputs.
            self.fn_finalize = _th.function(
                inputs=self.Xs + self.targets,
                updates=self.fin_updates,
                name=name + " finalize",
                on_unused_input='ignore'
            )


    def reinit(self):
        """
        This will reinitialize any state (such as momentum) that may be kept by
        this optimizer.
        """
        pass


    def fit_epoch(self, X, t, aug=None, batchsize=None, shuf=False, **kwargs):
        """
        Trains the model for one full epoch by iterating through minibatches.

        - `X`: A numpy array or a list of numpy arrays containing the model input(s).
            The first dimension of an input should be the datapoints,
            i.e. X.shape[0] == ndata,
            and any remaining dimensions should fit the model's expected input shape(s).
        - `t`: The target values where the first dimension should be the
               datapoints, just like for `X`.
        - `aug`: An optional data augmentation pipeline that can transform each
                 sample in the minibatch individually.
        - `batchsize`: Optionally override the batchsize given at construction.
        - `shuf`: If not False, go through `X` and `t` in lockstep-random order.
                  Use `shuf` as rng or seed for the shuffling.

        Any remaining arguments will be passed on to the optimization function;
        this can be used to pass values such as learning-rate, momentum etc.
        """
        self.model.pre_epoch()

        costs = []
        xtras = []

        # Sanitize inputs for more flexibility.
        Xs = _u.tuplize(X)
        ts = _u.tuplize(t)
        bs = batchsize or self.batchsize
        N = Xs[0].shape[0]

        assert all(X.shape[0] == N for X in Xs), "All inputs to fit_epoch should contain the same amount of datapoints."
        assert all(t.shape[0] == N for t in ts), "All targets to fit_epoch should contain the same amount of datapoints."

        # Keyword arguments for `batched`, for conciseness.
        if shuf is False:
            bxkw = btkw = {}
        else:
            common_seed = _u.check_random_state(shuf).randint(2**31)
            bxkw = dict(shuf=_np.random.RandomState(common_seed))
            btkw = dict(shuf=_np.random.RandomState(common_seed))

        # Go through the training in minibatches. Note that the last batch
        # may be smaller than the batchsize.
        for bxs, bts in zip(_u.batched(bs, *Xs, **bxkw), _u.batched(bs, *ts, **btkw)):
            # Possibly need to re-tuplize them because `batched` tries to be
            # smart and not return a tuple if batching a single array.
            bxs = _u.tuplize(bxs)
            bts = _u.tuplize(bts)

            # Potentially generate a new augmentation on-the-fly.
            if aug is not None:
                assert len(bxs) == 1, "Augmentation with multiple inputs not implemented yet. Please open an issue describing the use-case!"
                bx, bts = aug.augbatch_train(bxs[0], *bts)
                bxs = (bx,)

            self.model.pre_minibatch()

            # Uploads to the GPU, does the forward pass,
            # the backward pass *and* the weight updates!
            cost, *xtra = self.fn_train(*bxs+bts, **kwargs)

            # Collect stats over the batches, so we can aggregate.
            costs.append(cost)
            xtras.append(xtra)

            self.model.post_minibatch()

        self.model.post_epoch()

        # Average the stats over the batches.
        return _u.maybetuple((self.cost.aggregate_batches(costs),)
                        + tuple(x.aggregate_batches(b) for x, b in zip(self.xtras, zip(*xtras))))
        # The above zip transposes from minibatches of extras to extras of minibatches.


    def finalize(self, X, t, batchsize=None, aug=None, fast=False, **kwargs):
        """
        A forward-pass through the training data, but using only the
        `fin_updates` of layers such as batch-normalization.

        The call is just like that of `fit_epoch`, but a few parameters as well
        as most comments have been omitted.
        """
        # Early-exit if unnecessary.
        if len(self.fin_updates) == 0:
            return

        bs = batchsize or self.batchsize

        # Ignore that one.
        kwargs.pop('shuf', None)

        self.model.pre_finalize()
        for bxs, bts in zip(_u.batched(bs, *_u.tuplize(X)), _u.batched(bs, *_u.tuplize(t))):
            if aug is not None:
                for bxs_aug in aug.augbatch_pred(*_u.tuplize(bxs), fast=fast):
                    self.model.finalize_pre_minibatch()
                    self.fn_finalize(*_u.tuplize(bxs_aug)+_u.tuplize(bts), **kwargs)
                    self.model.finalize_post_minibatch()
            else:
                self.model.finalize_pre_minibatch()
                self.fn_finalize(*_u.tuplize(bxs)+_u.tuplize(bts), **kwargs)
                self.model.finalize_post_minibatch()
        self.model.post_finalize()


class StreaMiniSGD(StreaMiniOptimizer):
    """
    Vanilla Stochastic Gradient Descent on minibatches. The training is quite
    simple:

        p_{e+1} = p_e - lr * ∇p_e

    Additional parameters added to `fit_epoch`:

    - `lrate`: The learning-rate.
    """

    def __init__(self, batchsize, model, cost, *args, **kwargs):
        """
        See `StreaMiniOptimizer` for details on the arguments.
        """
        super(StreaMiniSGD, self).__init__(batchsize, model, cost, *args, **kwargs)

        self.sh_learningrate = _T.scalar('lrate')

        g = _T.grad(cost=self.cost_expr, wrt=self.model.params)

        self._mk_train_fn("StreaMiniSGD train",
            [(p, p - self.sh_learningrate * gp) for p, gp in zip(self.model.params, g)],
            extra_in=self.sh_learningrate)


class StreaMiniMomentum(StreaMiniOptimizer):
    """
    TL;DR: Nesterov allows for larger momentum to be used, making it better.
           Very finicky parameter-selection.

    Implements both the "Classical Momentum (CM)" and "Nesterov's
    Accelerated Gradient (NAG)" which are explained in further detail in

    "On the importance of initialization and momentum in deep learning"

    But the equation for NAG has been reshuffled by Nicolas Boulanger in

    https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617

    for easier implementation in Theano. The updates are:

        v_{e+1} = mom * v_e - lr * ∇p_e
        p_{e+1} = p_e + v_{e+1}

    for CM, and

        p_{e+1} = p_e + mom * v_{e+1} - lr * ∇p_e

    for Nicolas' reformulated NAG.

    Additional parameters added to `fit_epoch`:

    - `lrate`: The learning-rate.
    - `momentum`: The momentum, defaulting to the one passed at construction.
    """

    def __init__(self, batchsize, model, cost, momentum, nesterov=False, *args, **kwargs):
        """
        See `StreaMiniOptimizer` for details on the arguments.

        - `momentum`: The amount of momentum to use, typically something around
            0.9, 0.95 or 0.99. This value sets the default, but it can also
            be overridden in each individual call to `fit_epoch`.
        - `nesterov`: If `True`, Nesterov's momentum (NAG) is used instead
            of classical momentum (CM).
        """
        super(StreaMiniMomentum, self).__init__(batchsize, model, cost, *args, **kwargs)

        self.sh_learningrate = _T.scalar('lrate')
        self.sh_momentum = _T.scalar('momentum')

        # For momentum, we need a "mirror" of each parameter, which keeps track
        # of the "velocity" of that parameter during training.
        self.sh_v = [
            _th.shared(_np.zeros_like(p.get_value()), broadcastable=p.broadcastable, name='v_'+p.name)
            for p in model.params
        ]

        g = _T.grad(cost=self.cost_expr, wrt=self.model.params)

        updates = []
        for sh_p, gp, sh_v in zip(self.model.params, g, self.sh_v):
            v = self.sh_momentum * sh_v - self.sh_learningrate * gp
            updates.append((sh_v, v))

            if not nesterov:
                updates.append((sh_p, sh_p + v))
            else:
                updates.append((sh_p, sh_p + self.sh_momentum * v - self.sh_learningrate * gp))

        self._mk_train_fn("StreaMiniMomentum train",
            updates,
            extra_in=(self.sh_learningrate, _th.Param(self.sh_momentum, momentum))
        )


    def reinit(self):
        for sh_v in self.sh_v:
            sh_v.set_value(_np.zeros_like(sh_v.get_value()))


class StreaMiniAdaGrad(StreaMiniOptimizer):
    """
    Implements Duchi's "Adaptive Subgradient" method, aka AdaGrad.
    Chris Dyer's "Notes on AdaGrad" are pretty awesome for practical purposes.

    TL;DR: AdaGrad doesn't need additional parameters (a lie) and makes the
           optimization much less sensitive to the learning-rate!

    The updates are:

        g²_{e+1} = g²_e + ∇(p_e)²
        p_{e+1} = p_e - (lr / √g²_{e+1}) * ∇p_e

    that is, divide the learning-rate by a running square of the gradients.

    Note that this would lead to division by 0 in the beginning for those
    weights which don't receive a gradient (might be many with ReLUs), so we
    initialize g² with a small value.

    Additional parameters added to `fit_epoch`:

    - `lrate`: The learning-rate.
    """

    def __init__(self, batchsize, model, cost, eps=1e-5, *args, **kwargs):
        """
        See `StreaMiniOptimizer` for details on the arguments.

        - `eps`: A regularization-factor, should be smaller than the
            square of the weight gradients.
        """
        super(StreaMiniAdaGrad, self).__init__(batchsize, model, cost, *args, **kwargs)

        self.sh_learningrate = _T.scalar('lrate')

        # Adagrad needs to accumulate the square gradient of each parameter.
        # I wonder if this won't explode at some point? Probably should fully
        # read the original paper!
        # Edit: RMSProp fixes exactly that.
        # Edit: Matt Zeiler seems to agree cf. AdaDelta.
        self.eps = eps
        self.sh_g2 = [
            _th.shared(_np.full_like(p.get_value(), eps), broadcastable=p.broadcastable, name='g2_'+p.name)
            for p in model.params
        ]

        g = _T.grad(cost=self.cost_expr, wrt=self.model.params)

        updates = []
        for sh_p, gp, sh_g2 in zip(self.model.params, g, self.sh_g2):
            g2 = sh_g2 + gp*gp
            updates.append((sh_g2, g2))
            updates.append((sh_p, sh_p - self.sh_learningrate/_T.sqrt(g2) * gp))
            # Instead of adding eps inside the square-root like most
            # implementations do, I just initialize `g2` to eps, that should
            # have the same effect, but cheaper.

        self._mk_train_fn("StreaMiniAdaGrad train",
            updates,
            extra_in=self.sh_learningrate
        )


    def reinit(self):
        for sh_g2 in self.sh_g2:
            sh_g2.set_value(_np.full_like(sh_g2.get_value(), self.eps))


class StreaMiniRMSProp(StreaMiniOptimizer):
    """
    Implements Hinton's "RMSProp" method presented in his Coursera lecture 6.5.
    Essentially, it sits right in-between AdaGrad and AdaDelta by being a
    windowed version of AdaGrad.

    The updates are:

        g²_{e+1} = ρ * g²_e + (1-ρ) * ∇p_e²
        p_{e+1} = p_e - (lr / √g²_{e+1}) * ∇p_e

    Note that in this case just initializing with epsilon is not enough anymore
    as we could get zero-gradient for some units long enough as to completely
    dominate the window.

    Additional parameters added to `fit_epoch`:

    - `lrate`: The learning-rate.
    - `rho`: The momentum for square-gradient accumulation, defaulting to the
        one passed at construction.
    """

    def __init__(self, batchsize, model, cost, rho=0.95, eps=1e-5, *args, **kwargs):
        """
        See `StreaMiniOptimizer` for details on the arguments.

        - `rho`: The "momentum" to use for averaging past gradients.
        - `eps`: A regularization-factor, should be smaller than the
            square of the weight gradients.
        """
        super(StreaMiniRMSProp, self).__init__(batchsize, model, cost, *args, **kwargs)

        self.sh_learningrate = _T.scalar('lrate')
        self.sh_rho = _T.scalar('rho')

        # This too needs to accumulate the square gradient of each parameter.
        self.sh_g2 = [
            _th.shared(_np.zeros_like(p.get_value()), broadcastable=p.broadcastable, name='g2_'+p.name)
            for p in model.params
        ]

        g = _T.grad(cost=self.cost_expr, wrt=self.model.params)

        updates = []
        for sh_p, gp, sh_g2 in zip(self.model.params, g, self.sh_g2):
            g2 = self.sh_rho*sh_g2 + (1-self.sh_rho)*gp*gp
            updates.append((sh_g2, g2))
            updates.append((sh_p, sh_p - self.sh_learningrate/_T.sqrt(eps+g2) * gp))

        self._mk_train_fn("StreaMiniRMSProp train",
            updates,
            extra_in=(self.sh_learningrate, _th.Param(self.sh_rho, rho))
        )


    def reinit(self):
        for sh_g2 in self.sh_g2:
            sh_g2.set_value(_np.zeros_like(sh_g2.get_value()))


class StreaMiniAdaDelta(StreaMiniOptimizer):
    """
    Implements Matt Zeiler's "Adaptive Learningrate" method, aka. AdaDelta.
    The paper itself is really neat, and both very convincing and practical.

    TL;DR: 1. AdaGrad quickly anneals, AdaDelta doesn't. (No proof.)
           2. AdaGrad *is* sensitive to learning-rate, AdaGrad not so much. (Table 1.)
           3. AdaGrad includes 2nd-order approximation. (3.2)

    The updates are:

        g²_{e+1} = ρ * g²_e + (1-ρ) * ∇p_e²
        up_{e+1} = √(d²_e / g²_{e+1}) * ∇p_e
        d²_{e+1} = ρ * d²_e + (1-ρ) * up²
        p_{e+1} = p_e - up_{e+1}

    As in RMSProp, we need to add epsilons in order to create stability.

    It turns out that the effective learning-rate will converge to 1 as the
    gradients decrease (and thus learning grinds to a halt). This could be used
    to check for convergence by a specialized trainer.
    """

    def __init__(self, batchsize, model, cost, rho=0.95, eps=1e-5, *args, **kwargs):
        """
        See `StreaMiniOptimizer` for details on the arguments.

        - `rho`: The "momentum decay" of AdaDelta. The paper tests three values
            on MNIST: 0.9, 0.95 and 0.99, they don't change the score much.
            The paper also uses the same values for a speech task.
        - `eps`: A regularization term only used to avoid singularities. The
            paper tests four values on MNIST: 1e-2, 1e-4, 1e-6, 1e-8;
            all of them work pretty well.
        """
        super(StreaMiniAdaDelta, self).__init__(batchsize, model, cost, *args, **kwargs)

        self.sh_rho = _T.scalar('rho')

        # Similarly to Adagrad, AdaDelta accumulates the square gradient of
        # each parameter, it just exponentially decays the old value,
        # effectively only summing over a recent window.
        self.sh_g2 = [
            _th.shared(_np.zeros_like(p.get_value()), broadcastable=p.broadcastable, name='g2_'+p.name)
            for p in model.params
        ]

        # Similarly to momentum, AdaDelta accumulates previous update values.
        # This also happens in a decaying fashion, so as to cover a window.
        self.sh_delta2 = [
            _th.shared(_np.zeros_like(p.get_value()), broadcastable=p.broadcastable, name='d2_'+p.name)
            for p in model.params
        ]

        g = _T.grad(cost=self.cost_expr, wrt=self.model.params)

        updates = []
        for sh_p, gp, sh_g2, sh_d2 in zip(self.model.params, g, self.sh_g2, self.sh_delta2):
            g2 = self.sh_rho*sh_g2 + (1-self.sh_rho)*gp*gp
            up = _T.sqrt((sh_d2+eps) / (g2+eps)) * gp
            d2 = self.sh_rho*sh_d2 + (1-self.sh_rho)*up*up
            updates.append((sh_g2, g2))
            updates.append((sh_p, sh_p - up))
            updates.append((sh_d2, d2))

        self._mk_train_fn("StreaMiniAdaDelta train",
            updates,
            extra_in=_th.Param(self.sh_rho, rho)
        )


    def reinit(self):
        for sh_g2 in self.sh_g2:
            sh_g2.set_value(_np.zeros_like(sh_g2.get_value()))
        for sh_delta2 in self.sh_delta2:
            sh_delta2.set_value(_np.zeros_like(sh_delta2.get_value()))
