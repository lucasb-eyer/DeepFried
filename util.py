#!/usr/bin/env python3

import numpy as _np
import numbers as _num


def tuplize(what, lists=True, tuplize_none=False):
    """
    If `what` is a tuple, return it as-is, otherwise put it into a tuple.
    If `lists` is true, also consider lists to be tuples (the default).
    If `tuplize_none` is true, a lone `None` results in an empty tuple,
    otherwise it will be returned as `None` (the default).
    """
    if what is None:
        if tuplize_none:
            return tuple()
        else:
            return None

    if isinstance(what, tuple) or (lists and isinstance(what, list)):
        return tuple(what)
    else:
        return (what,)


def maybetuple(what):
    """
    Transforms `what` into a tuple, except if it's of length one.
    """
    t = tuple(what)
    return t if len(t) > 1 else t[0]


def collect(what, drop_nones=True):
    """
    Returns a tuple that is the concatenation of all tuplized things in `what`.
    """
    return sum((tuplize(w, tuplize_none=drop_nones) for w in what), tuple())


def batched(batchsize, *args, shuf=False):
    """
    A generator function which goes through all of `args` together,
    but in batches of size `batchsize` along the first dimension.

    batched_padded(3, np.arange(10), np.arange(10))

    will yield sub-arrays of the given ones four times, the fourth one only
    containing a single value.
    """

    assert(len(args) > 0)

    n = args[0].shape[0]

    # Assumption: all args have the same 1st dimension as the first one.
    assert(all(x.shape[0] == n for x in args))

    indices = _np.arange(n)
    if shuf is not False:
        rng = check_random_state(shuf)
        rng.shuffle(indices)

    # First, go through all full batches.
    for i in range(n // batchsize):
        yield maybetuple(x[indices[i*batchsize:(i+1)*batchsize]] for x in args)

    # And now maybe return the last batch.
    rest = n % batchsize
    if rest != 0:
        yield maybetuple(x[indices[-rest:]] for x in args)


# Blatantly "inspired" by sklearn, for when that's not available.
def check_random_state(seed):
    """
    Turn `seed` into a `np.random.RandomState` instance.

    - If `seed` is `None`, return the `RandomState` singleton used by `np.random`.
    - If `seed` is an `int`, return a new `RandomState` instance seeded with `seed`.
    - If `seed` is already a `RandomState` instance, return it.
    - Otherwise raise `ValueError`.
    """
    if seed is None or seed is _np.random:
        return _np.random.mtrand._rand

    if isinstance(seed, (_num.Integral, _np.integer)):
        return _np.random.RandomState(seed)

    if isinstance(seed, _np.random.RandomState):
        return seed

    raise ValueError('{!r} cannot be used to seed a numpy.random.RandomState instance'.format(seed))


def check_all_initialized(model):
    return all(not _np.any(_np.isnan(p.get_value())) for p in model.params)


def flipdim(a, dim):
    """
    `flipdim(a, 0)` is equivalent to `flipud(a)`,
    `flipdim(a, 1)` is equivalent to `fliplr(a)` and the rest follows naturally.
    """
    # Put the axis in front, flip that axis, then move it back.
    return _np.swapaxes(_np.swapaxes(a, 0, dim)[::-1], 0, dim)
