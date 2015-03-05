#!/usr/bin/env python3


def tuplize(what):
    """
    If `what` is a tuple, return it as-is, otherwise put it into a tuple.
    """
    if isinstance(what, tuple):
        return what
    else:
        return (what,)


def maybetuple(what):
    """
    Transforms `what` into a tuple, except if it's of length one.
    """
    t = tuple(what)
    return t if len(t) > 1 else t[0]


def batched(batchsize, *args):
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

    # First, go through all full batches.
    for i in range(n // batchsize):
        yield maybetuple(x[i*batchsize:(i+1)*batchsize] for x in args)

    # And now maybe return the last batch.
    rest = n % batchsize
    if rest != 0:
        yield maybetuple(x[-rest:] for x in args)
