# -*- coding: utf-8 -*-
"""
Pre-allocating memory with TensorDict
=====================================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

In this tutorial you will learn how to take advantage of memory pre-allocation in
:class:`~.TensorDict`.
"""

##############################################################################
# Suppose that we have a function that returns a :class:`~.TensorDict`


# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore
import torch
from tensordict.tensordict import TensorDict


def make_tensordict():
    return TensorDict({"a": torch.rand(3), "b": torch.rand(3, 4)}, [3])


###############################################################################
# Perhaps we want to call this function multiple times and use the results to populate
# a single :class:`~.TensorDict`.

N = 10
tensordict = TensorDict({}, batch_size=[N, 3])

for i in range(N):
    tensordict[i] = make_tensordict()

print(tensordict)

###############################################################################
# Because we have specified the ``batch_size`` of ``tensordict``, during the first
# iteration of the loop we populate ``tensordict`` with empty tensors whose first
# dimension is size ``N``, and whose remaining dimensions are determined by the return
# value of ``make_tensordict``. In the above example, we pre-allocate an array of zeros
# of size ``torch.Size([10, 3])`` for the key ``"a"``, and an array size
# ``torch.Size([10, 3, 4])`` for the key ``"b"``. Subsequent iterations of the loop are
# written in place. As a result, if not all values are filled, they get the default
# value of zero.
#
# Let us demonstrate what is going on by stepping through the above loop. We first
# initialise an empty :class:`~.TensorDict`.

N = 10
tensordict = TensorDict({}, batch_size=[N, 3])
print(tensordict)

##############################################################################
# After the first iteration, ``tensordict`` has been prepopulated with tensors for both
# ``"a"`` and ``"b"``. These tensors contain zeros except for the first row which we
# have assigned random values to.

random_tensordict = make_tensordict()
tensordict[0] = random_tensordict

assert (tensordict[1:] == 0).all()
assert (tensordict[0] == random_tensordict).all()

print(tensordict)

##############################################################################
# Subsequent iterations, we update the pre-allocated tensors in-place.

a = tensordict["a"]
random_tensordict = make_tensordict()
tensordict[1] = random_tensordict

# the same tensor is stored under "a", but the values have been updated
assert tensordict["a"] is a
assert (tensordict[:2] != 0).all()
