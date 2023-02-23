# -*- coding: utf-8 -*-
"""
Pre-allocating memory with TensorDict
=====================================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

In this tutorial you will learn how to take advantage of memory pre-allocation in
``TensorDict``.
"""

##############################################################################
# Suppose that we have a function that returns a ``TensorDict``


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
# a single ``TensorDict``.

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
# written in place.
#
# As a result, if not all values are filled, they get the default value of zero.

N = 10
tensordict = TensorDict({}, batch_size=[N, 3])

for i in range(2):
    tensordict[i] = make_tensordict()

assert (tensordict[2:] == 0).all()
