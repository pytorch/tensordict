"""
Pre-allocating memory with TensorDict
=====================================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

This tutorial shows how to fill an empty :class:`~tensordict.TensorDict` row by
row. The tensordict allocates the memory for all the rows when you write the
first row.

What you will learn:

- How to create an empty tensordict with a batch size and no keys.
- What tensordict allocates when you write the first row with an index.
- How the later rows are written into the same tensors.
"""

##############################################################################
# Here is a function that returns a :class:`~tensordict.TensorDict` with a
# batch size of ``[3]``:

import torch
from tensordict import TensorDict

torch.manual_seed(0)


def make_tensordict():
    return TensorDict({"a": torch.rand(3), "b": torch.rand(3, 4)}, [3])


###############################################################################
# Call this function ``N`` times, and write the results into one
# :class:`~tensordict.TensorDict` with ``batch_size=[N, 3]``:

N = 10
tensordict = TensorDict({}, batch_size=[N, 3])

for i in range(N):
    tensordict[i] = make_tensordict()

print(tensordict)

###############################################################################
# ``tensordict`` has ``batch_size=[N, 3]`` and no keys. When the first
# iteration writes ``tensordict[0]``, tensordict creates each missing key as a
# zero-filled tensor. Its shape is the batch size ``[N, 3]`` followed by the
# other dimensions of the value from ``make_tensordict``. Here ``"a"`` gets
# shape ``torch.Size([10, 3])`` and ``"b"`` gets shape
# ``torch.Size([10, 3, 4])``. The later iterations write into these tensors in
# place. Rows that the loop does not write stay zero.
#
# The steps below go through the loop one iteration at a time. First, create
# an empty :class:`~tensordict.TensorDict`:

N = 10
tensordict = TensorDict({}, batch_size=[N, 3])
print(tensordict)

##############################################################################
# After the first iteration, ``tensordict`` has tensors for ``"a"`` and
# ``"b"``. These tensors are zero, except for the first row, which holds the
# random values that you wrote.

random_tensordict = make_tensordict()
tensordict[0] = random_tensordict

assert (tensordict[1:] == 0).all()
assert (tensordict[0] == random_tensordict).all()

print(tensordict)

##############################################################################
# The next iterations write into the same tensors, in place.

a = tensordict["a"]
random_tensordict = make_tensordict()
tensordict[1] = random_tensordict

# the same tensor is stored under "a", but its values changed
assert tensordict["a"] is a
assert (tensordict[:2] != 0).all()

##############################################################################
# Conclusion
# ----------
#
# In this tutorial, you created a :class:`~tensordict.TensorDict` with a batch
# size and no keys, and you filled it one row at a time. The first write
# allocates a zero-filled tensor for each key. The later writes go into the
# same tensors.
#
# To allocate all the keys in advance, from their shapes and dtypes, without a
# first row, use :meth:`~tensordict.TensorDictBase.from_schema`. It can also
# allocate the tensors on disk.
#
# Further reading
# ---------------
#
# - :ref:`batch-size` explains batch sizes and indexing.
# - :ref:`storage` shows how to pre-allocate a tensordict on disk or on a
#   storage backend.
# - :ref:`sphx_glr_tutorials_tensordict_memory.py` allocates memory-mapped
#   tensors with ``memmap_like``.
# - :ref:`sphx_glr_tutorials_tensorclass_imagenet.py` fills a memory-mapped
#   dataset batch by batch.
