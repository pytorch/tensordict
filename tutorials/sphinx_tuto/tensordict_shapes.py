# -*- coding: utf-8 -*-
"""
Manipulating the shape of a TensorDict
======================================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

In this tutorial you will learn how to manipulate the shape of a ``TensorDict`` and its
contents.
"""

##############################################################################
# When we create a ``TensorDict`` we specify a ``batch_size``, which must agree with
# the leading dimensions of all entries in the ``TensorDict``. Since we have a guarantee
# that all entries share those dimensions in common, ``TensorDict`` is able to expose
# a number of methods with which we can manipulate the shape of the ``TensorDict`` and
# its contents.

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore
import torch
from tensordict.tensordict import TensorDict

##############################################################################
# Reshaping a ``TensorDict``
# --------------------------
#
# ``TensorDict.reshape`` works just like ``torch.Tensor.reshape``. It applies to all of
# the contents of the ``TensorDict`` along the batch dimensions - note the shape of
# ``b`` in the example below. It also updates the ``batch_size`` attribute.

a = torch.rand(3, 4)
b = torch.rand(3, 4, 5)
tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4])

reshaped_tensordict = tensordict.reshape(-1)
assert reshaped_tensordict.batch_size == torch.Size([12])
assert reshaped_tensordict["a"].shape == torch.Size([12])
assert reshaped_tensordict["b"].shape == torch.Size([12, 5])

##############################################################################
# Splitting a ``TensorDict``
# --------------------------
#
# ``TensorDict.split`` is similar to ``torch.Tensor.split``. It splits the
# ``TensorDict`` into chunks. Each chunk is a ``TensorDict`` with the same structure
# as the original one, but whose entries are views of the corresponding entries in the
# original tensordict.

chunks = tensordict.split([3, 1], dim=1)
assert chunks[0].batch_size == torch.Size([3, 3])
assert chunks[1].batch_size == torch.Size([3, 1])
torch.testing.assert_allclose(chunks[0]["a"], tensordict["a"][:, :-1])


##############################################################################
# Stacking and concatenating
# --------------------------
#
# ``TensorDict`` can be used in conjunction with ``torch.cat`` and ``torch.stack``.
#
# Stacking ``TensorDict``
# ^^^^^^^^^^^^^^^^^^^^^^^
# By default, stacking is done in a lazy fashion, returning a ``LazyStackedTensorDict``
# object. In this case values are only stacked on-demand when they are accessed. This
# in cases where you have a large ``TensorDict`` with many entries, and you don't need
# to stack all of them.

cloned_tensordict = tensordict.clone()
# no stacking happens on the next line
stacked_tensordict = torch.stack([tensordict, cloned_tensordict], dim=0)
print(stacked_tensordict)

##############################################################################
# If we index a ``LazyStackedTensorDict`` we recover the original ``TensorDict``.

assert stacked_tensordict[0] is tensordict
assert stacked_tensordict[1] is cloned_tensordict

##############################################################################
# Accessing a key in the ``LazyStackedTensorDict`` results in those values being
# stacked. If the key corresponds to a nested ``TensorDict`` then we will recover
# another ``LaxyStackedTensorDict``.

assert stacked_tensordict["a"].shape == torch.Size([2, 3, 4])

###############################################################################
# If we want to have a contiguous ``TensorDict``, we can call ``.to_tensordict()``
# or ``.contiguous()``. It is recommended to perform this operation before
# accessing the values of the stacked ``TensorDict`` so that the stacking operation is
# not performed more often than is necessary.

assert isinstance(stacked_tensordict.contiguous(), TensorDict)
assert isinstance(stacked_tensordict.to_tensordict(), TensorDict)

###############################################################################
# Concatenating ``TensorDict``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Concatenation is not done lazily, instead calling ``torch.cat`` on a list of
# ``TensorDict`` instances simply returns a ``TensorDict`` whose entries are the
# concatenated entries of the elements of the list.

concatenated_tensordict = torch.cat([tensordict, cloned_tensordict], dim=0)
assert isinstance(concatenated_tensordict, TensorDict)
assert concatenated_tensordict.batch_size == torch.Size([6, 4])
assert concatenated_tensordict["b"].shape == torch.Size([6, 4, 5])

##############################################################################
# Expanding ``TensorDict``
# ------------------------
# We can expand all of the entries of a ``TensorDict`` using ``TensorDict.expand``.

exp_tensordict = tensordict.expand(2, *tensordict.batch_size)
assert exp_tensordict.batch_size == torch.Size([2, 3, 4])
torch.testing.assert_allclose(exp_tensordict["a"][0], exp_tensordict["a"][1])
