# -*- coding: utf-8 -*-
"""
Slicing, Indexing, and Masking
==============================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

In this tutorial you will learn how to slice, index, and mask a :class:`~.TensorDict`.
"""

##############################################################################
# As discussed in the tutorial
# `Manipulating the shape of a TensorDict <./tensordict_shapes.html>`_, when we create a
# :class:`~.TensorDict` we specify a ``batch_size``, which must agree
# with the leading dimensions of all entries in the :class:`~.TensorDict`. Since we have
# a guarantee that all entries share those dimensions in common, we are able to index
# and mask the batch dimensions in the same way that we would index a
# :class:`torch.Tensor`. The indices are applied along the batch dimensions to all of
# the entries in the :class:`~.TensorDict`.
#
# For example, given a :class:`~.TensorDict` with two batch dimensions,
# ``tensordict[0]`` returns a new :class:`~.TensorDict` with the same structure, and
# whose values correspond to the first "row" of each entry in the original
# :class:`~.TensorDict`.

import torch
from tensordict import TensorDict

tensordict = TensorDict(
    {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)

print(tensordict[0])

##############################################################################
# The same syntax applies as for regular tensors. For example if we wanted to drop the
# first row of each entry we could index as follows

print(tensordict[1:])

##############################################################################
# We can index multiple dimensions simultaneously

print(tensordict[:, 2:])

##############################################################################
# We can also use ``Ellipsis`` to represent as many ``:`` as would be needed to make
# the selection tuple the same length as ``tensordict.batch_dims``.

print(tensordict[..., 2:])

##############################################################################
# .. note:
#
#    Remember that all indexing is applied relative to the batch dimensions. In the
#    above example there is a difference between ``tensordict["a"][..., 2:]`` and
#    ``tensordict[..., 2:]["a"]``. The first retrieves the three-dimensional tensor
#    stored under the key ``"a"`` and applies the index ``2:`` to the final dimension.
#    The second applies the index ``2:`` to the final *batch dimension*, which is the
#    second dimension, before retrieving the result.
#
# Setting Values with Indexing
# ----------------------------
# In general, ``tensordict[index] = new_tensordict`` will work as long as the batch
# sizes are compatible.

tensordict = TensorDict(
    {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)

td2 = TensorDict({"a": torch.ones(2, 4, 5), "b": torch.ones(2, 4)}, batch_size=[2, 4])
tensordict[:-1] = td2
print(tensordict["a"], tensordict["b"])

##############################################################################
# Masking
# -------
# We mask :class:`TensorDict` as we mask tensors.

mask = torch.BoolTensor([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])
tensordict[mask]

##############################################################################
# SubTensorDict
# -------------
# When we index a :class:`~.TensorDict` with a contiguous index, we obtain a new
# :class:`~.TensorDict` whose values are all views on the values of the original
# :class:`~.TensorDict`. That means updates to the indexed :class:`~.TensorDict` are
# applied to the original also.

tensordict = TensorDict(
    {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)
td2 = tensordict[1:]
td2.fill_("b", 1)

assert (tensordict["b"][1:] == 1).all()
print(tensordict["b"])

##############################################################################
# This doesn't work however if we use a non-contiguous index

tensordict = TensorDict(
    {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)
td2 = tensordict[[0, 2]]
td2.fill_("b", 1)

assert (tensordict == 0).all()
print(tensordict["b"])

##############################################################################
# In case such functionality is needed, one can use
# :meth:`TensorDict.get_sub_tensordict <tensordict.TensorDict.get_sub_tensordict>`
# instead. The :class:`~.SubTensorDict` holds a reference to the orgiinal
# :class:`~.TensorDict` so that updates to the sub-tensordict can be written back to the
# source.

tensordict = TensorDict(
    {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)
td2 = tensordict.get_sub_tensordict(([0, 2],))
td2.fill_("b", 1)
print(tensordict["b"])
