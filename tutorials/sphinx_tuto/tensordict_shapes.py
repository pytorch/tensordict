"""
Changing the shape of a TensorDict
==================================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

A :class:`~tensordict.TensorDict` has the shape operations of a tensor, such
as ``reshape``, ``split``, ``stack`` and ``permute``. Each call changes every
tensor in it in the same way, so you write the call once, and the tensors stay
aligned. This tutorial is for readers who know these operations on a
:class:`torch.Tensor`.

What you will learn:

- How a shape operation on a tensordict applies to the batch dimensions of
  every entry.
- How to split, unbind, stack and concatenate tensordicts.
- The difference between a lazy stack and a dense stack.
- How to change the shape of a tensordict for the length of a ``with`` block.
"""

##############################################################################
# When you create a :class:`~tensordict.TensorDict`, you give it a
# ``batch_size``: the leading dimensions that all the entries share. An entry
# can have more dimensions after them (see :ref:`batch-size`). The shape
# operations below change the batch dimensions of every entry, and keep the
# other dimensions.

import torch
from tensordict import TensorDict

torch.manual_seed(0)

##############################################################################
# Indexing
# --------
#
# An index applies to the batch dimensions. Each entry of the
# :class:`~tensordict.TensorDict` is indexed in the same way. The
# :ref:`sphx_glr_tutorials_tensordict_slicing.py` tutorial covers indexing
# in detail.

a = torch.rand(3, 4)
b = torch.rand(3, 4, 5)
tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4])

indexed_tensordict = tensordict[:2, 1]
assert indexed_tensordict["a"].shape == torch.Size([2])
assert indexed_tensordict["b"].shape == torch.Size([2, 5])

##############################################################################
# Reshaping
# ---------
#
# :meth:`TensorDict.reshape <tensordict.TensorDict.reshape>` works like
# :meth:`torch.Tensor.reshape`. It reshapes the batch dimensions of every
# entry, and keeps the other dimensions, as the shape of ``b`` below shows. It
# also updates the ``batch_size`` attribute.


reshaped_tensordict = tensordict.reshape(-1)
assert reshaped_tensordict.batch_size == torch.Size([12])
assert reshaped_tensordict["a"].shape == torch.Size([12])
assert reshaped_tensordict["b"].shape == torch.Size([12, 5])

##############################################################################
# Splitting
# ---------
#
# :meth:`TensorDict.split <tensordict.TensorDict.split>` works like
# :meth:`torch.Tensor.split`. It splits the :class:`~tensordict.TensorDict` into
# chunks. Each chunk is a :class:`~tensordict.TensorDict` with the same keys as
# the original one. Its entries are views of the entries of the original
# :class:`~tensordict.TensorDict`: they share memory with them.

chunks = tensordict.split([3, 1], dim=1)
assert chunks[0].batch_size == torch.Size([3, 3])
assert chunks[1].batch_size == torch.Size([3, 1])
torch.testing.assert_close(chunks[0]["a"], tensordict["a"][:, :-1])

##############################################################################
# .. note::
#
#    If a function or method takes a ``dim`` argument, a negative ``dim`` counts
#    from the end of the ``batch_size`` of the :class:`~tensordict.TensorDict`
#    that you call it on. If the tensordict contains nested tensordicts with
#    more batch dimensions, a negative ``dim`` still counts from the end of
#    the batch dimensions of the root tensordict.
#
#       >>> tensordict = TensorDict(
#       ...     {
#       ...         "a": torch.rand(3, 4),
#       ...         "nested": TensorDict({"b": torch.rand(3, 4, 5)}, [3, 4, 5])
#       ...     },
#       ...     [3, 4],
#       ... )
#       >>> # dim=-2 is the first dimension for all entries: the root
#       >>> # TensorDict has 2 batch dimensions, the nested TensorDict has 3
#       >>> chunks = tensordict.split([2, 1], dim=-2)
#       >>> assert chunks[0].batch_size == torch.Size([2, 4])
#       >>> assert chunks[0]["nested"].batch_size == torch.Size([2, 4, 5])
#
#    In this example, :meth:`TensorDict.split <tensordict.TensorDict.split>`
#    gives the same result as with ``dim=tensordict.batch_dims - 2``.
#
# Unbinding
# ---------
# :meth:`TensorDict.unbind <tensordict.TensorDict.unbind>` works like
# :meth:`torch.Tensor.unbind`. It removes a dimension and returns a ``tuple``
# of all the slices along that dimension.

slices = tensordict.unbind(dim=1)
assert len(slices) == 4
assert all(s.batch_size == torch.Size([3]) for s in slices)
torch.testing.assert_close(slices[0]["a"], tensordict["a"][:, 0])

##############################################################################
# Stacking and concatenating
# --------------------------
#
# :func:`torch.cat` and :func:`torch.stack` accept lists of
# :class:`~tensordict.TensorDict` instances.
#
# Stacking
# ^^^^^^^^
# You can stack tensordicts in two ways. :func:`torch.stack` copies the values
# into new tensors and returns a :class:`~tensordict.TensorDict`.
# :func:`~tensordict.lazy_stack` keeps the original tensordicts in a list and
# returns a :class:`~tensordict.LazyStackedTensorDict`, which stacks a value
# only when you read it. Use a lazy stack when the tensordicts have different
# shapes, devices or keys, or when you read only a few keys of the stack. The
# :ref:`stacking` page explains both kinds of stacks.

from tensordict import lazy_stack

cloned_tensordict = tensordict.clone()
stacked_tensordict = lazy_stack([tensordict, cloned_tensordict], dim=0)
print(stacked_tensordict)

##############################################################################
# Like :func:`torch.stack` on tensors, :func:`torch.stack` on tensordicts
# returns a dense :class:`~tensordict.TensorDict`:

dense_stack = torch.stack([tensordict, cloned_tensordict])
assert isinstance(dense_stack, TensorDict)

##############################################################################
# If you index a :class:`~tensordict.LazyStackedTensorDict` along the stack
# dimension, you get back the original :class:`~tensordict.TensorDict`.

assert stacked_tensordict[0] is tensordict
assert stacked_tensordict[1] is cloned_tensordict

##############################################################################
# If you read a key of the :class:`~tensordict.LazyStackedTensorDict`, it
# stacks the values of this key. If the value is a nested
# :class:`~tensordict.TensorDict`, you get another
# :class:`~tensordict.LazyStackedTensorDict`.

assert stacked_tensordict["a"].shape == torch.Size([2, 3, 4])

###############################################################################
# .. note::
#
#    A lazy stack stacks a value each time you read it. If you read a value of
#    the stack more than once, convert the
#    :class:`~tensordict.LazyStackedTensorDict` to a dense
#    :class:`~tensordict.TensorDict` first, with
#    :meth:`LazyStackedTensorDict.to_tensordict <tensordict.LazyStackedTensorDict.to_tensordict>`
#    or :meth:`LazyStackedTensorDict.contiguous <tensordict.LazyStackedTensorDict.contiguous>`.
#
#       >>> assert isinstance(stacked_tensordict.to_tensordict(), TensorDict)
#       >>> assert isinstance(stacked_tensordict.contiguous(), TensorDict)
#
#    The result holds the stacked values, and reading a value does not stack
#    it again.
#
# Concatenating
# ^^^^^^^^^^^^^
# Concatenation is never lazy. :func:`torch.cat` on a list of
# :class:`~tensordict.TensorDict` instances returns a
# :class:`~tensordict.TensorDict` whose entries are the concatenated entries
# of the tensordicts of the list.

concatenated_tensordict = torch.cat([tensordict, cloned_tensordict], dim=0)
assert isinstance(concatenated_tensordict, TensorDict)
assert concatenated_tensordict.batch_size == torch.Size([6, 4])
assert concatenated_tensordict["b"].shape == torch.Size([6, 4, 5])

##############################################################################
# Expanding
# ---------
# :meth:`TensorDict.expand <tensordict.TensorDict.expand>` expands all the
# entries of a :class:`~tensordict.TensorDict`. As with
# :meth:`torch.Tensor.expand`, the expanded entries are views: they share
# memory with the original entries. Call ``clone()`` on the result before
# you write into it.

exp_tensordict = tensordict.expand(2, *tensordict.batch_size)
assert exp_tensordict.batch_size == torch.Size([2, 3, 4])
torch.testing.assert_close(exp_tensordict["a"][0], exp_tensordict["a"][1])

##############################################################################
# Squeezing and unsqueezing
# -------------------------
# The :meth:`~tensordict.TensorDictBase.squeeze` and
# :meth:`~tensordict.TensorDictBase.unsqueeze` methods remove or add a batch
# dimension of size one.

tensordict = TensorDict({"a": torch.rand(3, 1, 4)}, [3, 1, 4])
squeezed_tensordict = tensordict.squeeze()
assert squeezed_tensordict["a"].shape == torch.Size([3, 4])
print(squeezed_tensordict, end="\n\n")

unsqueezed_tensordict = tensordict.unsqueeze(-1)
assert unsqueezed_tensordict["a"].shape == torch.Size([3, 1, 4, 1])
print(unsqueezed_tensordict)

##############################################################################
# .. note::
#    :meth:`~tensordict.TensorDictBase.unsqueeze`,
#    :meth:`~tensordict.TensorDictBase.squeeze`, :meth:`~tensordict.TensorDictBase.view`,
#    :meth:`~tensordict.TensorDictBase.permute` and
#    :meth:`~tensordict.TensorDictBase.transpose` apply the operation to the
#    entries when you call them, and return a :class:`~tensordict.TensorDict`.
#
# These methods apply only to the batch dimensions. The other dimensions of
# the entries do not change:

tensordict = TensorDict({"a": torch.rand(3, 1, 1, 4)}, [3, 1])
squeezed_tensordict = tensordict.squeeze()
# only one of the two dimensions of size one is removed: the other
# one is not a batch dimension
assert squeezed_tensordict["a"].shape == torch.Size([3, 1, 4])

##############################################################################
# Viewing
# -------
# :meth:`~tensordict.TensorDictBase.view` returns a
# :class:`~tensordict.TensorDict` whose entries are views of the original
# entries.

tensordict = TensorDict({"a": torch.arange(12)}, [12])
# the entries are viewed at this step
viewed_tensordict = tensordict.view((2, 3, 2))

# "a" is a view of the original tensor
assert viewed_tensordict["a"].shape == torch.Size([2, 3, 2])


##############################################################################
# Permuting the batch dimensions
# ------------------------------
# :meth:`TensorDict.permute <tensordict.TensorDict.permute>` reorders the batch
# dimensions, like :func:`torch.permute`. The other dimensions do not change.
#
# Like the other shape operations above, ``permute`` applies to the entries
# when you call it.

tensordict = TensorDict({"a": torch.rand(3, 4), "b": torch.rand(3, 4, 5)}, [3, 4])
# swap the batch dimensions
permuted_tensordict = tensordict.permute([1, 0])

assert permuted_tensordict["a"].shape == torch.Size([4, 3])
assert permuted_tensordict["b"].shape == torch.Size([4, 3, 5])

##############################################################################
# Reversible operations as context managers
# -----------------------------------------
#
# You can use the result of some reversible operations as a context manager,
# in a ``with`` block. At the end of the block, tensordict applies the inverse
# operation and writes the changes back into the original tensordict. The
# shape operations :meth:`~tensordict.TensorDictBase.view`,
# :meth:`~tensordict.TensorDictBase.permute`,
# :meth:`~tensordict.TensorDictBase.transpose`,
# :meth:`~tensordict.TensorDictBase.squeeze` and
# :meth:`~tensordict.TensorDictBase.unsqueeze` work in this way. So do
# :meth:`~tensordict.TensorDictBase.to_module` for functional calls, and
# :meth:`~tensordict.TensorDictBase.lock_` and
# :meth:`~tensordict.TensorDictBase.unlock_`. This example uses ``transpose``:

tensordict = TensorDict({"a": torch.rand(3, 4), "b": torch.rand(3, 4, 5)}, [3, 4])

with tensordict.transpose(1, 0) as tdt:
    tdt.set("c", torch.ones(4, 3))  # the batch size of tdt is [4, 3]

##############################################################################
# After the block, the original tensordict has the ``"c"`` entry, with the
# batch dimensions in the original order:

assert (tensordict.get("c") == 1).all()
assert tensordict["c"].shape == torch.Size([3, 4])

##############################################################################
# Gathering values
# ----------------
# :meth:`TensorDict.gather <tensordict.TensorDict.gather>` gathers values along
# a batch dimension with an index tensor, like :func:`torch.gather`.

index = torch.randint(4, (3, 4))
gathered_tensordict = tensordict.gather(dim=1, index=index)
print("index:\n", index, end="\n\n")
print("tensordict['a']:\n", tensordict["a"], end="\n\n")
print("gathered_tensordict['a']:\n", gathered_tensordict["a"], end="\n\n")

##############################################################################
# Conclusion
# ----------
#
# You can now index, ``reshape``, ``split``, ``unbind``, ``expand``,
# ``squeeze``, ``unsqueeze``, ``view``, ``permute`` and ``gather`` a
# :class:`~tensordict.TensorDict` with one call each. Each call changes the
# batch dimensions of every entry and keeps the other dimensions. You can
# stack tensordicts densely with :func:`torch.stack`, or lazily with
# :func:`~tensordict.lazy_stack` when their shapes differ. You can also change
# the shape for the length of a ``with`` block, and get the changes back in the
# original tensordict.
#
# The same operations work on a :class:`~tensordict.TensorClass`, the class
# form of a tensordict with typed fields (see :ref:`tensorclass-guide`).
#
# Further reading
# ---------------
#
# - :ref:`batch-size` explains batch sizes, shapes and indexing.
# - :ref:`stacking` explains lazy stacks and data of different sizes.
# - :ref:`sphx_glr_tutorials_tensordict_slicing.py` shows indexing, masking
#   and writing into a tensordict with an index.
# - :ref:`api-methods-shape` lists all the shape methods.
