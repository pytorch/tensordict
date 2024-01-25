# -*- coding: utf-8 -*-
"""
Manipulating the shape of a TensorDict
======================================
**Author**: `Tom Begley <https://github.com/tcbegley>`_

In this tutorial you will learn how to manipulate the shape of a :class:`~.TensorDict`
and its contents.
"""

##############################################################################
# When we create a :class:`~.TensorDict` we specify a ``batch_size``, which must agree
# with the leading dimensions of all entries in the :class:`~.TensorDict`. Since we have
# a guarantee that all entries share those dimensions in common, :class:`~.TensorDict`
# is able to expose a number of methods with which we can manipulate the shape of the
# :class:`~.TensorDict` and its contents.

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore
import torch
from tensordict.tensordict import TensorDict

##############################################################################
# Indexing a ``TensorDict``
# -------------------------
#
# Since the batch dimensions are guaranteed to exist on all entries, we can index them
# as we please, and each entry of the :class:`~.TensorDict` will be indexed in the same
# way.

a = torch.rand(3, 4)
b = torch.rand(3, 4, 5)
tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4])

indexed_tensordict = tensordict[:2, 1]
assert indexed_tensordict["a"].shape == torch.Size([2])
assert indexed_tensordict["b"].shape == torch.Size([2, 5])

##############################################################################
# Reshaping a ``TensorDict``
# --------------------------
#
# :meth:`TensorDict.reshape <tensordict.TensorDict.reshape>` works just like
# :meth:`torch.Tensor.reshape`. It applies to all of the contents of the
# :class:`~.TensorDict` along the batch dimensions - note the shape of ``b`` in the
# example below. It also updates the ``batch_size`` attribute.


reshaped_tensordict = tensordict.reshape(-1)
assert reshaped_tensordict.batch_size == torch.Size([12])
assert reshaped_tensordict["a"].shape == torch.Size([12])
assert reshaped_tensordict["b"].shape == torch.Size([12, 5])

##############################################################################
# Splitting a ``TensorDict``
# --------------------------
#
# :meth:`TensorDict.split <tensordict.TensorDict.split>` is similar to
# :meth:`torch.Tensor.split`. It splits the :class:`~.TensorDict` into chunks. Each
# chunk is a :class:`~.TensorDict` with the same structure as the original one, but
# whose entries are views of the corresponding entries in the original
# :class:`~.TensorDict`.

chunks = tensordict.split([3, 1], dim=1)
assert chunks[0].batch_size == torch.Size([3, 3])
assert chunks[1].batch_size == torch.Size([3, 1])
torch.testing.assert_close(chunks[0]["a"], tensordict["a"][:, :-1])

##############################################################################
# .. note::
#
#    Whenever a function or method accepts a ``dim`` argument, negative dimensions are
#    interpreted relative to the ``batch_size`` of the :class:`~.TensorDict` that the
#    function or method is called on. In particular, if there are nested
#    :class:`~.TensorDict` values with different batch sizes, the negative dimension is
#    always interpreted relative to the batch dimensions of the root.
#
#    .. code-block::
#
#       tensordict = TensorDict(
#           {
#               "a": torch.rand(3, 4),
#               "nested": TensorDict({"b": torch.rand(3, 4, 5)}, [3, 4, 5])
#           },
#           [3, 4],
#       )
#       # dim = -2 will be interpreted as the first dimension throughout, as the root
#       # TensorDict has 2 batch dimensions, even though the nested TensorDict has 3
#       chunks = tensordict.split([2, 1], dim=-2)
#       assert chunks[0].batch_size == torch.Size([2, 4])
#       assert chunks[0]["nested"].batch_size == torch.Size([2, 4, 5])
#
#    As you can see from this example, the
#    :meth:`TensorDict.split <tensordict.TensorDict.split>` method behaves exactly as
#    though we had replaced ``dim=-2`` with ``dim=tensordict.batch_dims - 2`` before
#    calling.
#
# Unbind
# ------
# :meth:`TensorDict.unbind <tensordict.TensorDict.unbind>` is similar to
# :meth:`torch.Tensor.unbind`, and conceptually similar to
# :meth:`TensorDict.split <tensordict.TensorDict.split>`. It removes the specified
# dimension and returns a ``tuple`` of all slices along that dimension.

slices = tensordict.unbind(dim=1)
assert len(slices) == 4
assert all(s.batch_size == torch.Size([3]) for s in slices)
torch.testing.assert_close(slices[0]["a"], tensordict["a"][:, 0])

##############################################################################
# Stacking and concatenating
# --------------------------
#
# :class:`~.TensorDict` can be used in conjunction with ``torch.cat`` and ``torch.stack``.
#
# Stacking ``TensorDict``
# ^^^^^^^^^^^^^^^^^^^^^^^
# Stacking can done lazily or contiguously. A lazy stack is just a list of tensordicts
# presented as a stack of tensordicts. It allows users to carry a bag of tensordicts
# with different content shape, device or key sets. Another advantage is that
# the stack operation can be expensive, and if only a small subset of keys is required,
# a lazy stack will be much faster than a proper stack.
# It relies on the :class:`~tensordict.LazyStackedTensorDict` class.
# In this case, values will only be stacked on-demand when they are accessed.

from tensordict import LazyStackedTensorDict

cloned_tensordict = tensordict.clone()
stacked_tensordict = LazyStackedTensorDict.lazy_stack(
    [tensordict, cloned_tensordict], dim=0
)
print(stacked_tensordict)

# Previously, torch.stack was always returning a lazy stack. For consistency with
# the regular PyTorch API, this behaviour will soon be adapted to deliver only
# dense tensordicts. To control which behaviour you are relying on, you can use
# the :func:`~tensordict.utils.set_lazy_legacy` decorator/context manager:

from tensordict.utils import set_lazy_legacy

with set_lazy_legacy(True):  # old behaviour
    lazy_stack = torch.stack([tensordict, cloned_tensordict])
assert isinstance(lazy_stack, LazyStackedTensorDict)

with set_lazy_legacy(False):  # new behaviour
    dense_stack = torch.stack([tensordict, cloned_tensordict])
assert isinstance(dense_stack, TensorDict)
##############################################################################
# If we index a :class:`~.LazyStackedTensorDict` along the stacking dimension we recover
# the original :class:`~.TensorDict`.

assert stacked_tensordict[0] is tensordict
assert stacked_tensordict[1] is cloned_tensordict

##############################################################################
# Accessing a key in the :class:`~.LazyStackedTensorDict` results in those values being
# stacked. If the key corresponds to a nested :class:`~.TensorDict` then we will recover
# another :class:`~.LazyStackedTensorDict`.

assert stacked_tensordict["a"].shape == torch.Size([2, 3, 4])

###############################################################################
# .. note::
#
#    Since values are stacked on-demand, accessing an item multiple times will mean it
#    gets stacked multiple times, which is inefficient. If you need to access a value
#    in the stacked :class:`~.TensorDict` more than once, you may want to consider
#    converting the :class:`LazyStackedTensorDict` to a contiguous
#    :class:`~.TensorDict`, which can be done with the
#    :meth:`LazyStackedTensorDict.to_tensordict <tensordict.LazyStackedTensorDict.to_tensordict>`
#    or :meth:`LazyStackedTensorDict.contiguous <tensordict.LazyStackedTensorDict.contiguous>`
#    methods.
#
#    .. code-block::
#       assert isinstance(stacked_tensordict.contiguous(), TensorDict)
#       assert isinstance(stacked_tensordict.contiguous(), TensorDict)
#
#    After calling either of these methods, we will have a regular :class:`TensorDict`
#    containing the stacked values, and no additional computation is performed when
#    values are accessed.
#
# Concatenating ``TensorDict``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Concatenation is not done lazily, instead calling :func:`torch.cat` on a list of
# :class:`~.TensorDict` instances simply returns a :class:`~.TensorDict` whose entries
# are the concatenated entries of the elements of the list.

concatenated_tensordict = torch.cat([tensordict, cloned_tensordict], dim=0)
assert isinstance(concatenated_tensordict, TensorDict)
assert concatenated_tensordict.batch_size == torch.Size([6, 4])
assert concatenated_tensordict["b"].shape == torch.Size([6, 4, 5])

##############################################################################
# Expanding ``TensorDict``
# ------------------------
# We can expand all of the entries of a :class:`~.TensorDict` using
# :meth:`TensorDict.expand <tensordict.TensorDict.expand>`.

exp_tensordict = tensordict.expand(2, *tensordict.batch_size)
assert exp_tensordict.batch_size == torch.Size([2, 3, 4])
torch.testing.assert_close(exp_tensordict["a"][0], exp_tensordict["a"][1])

##############################################################################
# Squeezing and Unsqueezing ``TensorDict``
# ----------------------------------------
# We can squeeze or unsqueeze the contents of a :class:`~.TensorDict` with the
# :meth:`~tensordict.TensorDictBase.squeeze` and
# :meth:`~tensordict.TensorDictBase.unsqueeze` methods.

tensordict = TensorDict({"a": torch.rand(3, 1, 4)}, [3, 1, 4])
squeezed_tensordict = tensordict.squeeze()
assert squeezed_tensordict["a"].shape == torch.Size([3, 4])
print(squeezed_tensordict, end="\n\n")

unsqueezed_tensordict = tensordict.unsqueeze(-1)
assert unsqueezed_tensordict["a"].shape == torch.Size([3, 1, 4, 1])
print(unsqueezed_tensordict)

##############################################################################
# .. note::
#    Until now, operations like :meth:`~tensordict.TensorDictBase.unsqueeze`,
#    :meth:`~tensordict.TensorDictBase.squeeze`, :meth:`~tensordict.TensorDictBase.view`,
#    :meth:`~tensordict.TensorDictBase.permute`, :meth:`~tensordict.TensorDictBase.transpose`
#    were all returning a lazy version of these operations (ie, a container where the original
#    tensordict was stored and where the operations was applied every time a key was accessed).
#    This behaviour will be deprecated in the future and can be already controlled via the
#    :func:`~tensordict.utils.set_lazy_legacy` function:
#
#       >>> with set_lazy_legacy(True):
#       ...     lazy_unsqueeze = tensordict.unsqueeze(0)
#       >>> with set_lazy_legacy(False):
#       ...     dense_unsqueeze = tensordict.unsqueeze(0)
#
# Bear in mind that as ever, these methods apply only to the batch dimensions. Any non
# batch dimensions of the entries will be unaffected

tensordict = TensorDict({"a": torch.rand(3, 1, 1, 4)}, [3, 1])
squeezed_tensordict = tensordict.squeeze()
# only one of the singleton dimensions is dropped as the other
# is not a batch dimension
assert squeezed_tensordict["a"].shape == torch.Size([3, 1, 4])

##############################################################################
# Viewing a TensorDict
# --------------------
# :class:`~.TensorDict` also supports ``view``. This creates a ``_ViewedTensorDict``
# which lazily creates views on its contents when they are accessed.

tensordict = TensorDict({"a": torch.arange(12)}, [12])
# no views are created at this step
viewed_tensordict = tensordict.view((2, 3, 2))

# the view of "a" is created on-demand when we access it
assert viewed_tensordict["a"].shape == torch.Size([2, 3, 2])


##############################################################################
# Permuting batch dimensions
# --------------------------
# The :meth:`TensorDict.permute <tensordict.TensorDict.permute>` method can be used to
# permute the batch dimensions much like :func:`torch.permute`. Non batch dimensions are
# left untouched.
#
# This operation is lazy, so batch dimensions are only permuted when we try to access
# the entries. As ever, if you are likely to need to access a particular entry multiple
# times, consider converting to a :class:`~.TensorDict`.

tensordict = TensorDict({"a": torch.rand(3, 4), "b": torch.rand(3, 4, 5)}, [3, 4])
# swap the batch dimensions
permuted_tensordict = tensordict.permute([1, 0])

assert permuted_tensordict["a"].shape == torch.Size([4, 3])
assert permuted_tensordict["b"].shape == torch.Size([4, 3, 5])

##############################################################################
# Using tensordicts as decorators
# -------------------------------
#
# For a bunch of reversible operations, tensordicts can be used as decorators.
# These operations include :meth:`~tensordict.TensorDictBase.to_module` for functional
# calls, :meth:`~tensordict.TensorDictBase.unlock_` and :meth:`~tensordict.TensorDictBase.lock_`
# or shape operations such as :meth:`~tensordict.TensorDictBase.view`, :meth:`~tensordict.TensorDictBase.permute`
# :meth:`~tensordict.TensorDictBase.transpose`, :meth:`~tensordict.TensorDictBase.squeeze` and
# :meth:`~tensordict.TensorDictBase.unsqueeze`.
# Here is a quick example with the ``transpose`` function:

tensordict = TensorDict({"a": torch.rand(3, 4), "b": torch.rand(3, 4, 5)}, [3, 4])

with tensordict.transpose(1, 0) as tdt:
    tdt.set("c", torch.ones(4, 3))  # we have permuted the dims

# the ``"c"`` entry is now in the tensordict we used as decorator:
#

assert (tensordict.get("c") == 1).all()

##############################################################################
# Gathering values in ``TensorDict``
# ----------------------------------
# The :meth:`TensorDict.gather <tensordict.TensorDict.gather>` method can be used to
# index along the batch dimensions and gather the results into a single dimension much
# like :func:`torch.gather`.

index = torch.randint(4, (3, 4))
gathered_tensordict = tensordict.gather(dim=1, index=index)
print("index:\n", index, end="\n\n")
print("tensordict['a']:\n", tensordict["a"], end="\n\n")
print("gathered_tensordict['a']:\n", gathered_tensordict["a"], end="\n\n")
