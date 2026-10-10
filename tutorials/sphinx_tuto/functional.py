"""
Functional calls and ensembles with TensorDict
==============================================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

The parameters of a :class:`~torch.nn.Module` form a tree of named tensors,
which a :class:`~tensordict.TensorDict` holds with the same structure. With
the parameters in a tensordict, one call replaces all the parameters of a
module, stacks the parameters of several models, or passes them to
:mod:`torch.func`. This tutorial is for readers who train ensembles of
models, or who swap the parameters of a model at run time.

What you will learn:

- How to extract the parameters of a module with
  :meth:`~tensordict.TensorDict.from_module`, and swap them with
  :meth:`~tensordict.TensorDict.to_module`.
- How to run an ensemble of models with :func:`torch.vmap` over a stack of
  parameters.
- How to stack the parameters of several modules with
  :meth:`~tensordict.TensorDict.from_modules`, and how
  :class:`~tensordict.nn.EnsembleModule` does it for you.
- How to call :func:`torch.func.functional_call` with these parameters.

Before you start, read :ref:`sphx_glr_tutorials_tensordict_keys.py` for nested
keys.
"""

##############################################################################
# TensorDict as a parameter container
# ------------------------------------
#
# :meth:`~tensordict.TensorDict.from_module` copies the references to the
# parameters and buffers of a module into a nested
# :class:`~tensordict.TensorDict`. The structure of the tensordict is the
# structure of the module.

import torch
import torch.nn as nn
from tensordict import TensorDict

torch.manual_seed(0)
module = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
params = TensorDict.from_module(module)
print(params)

##############################################################################
# The :class:`~tensordict.TensorDict` holds the same
# :class:`~torch.nn.parameter.Parameter` objects as the module. Tensordict operations
# apply to all of them at once. For example, this line makes a copy of all
# the parameters, with zeros as values:

params_zero = params.detach().clone().zero_()
print("All zeros:", (params_zero == 0).all())

##############################################################################
# Swapping parameters with a context manager
# -------------------------------------------
#
# :meth:`~tensordict.TensorDict.to_module` replaces the parameters of a module
# with the values of the tensordict. Used in a ``with`` block, it puts the
# original parameters back at the end of the block.

x = torch.randn(5, 3)

with params_zero.to_module(module):
    y_zero = module(x)

print("Output with zeroed params:", y_zero)
assert (y_zero == 0).all()

y_original = module(x)
print("Output with original params:", y_original)
assert not (y_original == 0).all()

##############################################################################
# Model ensembling with ``torch.vmap``
# -------------------------------------
#
# You can stack parameter tensordicts like any tensordict. The stack has a
# batch dimension, with one element per set of parameters.
# :func:`~torch.vmap` then runs the model with each set of parameters in one
# vectorized call.

params_ones = params.detach().clone().apply_(lambda t: t.fill_(1.0))
params_stack = torch.stack([params_zero, params_ones, params])

print("Stacked params batch_size:", params_stack.batch_size)


def call(x, td):
    with td.to_module(module):
        return module(x)


x = torch.randn(3, 5, 3)
y = torch.vmap(call)(x, params_stack)
print("Output shape:", y.shape)

assert (y[0] == 0).all()

##############################################################################
# Stacking the parameters of several modules
# ------------------------------------------
#
# An ensemble usually has several copies of a model, each with its own random
# initialization. :meth:`~tensordict.TensorDict.from_modules` stacks the
# parameters of several modules with the same structure into one tensordict.
# The first dimension of the result is the index of the module:

modules = [nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1)) for _ in range(3)]
ensemble_params = TensorDict.from_modules(*modules)
print("Ensemble params batch_size:", ensemble_params.batch_size)

##############################################################################
# Use one of the modules as the structure, and run it with each set of
# parameters. ``in_dims=(None, 0)`` gives the same input ``x`` to every
# model, and maps over the first dimension of the parameters:

x = torch.randn(5, 3)
y = torch.vmap(call, in_dims=(None, 0))(x, ensemble_params)
print("Output shape:", y.shape)
for i, member in enumerate(modules):
    torch.testing.assert_close(y[i], member(x))

##############################################################################
# To train the stacked parameters, pass ``as_module=True``. The result is a
# :class:`~tensordict.TensorDictParams`, a tensordict that is also an
# :class:`~torch.nn.Module`, so that an optimizer can find its parameters
# with ``parameters()``.
#
# For a :class:`~tensordict.nn.TensorDictModule`,
# :class:`~tensordict.nn.EnsembleModule` does all of this in one step. It
# makes ``num_copies`` copies of the module with new random parameters, and
# adds a leading dimension of size ``num_copies`` to its input and outputs:

from tensordict.nn import EnsembleModule, TensorDictModule

member = TensorDictModule(nn.Linear(3, 1), in_keys=["x"], out_keys=["y"])
ensemble = EnsembleModule(member, num_copies=4)
out = ensemble(TensorDict(x=torch.randn(5, 3), batch_size=[5]))
print("EnsembleModule output batch_size:", out.batch_size)

##############################################################################
# Functional calls with ``torch.func``
# --------------------------------------
#
# :func:`~torch.func.functional_call` takes a ``dict`` of parameters, with the
# names of a state dict as keys. ``flatten_keys(".")`` turns the nested keys of
# the tensordict into these names, and ``dict(...items())`` turns the result
# into a ``dict``:

from torch.func import functional_call

flat_params = params.flatten_keys(".")
state_dict = dict(flat_params.items())
x = torch.randn(5, 3)
y = functional_call(module, state_dict, x)
print("functional_call output:", y.shape)

##############################################################################
# Conclusion
# ----------
#
# You can now extract the parameters of a module into a
# :class:`~tensordict.TensorDict` with ``from_module``. ``to_module`` swaps
# them for the length of a ``with`` block. You can stack parameters by
# hand or with ``from_modules``, and run the whole ensemble in one
# :func:`torch.vmap` call. :class:`~tensordict.nn.EnsembleModule` builds such
# an ensemble in one step. ``flatten_keys(".")`` gives the parameter names
# that :func:`torch.func.functional_call` expects.
#
# Further reading
# ---------------
#
# - :ref:`compile` explains how tensordicts work with :mod:`torch.func`,
#   including per-sample gradients with :func:`torch.vmap`.
# - :ref:`modules-guide` explains :class:`~tensordict.TensorDictParams` and
#   the other modules of ``tensordict.nn``.
# - :ref:`sphx_glr_tutorials_tensordict_module.py` shows how to build models
#   from :class:`~tensordict.nn.TensorDictModule`.
# - :ref:`api-nn` lists all the classes of ``tensordict.nn``.
