"""
Functional Programming with TensorDict
=======================================
**Author**: `Vincent Moens <https://github.com/vmoens>`_

In this tutorial you will learn how to use :class:`~.TensorDict` for
functional-style programming with :class:`~torch.nn.Module`, including
parameter swapping, model ensembling with :func:`~torch.vmap`, and
functional calls with :func:`~torch.func.functional_call`.
"""

##############################################################################
# TensorDict as a parameter container
# ------------------------------------
#
# :meth:`~.TensorDict.from_module` extracts the parameters of a module into a
# nested :class:`~.TensorDict` whose structure mirrors the module hierarchy.

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore
import torch
import torch.nn as nn
from tensordict import TensorDict

module = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
params = TensorDict.from_module(module)
print(params)

##############################################################################
# The resulting :class:`~.TensorDict` holds the same
# :class:`~torch.nn.Parameter` objects as the module. We can manipulate them
# as a batch -- for example, zeroing all parameters at once:

params_zero = params.detach().clone().zero_()
print("All zeros:", (params_zero == 0).all())

##############################################################################
# Swapping parameters with a context manager
# -------------------------------------------
#
# :meth:`~.TensorDict.to_module` temporarily replaces the parameters of a
# module within a context manager. The original parameters are restored on
# exit.

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
# Because :class:`~.TensorDict` supports batching and stacking, we can stack
# multiple parameter configurations and use :func:`~torch.vmap` to run the
# model across all of them in a single vectorized call.

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
# Functional calls with ``torch.func``
# --------------------------------------
#
# :func:`~torch.func.functional_call` works with the state-dict extracted
# by :meth:`~.TensorDict.from_module`. Because ``from_module`` returns a
# :class:`~.TensorDict` with the same structure as a state-dict, we can
# convert it to a regular dict and pass it directly.

from torch.func import functional_call

flat_params = params.flatten_keys(".")
state_dict = dict(flat_params.items())
x = torch.randn(5, 3)
y = functional_call(module, state_dict, x)
print("functional_call output:", y.shape)

##############################################################################
# The combination of :meth:`~.TensorDict.from_module`,
# :meth:`~.TensorDict.to_module`, and :func:`~torch.vmap` makes it
# straightforward to do things like compute per-sample gradients, run
# model ensembles, or implement meta-learning inner loops -- all without
# leaving the standard PyTorch API.
