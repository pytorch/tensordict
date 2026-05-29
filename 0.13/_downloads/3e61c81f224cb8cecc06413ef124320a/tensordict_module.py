"""
TensorDictModule
================

.. _tensordictmodule:

**Author**: `Nicolas Dufour <https://github.com/nicolas-dufour>`_, `Vincent Moens <https://github.com/vmoens>`_

In this tutorial you will learn how to use :class:`~.TensorDictModule` and
:class:`~.TensorDictSequential` to create generic and reusable modules that can accept
:class:`~.TensorDict` as input.

"""

##############################################################################
#
# For a convenient usage of the :class:`~.TensorDict` class with :class:`~torch.nn.Module`,
# :mod:`tensordict` provides an interface between the two named :class:`~tensordict.nn.TensorDictModule`.
#
# The :class:`~tensordict.nn.TensorDictModule` class is an :class:`~torch.nn.Module` that takes a
# :class:`~tensordict.TensorDict` as input when called. It will read a sequence of input keys, pass them to the wrapped
# module or function as input, and write the outputs in the same tensordict after completing the execution.
#
# It is up to the user to define the keys to be read as input and output.
#

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

###############################################################################
#
# Simple example: coding a recurrent layer
# ----------------------------------------
#
# The simplest usage of :class:`~tensordict.nn.TensorDictModule` is exemplified below.
# If at first it may look like using this class introduces an unwated level of complexity, we will see
# later on that this API enables users to programatically concatenate modules together, cache values
# in between modules or programmatically build one.
# One of the simplest examples of this is a recurrent module in an architecture like ResNet, where the input of the
# module is cached and added to the output of a tiny multi-layered perceptron (MLP).
#
# To start, let's first consider we you would chunk an MLP, and code it using :mod:`tensordict.nn`.
# The first layer of the stack would presumably be a :class:`~torch.nn.Linear` layer, taking an entry as input
# (let us name it `x`) and outputting another entry (which we will name `y`).
#
# To feed to our module, we have a :class:`~tensordict.TensorDict` instance with a single entry,
# ``"x"``:

tensordict = TensorDict(
    x=torch.randn(5, 3),
    batch_size=[5],
)

###############################################################################
# Now, we build our simple module using :class:`tensordict.nn.TensorDictModule`. By default, this class writes in the
# input tensordict in-place (meaning that entries are written in the same tensordict as the input, not that entries
# are overwritten in-place!), such that we don't need to explicitly indicate what the output is:
#
linear0 = TensorDictModule(nn.Linear(3, 128), in_keys=["x"], out_keys=["linear0"])
linear0(tensordict)

assert "linear0" in tensordict

###############################################################################
#
# If the module outputs multiple tensors (or tensordicts!) their entries must be passed to
# :class:`~tensordict.nn.TensorDictModule` in the right order.
#
# Support for Callables
# ~~~~~~~~~~~~~~~~~~~~~
#
# When designing a model, it often happens that you want to incorporate an arbitrary non-parametric function into
# the network. For instance, you may wish to permute the dimensions of an image when it is passed to a convolutional network
# or a vision transformer, or divide the values by 255.
# There are several ways to do this: you could use a `forward_hook`, for example, or design a new
# :class:`~torch.nn.Module` that performs this operation.
#
# :class:`~tensordict.nn.TensorDictModule` works with any callable, not just modules, which makes it easy to
# incorporate arbitrary functions into a module. For instance, let's see how we can integrate the ``relu`` activation
# function without using the :class:`~torch.nn.ReLU` module:

relu0 = TensorDictModule(torch.relu, in_keys=["linear0"], out_keys=["relu0"])

###############################################################################
#
# Stacking modules
# ~~~~~~~~~~~~~~~~
#
# Our MLP isn't made of a single layer, so we now need to add another layer to it.
# This layer will be an activation function, for instance :class:`~torch.nn.ReLU`.
# We can stack this module and the previous one using :class:`~tensordict.nn.TensorDictSequential`.
#
# .. note:: Here comes the true power of ``tensordict.nn``: unlike :class:`~torch.nn.Sequential`,
#   :class:`~tensordict.nn.TensorDictSequential` will keep in memory all the previous inputs and outputs
#   (with the possibility to filter them out afterwards), making it easy to have complex network structures
#   built on-the-fly and programmatically.
#

block0 = TensorDictSequential(linear0, relu0)

block0(tensordict)
assert "linear0" in tensordict
assert "relu0" in tensordict

###############################################################################
# We can repeat this logic to get a full MLP:
#

linear1 = TensorDictModule(nn.Linear(128, 128), in_keys=["relu0"], out_keys=["linear1"])
relu1 = TensorDictModule(nn.ReLU(), in_keys=["linear1"], out_keys=["relu1"])
linear2 = TensorDictModule(nn.Linear(128, 3), in_keys=["relu1"], out_keys=["linear2"])
block1 = TensorDictSequential(linear1, relu1, linear2)

###############################################################################
# Multiple input keys
# ~~~~~~~~~~~~~~~~~~~
#
# The last step of the residual network is to add the input to the output of the last linear layer.
# No need to write a special :class:`~torch.nn.Module` subclass for this! :class:`~tensordict.nn.TensorDictModule`
# can be used to wrap simple functions too:

residual = TensorDictModule(
    lambda x, y: x + y, in_keys=["x", "linear2"], out_keys=["y"]
)

###############################################################################
# And we can now put together ``block0``, ``block1`` and ``residual`` for a fully fleshed residual block:

block = TensorDictSequential(block0, block1, residual)
block(tensordict)
assert "y" in tensordict

###############################################################################
# A genuine concern may be the accumulation of entries in the tensordict used as input: in some cases (e.g., when
# gradients are required) intermediate values may be cached anyway, but this isn't always the case and it can be useful
# to let the garbage collector know that some entries can be discarded. :class:`tensordict.nn.TensorDictModuleBase` and
# its subclasses (including :class:`tensordict.nn.TensorDictModule` and :class:`tensordict.nn.TensorDictSequential`)
# have the option of seeing their output keys filtered after execution. To do this, just call the
# :class:`tensordict.nn.TensorDictModuleBase.select_out_keys` method. This will update the module in-place and all the
# unwanted entries will be discarded:

block.select_out_keys("y")

tensordict = TensorDict(x=torch.randn(1, 3), batch_size=[1])
block(tensordict)
assert "y" in tensordict

assert "linear1" not in tensordict

###############################################################################
# However, the input keys are preserved:
assert "x" in tensordict

###############################################################################
# As a side note, ``selected_out_keys`` may also be passed to :class:`tensordict.nn.TensorDictSequential` to avoid
# calling this method separately.
#
# Using `TensorDictModule` without tensordict
# -------------------------------------------
#
# The opportunity offered by :class:`tensordict.nn.TensorDictSequential` to build complex architectures on-the-go
# does not mean that one necessarily has to switch to tensordict to represent the data. Thanks to
# :class:`~tensordict.nn.dispatch`, modules from `tensordict.nn` support arguments and keyword arguments that match the
# entry names too:

x = torch.randn(1, 3)
y = block(x=x)
assert isinstance(y, torch.Tensor)

###############################################################################
# Under the hood, :class:`~tensordict.nn.dispatch` rebuilds a tensordict, runs the module and then deconstructs it.
# This may cause some overhead but, as we will see just after, there is a solution to get rid of this.
#
# Runtime
# -------
#
# :class:`tensordict.nn.TensorDictModule` and :class:`tensordict.nn.TensorDictSequential` do incur some overhead when
# executed, as they need to read and write from a tensordict. However, we can greatly reduce this overhead by using
# :func:`~torch.compile`. For this, let us compare the three versions of this code with and without compile:


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear0 = nn.Linear(3, 128)
        self.relu0 = nn.ReLU()
        self.linear1 = nn.Linear(128, 128)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(128, 3)

    def forward(self, x):
        y = self.linear0(x)
        y = self.relu0(y)
        y = self.linear1(y)
        y = self.relu1(y)
        return self.linear2(y) + x


print("Without compile")
x = torch.randn(256, 3)
block_notd = ResidualBlock()
block_tdm = TensorDictModule(block_notd, in_keys=["x"], out_keys=["y"])
block_tds = block

from torch.utils.benchmark import Timer

print(
    f"Regular: {Timer('block_notd(x=x)', globals=globals()).adaptive_autorange().median * 1_000_000: 4.4f} us"
)
print(
    f"TDM: {Timer('block_tdm(x=x)', globals=globals()).adaptive_autorange().median * 1_000_000: 4.4f} us"
)
print(
    f"Sequential: {Timer('block_tds(x=x)', globals=globals()).adaptive_autorange().median * 1_000_000: 4.4f} us"
)

print("Compiled versions")
block_notd_c = torch.compile(block_notd, mode="reduce-overhead")
for _ in range(5):  # warmup
    block_notd_c(x)
print(
    f"Compiled regular: {Timer('block_notd_c(x=x)', globals=globals()).adaptive_autorange().median * 1_000_000: 4.4f} us"
)
block_tdm_c = torch.compile(block_tdm, mode="reduce-overhead")
for _ in range(5):  # warmup
    block_tdm_c(x=x)
print(
    f"Compiled TDM: {Timer('block_tdm_c(x=x)', globals=globals()).adaptive_autorange().median * 1_000_000: 4.4f} us"
)
block_tds_c = torch.compile(block_tds, mode="reduce-overhead")
for _ in range(5):  # warmup
    block_tds_c(x=x)
print(
    f"Compiled sequential: {Timer('block_tds_c(x=x)', globals=globals()).adaptive_autorange().median * 1_000_000: 4.4f} us"
)

###############################################################################
# As one can see, the onverhead introduced by :class:`~tensordict.nn.TensorDictSequential` has been completely resolved.
#
# Do's and don't with TensorDictModule
# ------------------------------------
#
# - Don't use :class:`~torch.nn.Sequence` around modules from :mod:`tensordict.nn`. It would break the input/output
#   key structure.
#   Always try to rely on :class:`~tensordict.nn:TensorDictSequential` instead.
#
# - Don't assign the output tensordict to a new variable, as the output tensordict is just the input modified in-place.
#   Assigning a new variable name isn't strictly prohibited, but it means that you may wish for both of them to disappear
#   when one is deleted, when in fact the garbage collector will still see the tensors in the workspace and the no memory
#   will be freed:
#
#   .. code-block::
#
#     >>> tensordict = module(tensordict)  # ok!
#     >>> tensordict_out = module(tensordict)  # don't!
#
# Working with distributions: :class:`~tensordict.nn.ProbabilisticTensorDictModule`
# ---------------------------------------------------------------------------------
#
# :class:`~tensordict.nn.ProbabilisticTensorDictModule` is a non-parametric module representing a
# probability distribution. Distribution parameters are read from tensordict
# input, and the output is written to an output tensordict. The output is
# sampled given some rule, specified by the input ``default_interaction_type``
# argument and the :func:`~tensordict.nn.interaction_type` global function. If they conflict,
# the context manager precedes.
#
# It can be wired together with a :class:`~tensordict.nn.TensorDictModule` that returns
# a tensordict updated with the distribution parameters using
# :class:`~tensordict.nn.ProbabilisticTensorDictSequential`. This is a special case of
# :class:`~tensordict.nn.TensorDictSequential` whose last layer is a
# :class:`~tensordict.nn.ProbabilisticTensorDictModule` instance.
#
# :class:`~tensordict.nn.ProbabilisticTensorDictModule` is responsible for constructing the
# distribution (through the :meth:`~tensordict.nn.ProbabilisticTensorDictModule.get_dist` method) and/or
# sampling from this distribution (through a regular `forward` call to the module). The same
# :meth:`~tensordict.nn.ProbabilisticTensorDictModule.get_dist` method is exposed within
# :class:`~tensordict.nn.ProbabilisticTensorDictSequential`.
#
# One can find the parameters in the output tensordict as well as the log
# probability if needed.

from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
)
from tensordict.nn.distributions import NormalParamExtractor
from torch import distributions as dist

td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3])
net = torch.nn.GRUCell(4, 8)
net = TensorDictModule(net, in_keys=["input", "hidden"], out_keys=["hidden"])
extractor = NormalParamExtractor()
extractor = TensorDictModule(extractor, in_keys=["hidden"], out_keys=["loc", "scale"])
td_module = ProbabilisticTensorDictSequential(
    net,
    extractor,
    ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=dist.Normal,
        return_log_prob=True,
    ),
)
print(f"TensorDict before going through module: {td}")
td_module(td)
print(f"TensorDict after going through module now as keys action, loc and scale: {td}")

###############################################################################
# Conclusion
# ----------
#
# We have seen how `tensordict.nn` can be used to dynamically build complex neural architectures on-the-fly.
# This opens the possibility of building pipelines that are oblivious to the model signature, i.e., write generic codes
# that use networks with an arbitrary number of inputs or outputs in a flexible manner.
#
# We have also seen how :class:`~tensordict.nn.dispatch` enables to use `tensordict.nn` to build such networks and use
# them without recurring to :class:`~tensordict.TensorDict` directly. Thanks to :func:`~torch.compile`, the overhead
# introduced by :class:`tensordict.nn.TensorDictSequential` can be completely removed, leaving users with a neat,
# tensordict-free version of their module.
#
# In the next tutorial, we will be seeing how ``torch.export`` can be used to isolate a module and export it.
#

# sphinx_gallery_start_ignore
import time

time.sleep(3)
# sphinx_gallery_end_ignore
