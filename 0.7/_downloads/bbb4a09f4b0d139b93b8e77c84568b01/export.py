# -*- coding: utf-8 -*-

"""
Exporting tensordict modules
============================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

Prerequisites
~~~~~~~~~~~~~

Reading the :ref:`TensorDictModule <tensordictmodule>` tutorial is preferable to fully benefit from this tutorial.

Once a module has been written using ``tensordict.nn``, it is often useful to isolate the computational graph and export
that graph. The goal of this may be to execute the model on hardware (e.g., robots, drones, edge devices) or eliminate
the dependency on tensordict altogether.

PyTorch provides multiple methods for exporting modules, including ``onnx`` and ``torch.export``, both of which are
compatible with ``tensordict``.

In this short tutorial, we will see how one can use ``torch.export`` to isolate the computational graph of a model.
``torch.onnx`` support follows the same logic.

Key learnings
~~~~~~~~~~~~~

- Executing a ``tensordict.nn`` module without :class:`~tensordict.TensorDict` inputs;
- Selecting the output(s) of a model;
- Handling stochstic models;
- Exporting such model using `torch.export`;
- Saving the model to a file;
- Isolating the pytorch model;


"""
import time

import torch
from tensordict.nn import (
    InteractionType,
    NormalParamExtractor,
    ProbabilisticTensorDictModule as Prob,
    set_interaction_type,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torch import distributions as dists, nn

##################################################
# Designing the model
# -------------------
#
# In many applications, it is useful to work with stochastic models, i.e., models that output a variable that is not
# deterministically defined but that is sampled according to a parametric distribution. For instance, generative AI
# models will often generate different outputs when the same input if provided, because they sample the output based
# on a distribution which parameters are defined by the input.
#
# The ``tensordict`` library deals with this through the :class:`~tensordict.nn.ProbabilisticTensorDictModule` class.
# This primitive is built using a distribtion class (:class:`~torch.distributions.Normal` in our case) and an indicator
# of the input keys that will be used at execution time to build that distribution.
#
# The network we are building is therefore going to be the combination of three main components:
#
# - A network mapping the input to a latent parameter;
# - A :class:`tensordict.nn.NormalParamExtractor` module splitting the input in a location `"loc"` and `"scale"`
#   parameters to be passed to the ``Normal`` distrbution;
# - A distribution constructor module.
#
model = Seq(
    # 1. A small network for embedding
    Mod(nn.Linear(3, 4), in_keys=["x"], out_keys=["hidden"]),
    Mod(nn.ReLU(), in_keys=["hidden"], out_keys=["hidden"]),
    Mod(nn.Linear(4, 4), in_keys=["hidden"], out_keys=["latent"]),
    # 2. Extracting params
    Mod(NormalParamExtractor(), in_keys=["latent"], out_keys=["loc", "scale"]),
    # 3. Probabilistic module
    Prob(
        in_keys=["loc", "scale"],
        out_keys=["sample"],
        distribution_class=dists.Normal,
    ),
)

##################################################
# Let us run this model and see what the output looks like:
#

x = torch.randn(1, 3)
print(model(x=x))

##################################################
# As expected, running the model with a tensor input returns as many tensors as the module's output keys! For large
# models, this can be quite annoying and wasteful. Later, we will see how we can limit the number of outputs of the
# model to deal with this issue.
#
# Using ``torch.export`` with a ``TensorDictModule``
# --------------------------------------------------
#
# Now that we have successfully built our model, we would like to extract its computational graph in a single object that
# is independent of ``tensordict``. ``torch.export`` is a PyTorch module dedicated to isolate the graph of a module and
# represent it in a standardized way. Its main entry point is :func:`~torch.export.export` which returns a ``ExportedProgram``
# object. In turn, this object has several attributes of interest that we will explore below: a ``graph_module``,
# which represents the FX graph captured by ``export``, a ``graph_signature`` with input, outputs etc of the graph,
# and finally a ``module()`` that returns a callable that can be used in-place of the original module.
#
# Although our module accepts both args and kwargs, we will focus on its usage with kwargs as this is clearer.

from torch.export import export

model_export = export(model, args=(), kwargs={"x": x})

##################################################
# Let us look at the module:
#
print("module:", model_export.module())

##################################################
# This module can be run exactly like our original module (with a lower overhead):
#

t0 = time.time()
model(x=x)
print(f"Time for TDModule: {(time.time() - t0) * 1e6: 4.2f} micro-seconds")
exported = model_export.module()

# Exported version
t0 = time.time()
exported(x=x)
print(f"Time for exported module: {(time.time() - t0) * 1e6: 4.2f} micro-seconds")

##################################################
# and the FX graph:
print("fx graph:", model_export.graph_module.print_readable())

##################################################
# Working with nested keys
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Nested keys are a core feature of the tensordict library, and being able to export modules that read and write
# nested entries is therefore an important feature to support.
# Because keyword arguments must be regualar strings, it is not possible for :class:`~tensordict.nn.dispatch` to work
# directly with them. Instead, ``dispatch`` will unpack nested keys joined with a regular underscore (`"_"`), as the
# following example shows.

model_nested = Seq(
    Mod(lambda x: x + 1, in_keys=[("some", "key")], out_keys=["hidden"]),
    Mod(lambda x: x - 1, in_keys=["hidden"], out_keys=[("some", "output")]),
).select_out_keys(("some", "output"))

model_nested_export = export(model_nested, args=(), kwargs={"some_key": x})
print("exported module with nested input:", model_nested_export.module())


##################################################
# Note that the callable returned by `module()` is a pure python callable that can be in turn compiled using
# :func:`~torch.compile`.
#
# Saving the exported module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``torch.export`` has its own serialization protocol, :func:`~torch.export.save` and :func:`~torch.export.load`.
# Conventionally, the `".pt2"` extension is to be used:
#
#   >>> torch.export.save(model_export, "model.pt2")
#
# Selecting the outputs
# ---------------------
#
# Recall that the ``tensordict.nn`` is to keep every intermediate value in the output, unless the user specifically asks
# for only a specific value. During training, this can be very useful: one can easily log intermediate values of the
# graph, or use them for other purposes (e.g., reconstruct a distribution based on its saved parameters, rather than
# saving the :class:`~torch.distributions.Distribution` object itself). One could also argue that, during training, the
# impact on memory of registering intermediate values is negligeable since they are part of the computational graph
# used by ``torch.autograd`` to compute the parameter gradients.
#
# During inference, though, we most likely are only interested in the final sample of the model.
# Because we want to extract the model for usages that are independent of the ``tensordict`` library, it makes sense to
# isolate the only output we desire.
# To do this, we have several options:
#
# 1. Build the :meth:`~tensordict.nn.TensorDictSequential` with the ``selected_out_keys`` keyword argument, which will
#    induce the selection of the desired entries during calls to the module;
# 2. Using the :meth:`~tensordict.nn.TensorDictModule.select_out_keys` method, which will modify the ``out_keys``
#    attribute in-place (this can be reverted through :meth:`~tensordict.nn.TensorDictModule.reset_out_keys`).
# 3. Wrap the existing instance in a :meth:`~tensordict.nn.TensorDictSequential` that will filter out the unwanted keys:
#
#     >>> module_filtered = Seq(module, selected_out_keys=["sample"])
#
# Let us test the model after selecting its output keys.
# When an `x` input is provided, we expect our model to output a single tensor corresponding to a sample of the
# distribution:

model.select_out_keys("sample")
print(model(x=x))

##################################################
# We see that the output is now a single tensor, corresponding to the sample of the distribution.
# We can create a new exported graph from this. Its computational graph should be simplified:

model_export = export(model, args=(), kwargs={"x": x})
print("module:", model_export.module())

##################################################
# Controlling the Sampling Strategy
# ---------------------------------
#
# We have not yet discussed how the :class:`~tensordict.nn.ProbabilisticTensorDictModule` samples from the distribution.
# By sampling, we mean obtaining a value within the space defined by the distribution according to a specific strategy.
# For instance, one may desire to get stochastic samples during training but deterministic samples (e.g., the mean or
# the mode) at inference time. To address this, ``tensordict`` utilizes the :class:`~tensordict.nn.set_interaction_type`
# decorator and context manager, which accepts ``InteractionType`` Enum inputs:
#
#   >>> with set_interaction_type(InteractionType.MEAN):
#   ...     output = module(input)  # takes the input of the distribution, if ProbabilisticTensorDictModule is invoked
#
# The default ``InteractionType`` is ``InteractionType.DETERMINISTIC``, which, if not implemented directly, is either
# the mean of distributions with a real domain, or the mode of distributions with a discrete domain. This default value
# can be changed using the ``default_interaction_type`` keyword argument of ``ProbabilisticTensorDictModule``.
#
# Let us recap: to control the sampling strategy of our network, we can either define a default sampling strategy in the
# constructor, or override it at runtime through the ``set_interaction_type`` context manager.
#
# As we can see from the following example, ``torch.export`` respond correctly the usage of the decorator: if we ask for
# a random sample, the output is different than if we ask for the mean:
#

with set_interaction_type(InteractionType.RANDOM):
    model_export = export(model, args=(), kwargs={"x": x})
    print(model_export.module())

with set_interaction_type(InteractionType.MEAN):
    model_export = export(model, args=(), kwargs={"x": x})
    print(model_export.module())

##################################################
# This is all you need to know to use ``torch.export``. Please refer to the
# `official documentation <https://pytorch.org/docs/stable/export>`_ for more info.
#
# Next steps and further reading
# ------------------------------
#
# - Check the ``torch.export`` tutorial, available `here <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`__;
# - ONNX support: check the `ONNX tutorials <https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html>`_
#   to learn more about this feature. Exporting to ONNX is very similar to `torch.export` explained here.
# - For deployment of PyTorch code on servers without python environment, check the
#   `AOTInductor <https://pytorch.org/docs/main/torch.compiler_aot_inductor.html>`_ documentation.
#
