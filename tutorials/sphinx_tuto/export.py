# -*- coding: utf-8 -*-

"""
Exporting tensordict modules
============================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

After you write a model with ``tensordict.nn``, you can export its
computational graph with :func:`torch.export.export`. The exported program
takes and returns plain tensors, and it does not need tensordict at run time.
Use it to run the model on other hardware (robots, drones, edge devices), or
to remove the dependency on tensordict.

What you will learn:

- How to call a ``tensordict.nn`` module with tensors instead of a
  :class:`~tensordict.TensorDict`.
- How to export the module with ``torch.export`` and save the result to a file.
- How to export a module that reads and writes nested keys.
- How to select the outputs of the model before you export it.

Before you start, read the :ref:`TensorDictModule <tensordictmodule>`
tutorial.

PyTorch also has an ONNX exporter. In recent PyTorch versions,
``torch.onnx.export(..., dynamo=True)`` is built on ``torch.export``. This
tutorial covers ``torch.export`` only.
"""

import tempfile
from pathlib import Path

import torch
from tensordict.nn import (
    NormalParamExtractor,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torch import nn

##################################################
# Designing the model
# -------------------
#
# Build a small network with ``tensordict.nn``. The network has:
#
# - A linear layer that maps the input to a hidden representation.
# - A ReLU activation.
# - A second linear layer that writes a latent representation.
#
# A :class:`~tensordict.nn.distributions.NormalParamExtractor` then splits the latent
# tensor into two outputs, ``"loc"`` and ``"scale"``.
#
model = Seq(
    # 1. A small network for embedding
    Mod(nn.Linear(3, 4), in_keys=["x"], out_keys=["hidden"]),
    Mod(nn.ReLU(), in_keys=["hidden"], out_keys=["hidden"]),
    Mod(nn.Linear(4, 4), in_keys=["hidden"], out_keys=["latent"]),
    # 2. Extracting params (splits into loc and scale)
    Mod(NormalParamExtractor(), in_keys=["latent"], out_keys=["loc", "scale"]),
)

##################################################
# Run the model with a tensor as a keyword argument, and look at the output:
#

torch.manual_seed(0)
x = torch.randn(1, 3)
print(model(x=x))

##################################################
# Called with a tensor, the model returns one tensor per output key: here
# ``"hidden"``, ``"latent"``, ``"loc"`` and ``"scale"``. For a large model,
# you usually need fewer outputs. The section "Selecting the outputs" below
# shows how to keep only some of them.
#
# Using ``torch.export`` with a ``TensorDictModule``
# --------------------------------------------------
#
# ``torch.export`` captures the graph of a module in a standard form, which
# does not depend on ``tensordict``. Its main entry point is
# :func:`~torch.export.export`, which returns an ``ExportedProgram``. This
# object has the following attributes:
#
# - ``graph_module``: the FX graph that ``export`` captured.
# - ``graph_signature``: the inputs and outputs of the graph.
# - ``module()``: a method that returns a callable, which you can use in
#   place of the original module.
#
# The model accepts positional and keyword arguments. This tutorial uses
# keyword arguments, because they name the input keys.

from torch.export import export

model_export = export(model, args=(), kwargs={"x": x})

##################################################
# Look at the module:
#
print("module:", model_export.module())

##################################################
# You call this module like the original module, and it returns the same
# tensors:
#

exported = model_export.module()
for out_model, out_exported in zip(model(x=x), exported(x=x)):
    torch.testing.assert_close(out_model, out_exported)

##################################################
# Export does not optimize the graph. The exported module runs the same tensor
# operations as the original module, one by one, without the tensordict reads
# and writes. To make it faster, compile it (see :ref:`compile`), or compile
# it ahead of time with AOTInductor (see "Further reading").
#
# Print the FX graph:
print(model_export.graph_module.print_readable(print_output=False))

##################################################
# Working with nested keys
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Nested keys are tuples of strings, such as ``("some", "key")``. A keyword
# argument must be a single string, so :class:`~tensordict.nn.dispatch`
# joins the parts of a nested key with an underscore (``"_"``). In the
# example below, the input key ``("some", "key")`` becomes the keyword
# argument ``some_key``.

model_nested = Seq(
    Mod(lambda x: x + 1, in_keys=[("some", "key")], out_keys=["hidden"]),
    Mod(lambda x: x - 1, in_keys=["hidden"], out_keys=[("some", "output")]),
).select_out_keys(("some", "output"))

model_nested_export = export(model_nested, args=(), kwargs={"some_key": x})
print("exported module with nested input:", model_nested_export.module())

##################################################
# The ``forward`` method of the exported module takes ``some_key``:

out = model_nested_export.module()(some_key=x)
torch.testing.assert_close(out, x)

##################################################
# The callable that ``module()`` returns is a plain Python callable. You can
# compile it with :func:`~torch.compile`.
#
# Saving the exported module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# ``torch.export`` has its own file format. Save the program with
# :func:`~torch.export.save` and load it with :func:`~torch.export.load`. By
# convention, the file name ends with ``".pt2"``:

path = Path(tempfile.mkdtemp()) / "model.pt2"
torch.export.save(model_export, path)
model_loaded = torch.export.load(path)
for out_model, out_loaded in zip(model(x=x), model_loaded.module()(x=x)):
    torch.testing.assert_close(out_model, out_loaded)

##################################################
# Selecting the outputs
# ---------------------
#
# By default, a ``tensordict.nn`` module keeps every intermediate value in its
# output. During training, you can log these values, or rebuild a
# distribution from its saved parameters instead of saving the
# :class:`~torch.distributions.distribution.Distribution` object. The extra memory is small
# during training, because autograd keeps these tensors for the backward pass
# anyway.
#
# An exported inference model usually needs one or two outputs. Select them
# before you export, so that the exported graph returns only these tensors.
# You can select outputs in three ways:
#
# 1. Pass the ``selected_out_keys`` keyword argument when you build the
#    :class:`~tensordict.nn.TensorDictSequential`.
# 2. Call the :meth:`~tensordict.nn.TensorDictModuleBase.select_out_keys` method.
#    It changes the ``out_keys`` attribute in place. To undo it, call
#    :meth:`~tensordict.nn.TensorDictModuleBase.reset_out_keys`.
# 3. Wrap the module in a :class:`~tensordict.nn.TensorDictSequential` that
#    selects the keys:
#
#    >>> module_filtered = Seq(module, selected_out_keys=["loc"])
#
# Select the ``"loc"`` output with the second method. With an ``x`` input,
# the model now returns one tensor, the ``"loc"`` output:

model.select_out_keys("loc")
print(model(x=x))

##################################################
# Export the model again. The graph now returns a single tensor:

model_export = export(model, args=(), kwargs={"x": x})
print("module:", model_export.module())

##################################################
# Conclusion
# ----------
#
# In this tutorial, you called a ``tensordict.nn`` model with tensors as
# keyword arguments, and you exported it with ``torch.export``. The exported
# program takes and returns plain tensors. You passed nested keys as keyword
# arguments joined with ``"_"``, saved the program to a ``.pt2`` file, and
# selected the outputs of the model before the export.
#
# Further reading
# ---------------
#
# - :ref:`compile` explains how tensordicts go through ``torch.export``,
#   including dynamic batch sizes.
# - :ref:`modules-guide` explains ``tensordict.nn`` modules and
#   :class:`~tensordict.nn.dispatch`.
# - The `torch.export documentation <https://pytorch.org/docs/stable/export>`_
#   and the `torch.export tutorial
#   <https://pytorch.org/tutorials/intermediate/torch_export_tutorial.html>`__
#   explain the exporter.
# - The `ONNX tutorials
#   <https://pytorch.org/tutorials/beginner/onnx/export_simple_model_to_onnx_tutorial.html>`_
#   explain the ONNX exporter.
# - To run PyTorch code on servers without Python, read the
#   `AOTInductor <https://pytorch.org/docs/main/torch.compiler_aot_inductor.html>`_
#   documentation.
