# -*- coding: utf-8 -*-
"""
Functionalizing TensorDictModule
================================
In this tutorial you will learn how to use :class:`~.TensorDictModule` in conjunction
with functorch to create functionlized modules.
"""
##############################################################################
# Before we take a look at the functional utilities in :mod:`tensordict.nn`, let us
# reintroduce one of the example modules from the :class:`~.TensorDictModule` tutorial.
#
# We'll create a simple module that has two linear layers, which share the input and
# return separate outputs.

import functorch
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule


class MultiHeadLinear(nn.Module):
    def __init__(self, in_1, out_1, out_2):
        super().__init__()
        self.linear_1 = nn.Linear(in_1, out_1)
        self.linear_2 = nn.Linear(in_1, out_2)

    def forward(self, x):
        return self.linear_1(x), self.linear_2(x)


##############################################################################
# We can now create a :class:`~.TensorDictModule` that will read the input from a key
# ``"a"``, and write to the keys ``"output_1"`` and ``"output_2"``.
splitlinear = TensorDictModule(
    MultiHeadLinear(3, 4, 10), in_keys=["a"], out_keys=["output_1", "output_2"]
)

##############################################################################
# Ordinarily we would use this module by simply calling it on a :class:`~.TensorDict`
# with the required input keys.

tensordict = TensorDict({"a": torch.randn(5, 3)}, batch_size=[5])
splitlinear(tensordict)
print(tensordict)


##############################################################################
# However, we can also use :func:`functorch.make_functional_with_buffers` in order to
# functionalise the module.
func, params, buffers = functorch.make_functional_with_buffers(splitlinear)
print(func(params, buffers, tensordict))

###############################################################################
# This can be used with the vmap operator. For example, we use 3 replicas of the
# params and buffers and execute a vectorized map over these for a single batch
# of data:

params_expand = [p.expand(3, *p.shape) for p in params]
buffers_expand = [p.expand(3, *p.shape) for p in buffers]
print(torch.vmap(func, (0, 0, None))(params_expand, buffers_expand, tensordict))

###############################################################################
# We can also use the native :func:`make_functional <tensordict.nn.make_functional>`
# function from :mod:`tensordict.nn``, which modifies the module to make it accept the
# parameters as regular inputs:

from tensordict.nn import make_functional

tensordict = TensorDict({"a": torch.randn(5, 3)}, batch_size=[5])

num_models = 10
model = TensorDictModule(nn.Linear(3, 4), in_keys=["a"], out_keys=["output"])
params = make_functional(model)
# we stack two groups of parameters to show the vmap usage:
params = torch.stack([params, params.apply(lambda x: torch.zeros_like(x))], 0)
result_td = torch.vmap(model, (None, 0))(tensordict, params)
print("the output tensordict shape is: ", result_td.shape)
