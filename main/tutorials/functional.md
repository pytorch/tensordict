Note

Go to the end
to download the full example code.

# Functional calls and ensembles with TensorDict

**Author**: [Vincent Moens](https://github.com/vmoens)

The parameters of a [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) form a tree of named tensors,
which a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) holds with the same structure. With
the parameters in a tensordict, one call replaces all the parameters of a
module, stacks the parameters of several models, or passes them to
[`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func). This tutorial is for readers who train ensembles of
models, or who swap the parameters of a model at run time.

What you will learn:

- How to extract the parameters of a module with
[`from_module()`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.from_module), and swap them with
[`to_module()`](../reference/generated/tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module).
- How to run an ensemble of models with [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) over a stack of
parameters.
- How to stack the parameters of several modules with
[`from_modules()`](../reference/generated/tensordict.TensorDictBase.from_modules.html#tensordict.TensorDictBase.from_modules), and how
[`EnsembleModule`](../reference/generated/tensordict.nn.EnsembleModule.html#tensordict.nn.EnsembleModule) does it for you.
- How to call [`torch.func.functional_call()`](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html#torch.func.functional_call) with these parameters.

Before you start, read [Getting, setting, and nesting keys](tensordict_keys.html#sphx-glr-tutorials-tensordict-keys-py) for nested
keys.

## TensorDict as a parameter container

[`from_module()`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.from_module) copies the references to the
parameters and buffers of a module into a nested
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict). The structure of the tensordict is the
structure of the module.

```
import torch
import torch.nn as nn
from tensordict import TensorDict

torch.manual_seed(0)
module = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1))
params = TensorDict.from_module(module)
print(params)
```

```
TensorDict(
 fields={
 0: TensorDict(
 fields={
 bias: Parameter(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Parameter(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 2: TensorDict(
 fields={
 bias: Parameter(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Parameter(shape=torch.Size([1, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

The [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) holds the same
[`Parameter`](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) objects as the module. Tensordict operations
apply to all of them at once. For example, this line makes a copy of all
the parameters, with zeros as values:

```
params_zero = params.detach().clone().zero_()
print("All zeros:", (params_zero == 0).all())
```

```
All zeros: True
```

## Swapping parameters with a context manager

[`to_module()`](../reference/generated/tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) replaces the parameters of a module
with the values of the tensordict. Used in a `with` block, it puts the
original parameters back at the end of the block.

```
x = torch.randn(5, 3)

with params_zero.to_module(module):
 y_zero = module(x)

print("Output with zeroed params:", y_zero)
assert (y_zero == 0).all()

y_original = module(x)
print("Output with original params:", y_original)
assert not (y_original == 0).all()
```

```
Output with zeroed params: tensor([[0.],
 [0.],
 [0.],
 [0.],
 [0.]], grad_fn=<AddmmBackward0>)
Output with original params: tensor([[-0.0624],
 [ 0.0916],
 [ 0.2135],
 [ 0.1337],
 [ 0.1770]], grad_fn=<AddmmBackward0>)
```

## Model ensembling with `torch.vmap`

You can stack parameter tensordicts like any tensordict. The stack has a
batch dimension, with one element per set of parameters.
[`vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) then runs the model with each set of parameters in one
vectorized call.

```
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
```

```
Stacked params batch_size: torch.Size([3])
Output shape: torch.Size([3, 5, 1])
```

## Stacking the parameters of several modules

An ensemble usually has several copies of a model, each with its own random
initialization. [`from_modules()`](../reference/generated/tensordict.TensorDictBase.from_modules.html#tensordict.TensorDictBase.from_modules) stacks the
parameters of several modules with the same structure into one tensordict.
The first dimension of the result is the index of the module:

```
modules = [nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 1)) for _ in range(3)]
ensemble_params = TensorDict.from_modules(*modules)
print("Ensemble params batch_size:", ensemble_params.batch_size)
```

```
Ensemble params batch_size: torch.Size([3])
```

Use one of the modules as the structure, and run it with each set of
parameters. `in_dims=(None, 0)` gives the same input `x` to every
model, and maps over the first dimension of the parameters:

```
x = torch.randn(5, 3)
y = torch.vmap(call, in_dims=(None, 0))(x, ensemble_params)
print("Output shape:", y.shape)
for i, member in enumerate(modules):
 torch.testing.assert_close(y[i], member(x))
```

```
Output shape: torch.Size([3, 5, 1])
```

To train the stacked parameters, pass `as_module=True`. The result is a
[`TensorDictParams`](../reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams), a tensordict that is also an
[`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), so that an optimizer can find its parameters
with `parameters()`.

For a [`TensorDictModule`](../reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule),
[`EnsembleModule`](../reference/generated/tensordict.nn.EnsembleModule.html#tensordict.nn.EnsembleModule) does all of this in one step. It
makes `num_copies` copies of the module with new random parameters, and
adds a leading dimension of size `num_copies` to its input and outputs:

```
from tensordict.nn import EnsembleModule, TensorDictModule

member = TensorDictModule(nn.Linear(3, 1), in_keys=["x"], out_keys=["y"])
ensemble = EnsembleModule(member, num_copies=4)
out = ensemble(TensorDict({"x": torch.randn(5, 3)}, batch_size=[5]))
print("EnsembleModule output batch_size:", out.batch_size)
```

```
EnsembleModule output batch_size: torch.Size([4, 5])
```

## Functional calls with `torch.func`

[`functional_call()`](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html#torch.func.functional_call) takes a `dict` of parameters, with the
names of a state dict as keys. `flatten_keys(".")` turns the nested keys of
the tensordict into these names, and `dict(...items())` turns the result
into a `dict`:

```
from torch.func import functional_call

flat_params = params.flatten_keys(".")
state_dict = dict(flat_params.items())
x = torch.randn(5, 3)
y = functional_call(module, state_dict, x)
print("functional_call output:", y.shape)
```

```
functional_call output: torch.Size([5, 1])
```

## Conclusion

You can now extract the parameters of a module into a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with `from_module`. `to_module` swaps
them for the length of a `with` block. You can stack parameters by
hand or with `from_modules`, and run the whole ensemble in one
[`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) call. [`EnsembleModule`](../reference/generated/tensordict.nn.EnsembleModule.html#tensordict.nn.EnsembleModule) builds such
an ensemble in one step. `flatten_keys(".")` gives the parameter names
that [`torch.func.functional_call()`](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html#torch.func.functional_call) expects.

## Further reading

- [torch.compile, torch.func, torch.export, and CUDA graphs](../compile.html#compile) explains how tensordicts work with [`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func),
including per-sample gradients with [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap).
- [Modules: TensorDictModule and tensordict.nn](../modules.html#modules-guide) explains [`TensorDictParams`](../reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams) and
the other modules of `tensordict.nn`.
- [Building models with TensorDictModule](tensordict_module.html#sphx-glr-tutorials-tensordict-module-py) shows how to build models
from [`TensorDictModule`](../reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule).
- [tensordict.nn](../reference/nn.html#api-nn) lists all the classes of `tensordict.nn`.

**Total running time of the script:** (0 minutes 0.009 seconds)

[`Download Jupyter notebook: functional.ipynb`](../_downloads/105b30183ad4a9afd272c38cb842a9b7/functional.ipynb)

[`Download Python source code: functional.py`](../_downloads/b040ef29505ca268795314134a7e184e/functional.py)

[`Download zipped: functional.zip`](../_downloads/38dc145a07c79455cee05162f5bf646a/functional.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)