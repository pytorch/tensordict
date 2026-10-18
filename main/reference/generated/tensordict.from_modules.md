# from_modules

tensordict.from_modules(**modules*, *as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lazy_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *expand_identical: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Stacks the parameters and buffers of several modules in one tensordict.

The result has one batch dimension, with one element per module. Use it to
run an ensemble of modules that have the same structure with
[`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) and [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module). Same
as [`tensordict.TensorDict.from_modules()`](tensordict.TensorDictBase.from_modules.html#tensordict.TensorDictBase.from_modules): see that page for the
difference between a dense and a lazy result, and for a full ensemble
example.

Parameters:

***modules** ([*Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - the modules to get the parameters from.

Keyword Arguments:

- **as_module** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a
`TensorDictParams` instance is returned.
Defaults to `False`.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the resulting tensordict is
locked. Defaults to `True`.
- **use_state_dict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the state-dict of each
module is used. Defaults to `False`.
- **lazy_stack** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the parameters are stacked
in a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) that keeps the
original parameters. If `False`, they are stacked into new
parameters. Exclusive with `as_module`. Defaults to `False`.
- **expand_identical** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a parameter that is
the same object in every module is expanded instead of stacked.
Ignored when `lazy_stack=True`. Defaults to `False`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with batch size `[len(modules)]`,
a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) if `lazy_stack=True`,
or a `TensorDictParams` if `as_module=True`.

Examples

```
>>> import torch
>>> from torch import nn
>>> import tensordict
>>> modules = [nn.Linear(3, 4) for _ in range(2)]
>>> params = tensordict.from_modules(*modules)
>>> params.batch_size
torch.Size([2])
>>> params["weight"].shape
torch.Size([2, 4, 3])
```