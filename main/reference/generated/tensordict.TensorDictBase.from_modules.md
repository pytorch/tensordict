# TensorDictBase.from_modules

*classmethod*TensorDictBase.from_modules(**modules*, *as_module: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lock: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *lazy_stack: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *expand_identical: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Stacks the parameters and buffers of several modules in one tensordict.

The result has one batch dimension, with one element per module, and
the structure that [`from_module()`](tensordict.TensorDictBase.from_module.html#tensordict.TensorDictBase.from_module) gives. Use it to run an
ensemble of modules that have the same structure with
[`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) and [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) (see the example).

The default, dense result holds new parameters and buffers: the
stacks of those of the modules, with the same `requires_grad`. Pass
these new parameters to the optimizer: gradients reach them, not the
parameters of the modules. With `lazy_stack=True`, the result is a
lazy stack of the parameters of the modules, which receive the
gradients.

Parameters:

***modules** (*nn.Module*) - the modules to get the parameters from. With
a dense stack, they must have the same parameters and buffers,
with the same shapes.

Keyword Arguments:

- **as_module** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns a
`TensorDictParams`, which can be stored
as a submodule of a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Exclusive with
`lazy_stack`. Defaults to `False`.
- **lock** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the result is locked (see
[Locking and temporary changes](../../concepts/keys.html#keys-locking)). Defaults to `True`.
- **use_state_dict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, reads the
`state_dict()` of each module, so that its state-dict hooks
run. Defaults to `False`.
- **lazy_stack** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

if `True`, returns a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) of the parameters of
the modules; if `False`, stacks them into new parameters.
Exclusive with `as_module`, and not compatible with lazy
modules. Defaults to `False`.

Warning

A lazy stack stacks the parameters each time an entry is
read with [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get), which costs memory (`N` times
the size of the parameters, more if a graph is built) and
time, and the optimizer holds `N` times more parameters.
Use `lazy_stack=True` only when you need to train the
parameters of the modules themselves.
- **expand_identical** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a parameter that is
the same object in every module is expanded instead of
stacked, so that the result shares its memory. Ignored when
`lazy_stack=True`. Defaults to `False`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `lazy_stack=True` and `as_module=True`, or if
 `lazy_stack=True` and a module has uninitialized (lazy)
 parameters.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> _ = torch.manual_seed(0)
>>> modules = [nn.Linear(3, 4) for _ in range(2)]
>>> params = TensorDict.from_modules(*modules)
>>> params.batch_size, params["weight"].shape
(torch.Size([2]), torch.Size([2, 4, 3]))
```

Run the two modules on one input with [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap), through a
module on the `"meta"` device that holds no data. The gradients
reach the stacked parameters, not those of the modules:

```
>>> empty_module = nn.Linear(3, 4, device="meta")
>>> def exec_module(params, x):
... with params.to_module(empty_module):
... return empty_module(x)
>>> x = torch.randn(3)
>>> y = torch.vmap(exec_module, (0, None))(params, x)
>>> y.shape
torch.Size([2, 4])
>>> y.sum().backward()
>>> params["weight"].grad is not None, modules[0].weight.grad is None
(True, True)
```

With `lazy_stack=True`, the gradients reach the parameters of
the modules:

```
>>> params = TensorDict.from_modules(*modules, lazy_stack=True)
>>> type(params).__name__
'LazyStackedTensorDict'
>>> y = torch.vmap(exec_module, (0, None))(params, x)
>>> y.sum().backward()
>>> modules[0].weight.grad is not None
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `from_modules`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict