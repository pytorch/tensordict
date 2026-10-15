# TensorDictBase.to_module

TensorDictBase.to_module(*module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_swap: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *swap_dest=None*, *use_state_dict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *preserve_module_state: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *memo=None*)

Writes the content of the tensordict into the parameters and buffers of a module, recursively.

The keys of the tensordict must match the names of the module
attributes, as [`from_module()`](tensordict.TensorDictBase.from_module.html#tensordict.TensorDictBase.from_module) gives them. A nested key is a
submodule: `("layers", "0", "weight")` is `module.layers[0].weight`.

`to_module` can also be used as a context manager, to populate a
module with other parameters for the duration of a `with` block (see
the second example). When the block exits, the original parameters are
written back, also if the block raises an exception.

Parameters:

**module** (*nn.Module*) - a module to write the parameters into.

Keyword Arguments:

- **inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the parameters or tensors
in the module are updated in-place. Defaults to `None`, which
behaves like `False`. Cannot be passed together with
`use_state_dict=True`.
- **return_swap** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the old parameter configuration
will be returned. Defaults to `True`.
- **swap_dest** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - if `return_swap` is `True`,
the tensordict where the swap should be written.
- **memo** - deprecated. Passing it raises `RuntimeError`.
- **use_state_dict** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, state-dict API will be
used to load the parameters (including the state-dict hooks).
Defaults to `False`.
- **non_blocking** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and this copy is between
different devices, the copy may occur asynchronously with respect
to the host.
- **preserve_module_state** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - 

if `True`, existing
`Parameter` and buffer registrations are
preserved when writing tensor leaves to `module`: parameters
remain parameters with their original `requires_grad` value,
and buffers remain registered buffers. Leaves that are not
parameters but require gradients, or that are batched by
[`vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap), are registered as they are, so that
gradients still reach them. If `False`, tensor
leaves are written with the historical replacement semantics,
which may deregister an existing parameter when the source leaf
is not an `Parameter`. Defaults to `True`.
Pass `False` to retain the historical replacement behavior.

Changed in version 0.15: `preserve_module_state=None`, deprecated in 0.14, is no
longer accepted.

Returns:

A tensordict containing the values from the module if `return_swap` is `True`, `None` otherwise.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> module = nn.Linear(3, 4)
>>> params = TensorDict.from_module(module)
>>> zeros = params.data.clone().zero_()
>>> old_params = zeros.to_module(module)
>>> assert (module.weight == 0).all()
>>> _ = old_params.to_module(module) # restores the original values
>>> assert (module.weight != 0).any()
```

Use the tensordict as a context manager to make a functional call:

```
>>> zeros = params.data * 0
>>> with zeros.to_module(module):
... y = module(torch.ones(2, 3)) # runs with zeroed parameters
>>> assert (y == 0).all()
>>> # When the block exits, the original parameters are written back.
>>> assert (module.weight != 0).any()
```

> The original parameters are written back if the block raises an
> exception too:

```
>>> try:
... with zeros.to_module(module):
... raise ValueError("stop")
... except ValueError:
... pass
>>> assert (module.weight != 0).any()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `to_module`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict