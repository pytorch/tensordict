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

- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies the values into the
existing parameters and buffers of the module, which keep
their identity. Otherwise the leaves of the tensordict replace
them. Defaults to `None`, which behaves like `False`.
Cannot be passed with `use_state_dict=True`.
- **return_swap** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns the parameters
and buffers that the module held before the call. Defaults to
`True`.
- **swap_dest** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - with `return_swap`, a
tensordict to write the old values into and to return.
Defaults to `None`, which returns a new tensordict.
- **use_state_dict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, loads the values
through the state-dict API, so that the module's state-dict
hooks run. Defaults to `False`.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.
- **preserve_module_state** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

if `True`, existing
`Parameter` and buffer registrations are
preserved when writing tensor leaves to `module`: parameters
remain parameters with their original `requires_grad` value,
and buffers remain registered buffers. Leaves that are not
parameters but require gradients, or that are batched by
[`vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap), are registered as they are, so that
gradients still reach them. If `False`, a leaf that is not a
`Parameter` replaces the attribute as it is,
which can deregister an existing parameter (the historical
behavior). Defaults to `True`.

Changed in version 0.15: `preserve_module_state=None`, deprecated in 0.14, is no
longer accepted.
- **memo** - no longer accepted: passing it raises a `RuntimeError`.

Returns:

A tensordict of the values that the module held before the call
(or `swap_dest`) if `return_swap` is `True`, `None`
otherwise.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> module = nn.Linear(3, 4)
>>> params = TensorDict.from_module(module)
>>> zeros = params.data.clone().zero_()
>>> old_params = zeros.to_module(module)
>>> bool((module.weight == 0).all())
True
>>> _ = old_params.to_module(module) # restores the original values
>>> bool((module.weight != 0).any())
True
```

Use the tensordict as a context manager to make a functional call.
When the block exits, the original parameters are written back:

```
>>> with zeros.to_module(module):
... y = module(torch.ones(2, 3)) # runs with zeroed parameters
>>> bool((y == 0).all())
True
>>> bool((module.weight != 0).any())
True
```

The original parameters are written back if the block raises an
exception too:

```
>>> try:
... with zeros.to_module(module):
... raise ValueError("stop")
... except ValueError:
... pass
>>> bool((module.weight != 0).any())
True
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