# functional_vmap

tensordict.nn.functional_vmap(*module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, ***, *params: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *args: [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple) = ()*, *kwargs: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *in_dims: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [int](https://docs.python.org/3/builtins/functions.html#int)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *randomness: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['error', 'different', 'same'] = 'error'*, *chunk_size: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *method: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *loop: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Calls `module` once per set of parameters in `params` and stacks the outputs.

The batch dimensions of `params` index the sets, for example the
parameters of several modules stacked with
[`from_modules()`](tensordict.TensorDictBase.from_modules.html#tensordict.TensorDictBase.from_modules). They are flattened into one
dimension and [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) runs over it once, so every output gets
`params.batch_size` as its leading dimensions. With an empty batch
size, `module` is called once with `params`.

Parameters:

**module** (*nn.Module*) - the module to call. `params` must have the
structure of [`from_module()`](tensordict.TensorDict.html#tensordict.TensorDict.from_module) of it.

Keyword Arguments:

- **params** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the stacked parameters and buffers.
- **args** ([*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*,**optional*) - the positional inputs of `module`. Every set
receives the same ones. Defaults to `()`.
- **kwargs** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - the keyword inputs of `module`.
- **in_dims** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - maps keys of `kwargs` to the dimension that
indexes the sets. That input needs `params.batch_size` at this
dimension, and set `i` receives its slice `i`. With several
batch dimensions, the value names the first of them. The other
inputs are shared.
- **randomness** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - what a random op, such as dropout, does.
`"error"` will raise, `"different"` means the randomness for each batch
will be different, and `"same"` gives every set the same ones. Defaults to
`"error"`.
- **chunk_size** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of sets computed at once.
- **method** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the name of the method of `module` to call
instead of `forward`.
- **loop** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, calls the sets one at a time and
stacks the results, for modules that [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) cannot
handle. A loop cannot detect random ops, so it does not raise under
`randomness="error"`: each set gets its own random values, as
with `"different"`. Defaults to `False`.

Returns:

The outputs of `module`, with `params.batch_size` as leading
dimensions.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if a key of `in_dims` is not in `kwargs`.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `in_dims` is not empty and `params` has no batch
 dimension, if an input named in `in_dims` does not have
 `params.batch_size` at its dimension, or if `loop=True` with
 `randomness="same"`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import functional_vmap
>>> nets = [nn.Linear(3, 1) for _ in range(4)]
>>> params = TensorDict.from_modules(*nets)
>>> x = torch.randn(10, 3)
>>> functional_vmap(nets[0], params=params, args=(x,)).shape
torch.Size([4, 10, 1])
>>> x_per_set = torch.randn(4, 10, 3)
>>> functional_vmap(
... nets[0],
... params=params,
... kwargs={"input": x_per_set},
... in_dims={"input": 0},
... ).shape
torch.Size([4, 10, 1])
```