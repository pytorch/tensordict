# TensorDictModule

*class*tensordict.nn.TensorDictModule(*module: Callable*, *in_keys*, *out_keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | List[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*, ***, *out_to_in_map: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) = True*, *method: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *method_kwargs: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *get_kwargs: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Wraps a callable so that it reads its inputs from a tensordict and writes its outputs to it.

When you call the module with a tensordict, it reads the entries at
`in_keys`, passes them to `module`, and writes the results at
`out_keys`. By default it writes into the input tensordict and returns
it. You can also call it with tensors as positional or keyword arguments
(see [`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch)). It then returns the output tensors.
See [Modules: TensorDictModule and tensordict.nn](../../modules.html#modules-guide) for a guide.

Parameters:

- **module** (*Callable*) - the callable to wrap, typically a
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). It can return a tensor, a tuple of
tensors, or a dict or a tensordict. A dict or a tensordict that
holds all the `out_keys` is read by key, so its order does not
matter.
- **in_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*, or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the entries that the
module reads from the input tensordict. If a list, the values are
passed as positional arguments, in this order. If a dict, the values
are passed as keyword arguments. By default, the dict maps an
argument name to a tensordict key: `in_keys={"x": "obs"}` calls
`module(x=td["obs"])`. One tensordict key can feed several
arguments. `out_to_in_map=False` reverses the direction.
- **out_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys where the outputs
of `module` are written, one key per output, in order. An output
whose key is `"_"` is not written.

Keyword Arguments:

- **out_to_in_map** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the direction of a dict `in_keys`.
`True` reads it as `{argument: tensordict_key}`, `False` as
`{tensordict_key: argument}`. Only used when `in_keys` is a
dict. If you pass it with another type of `in_keys`, the
constructor warns. Defaults to `True` for a dict `in_keys`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - 

where the module writes its outputs.

- `True`: into the input tensordict, which `forward` returns.
- `False`: into a new [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) that holds only
the outputs, with an empty batch size and no device, whatever
the type of the input.
- `"empty"`: into `tensordict.empty()`, which keeps the type,
batch size and device of the input.

If you pass `tensordict_out` to `forward()`, the outputs are
written into it, whatever the value of `inplace`. Defaults to
`True`.
- **method** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the name of the method of `module` to call.
Unlike a lambda around the method, this keeps the parameters and
buffers of `module` registered in the wrapper. Defaults to
`None`, which calls `module` itself.
- **method_kwargs** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - keyword arguments passed to
`module` (or to `method`) at each call. Defaults to `None`.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a missing input raises a
`KeyError`. If `False`, `None` is passed in its place, and if
`module` then fails, the error names the missing keys. Defaults to
`False`.
- **get_kwargs** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - keyword arguments passed to
[`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) when the inputs are read,
for example `{"as_padded_tensor": True}` for ragged inputs.
Defaults to `None`.

Attributes that the wrapper does not have, such as `weight` for a
[`Linear`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear), are read from `module`. Inside a
[`set_skip_existing`](tensordict.nn.set_skip_existing.html#tensordict.nn.set_skip_existing) block, the module does not run
when all its `out_keys` are already in the input.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> module = TensorDictModule(nn.Linear(3, 4), in_keys=["x"], out_keys=["y"])
>>> data = TensorDict(x=torch.ones(2, 3), batch_size=[2])
>>> data = module(data)
>>> print(data)
TensorDict(
 fields={
 x: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 y: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
>>> module(torch.ones(2, 3)).shape
torch.Size([2, 4])
```

A plain function works too. This one reads a nested key and writes two
nested keys:

```
>>> module = TensorDictModule(
... lambda x: (x - 1, x + 1),
... in_keys=[("input", "x")],
... out_keys=[("output", "x-1"), ("output", "x+1")],
... )
>>> td = module(TensorDict({("input", "x"): torch.zeros(())}))
>>> td["output", "x+1"]
tensor(1.)
```

With empty `in_keys`, the module only writes entries:

```
>>> module = TensorDictModule(lambda: torch.zeros(3), in_keys=[], out_keys=["x"])
>>> module(TensorDict())["x"]
tensor([0., 0., 0.])
```

A dict `in_keys` passes the values as keyword arguments. By default
(`out_to_in_map=True`), its keys are the argument names and its values
are the tensordict keys. One tensordict key can feed several arguments:

```
>>> module = TensorDictModule(lambda x, *, y, z: x+y+z,
... in_keys={'x': '1', 'y': '2', 'z': '2'}, out_keys=['t']
... )
>>> td = module(TensorDict({'1': torch.ones(()), '2': torch.ones(())*2}, []))
>>> td['t']
tensor(5.)
```

With `out_to_in_map=False`, the keys of the dict are the tensordict keys
and its values are the argument names:

```
>>> module = TensorDictModule(lambda x, *, y: x+y,
... in_keys={'1': 'x', '2': 'y'}, out_keys=['z'], out_to_in_map=False
... )
>>> td = module(TensorDict({'1': torch.ones(()), '2': torch.ones(())*2}, []))
>>> td['z']
tensor(3.)
```

`method` calls another method of the module, here with a keyword
argument from `method_kwargs`:

```
>>> class MyNet(nn.Module):
... def my_func(self, tensor, *, an_integer):
... return tensor + an_integer
...
>>> module = TensorDictModule(
... MyNet(),
... in_keys=["a"],
... out_keys=["b"],
... method="my_func",
... method_kwargs={"an_integer": 2},
... )
>>> module(TensorDict(a=0))["b"]
tensor(2)
```

To call the module with other parameters, put them in a tensordict and
use [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) as a context manager:

```
>>> module = TensorDictModule(
... nn.GRUCell(4, 8), in_keys=["input", "hidden"], out_keys=["output"]
... )
>>> params = TensorDict.from_module(module)
>>> params = params.apply(torch.zeros_like)
>>> td = TensorDict(input=torch.ones(3, 4), hidden=torch.ones(3, 8), batch_size=[3])
>>> with params.to_module(module):
... td = module(td)
>>> # with zero weights, the output is half the hidden state
>>> (td["output"] == 0.5).all()
tensor(True)
```

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

The device of the first parameter of the module, or `cpu` if it has no parameter.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17 because modules may
contain parameters and buffers on multiple devices.

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *args=None*, ***, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Reads `in_keys` from `tensordict`, calls the wrapped module, and writes `out_keys`.

If you call the module with tensors instead of a tensordict, the
tensors are put in a new tensordict first, and the call returns the
output tensors (see [`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch)).

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input tensordict.

Keyword Arguments:

- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - if given, the outputs
are written into it and it is returned, whatever the value of
`inplace`. Defaults to `None`.
- ****kwargs** - passed to the wrapped module.

Returns:

The tensordict that holds the outputs (see `inplace`).

*property*in_keys

The keys that the module reads from the input tensordict.

*property*is_functional*: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the wrapped module is a legacy functorch functional module.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17. It only checks
for the `FunctionalModule` classes of the legacy
`functorch.make_functional` API and has no replacement.

*static*is_tdmodule_compatible(*module*)

Returns `True` if `module` has both an `in_keys` and an `out_keys` attribute.

[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) wraps the callables that
fail this check in a [`WrapModule`](tensordict.nn.WrapModule.html#tensordict.nn.WrapModule).

Parameters:

**module** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) - the object to check.

*property*out_keys

The keys that the module writes to the output tensordict.

After `select_out_keys()`, only the selected keys. The full list
is in `out_keys_source`.

*property*out_keys_source

All the keys that the module writes, before `select_out_keys()`.

reset_out_keys()

Undoes `select_out_keys()`: the module writes all its `out_keys` again.

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> mod = TensorDictModule(
... lambda x, y: (x + 2, y + 2), in_keys=["a", "b"], out_keys=["c", "d"]
... )
>>> _ = mod.select_out_keys("d")
>>> mod.out_keys
['d']
>>> _ = mod.reset_out_keys()
>>> mod.out_keys
['c', 'd']
>>> td = mod(TensorDict(a=torch.zeros(()), b=torch.ones(())))
>>> sorted(td.keys())
['a', 'b', 'c', 'd']
```

reset_parameters_recursive(*parameters: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None)

Resets the parameters of every submodule that has a `reset_parameters` method.

Without `parameters`, the parameters of the module are reset in
place. With `parameters`, the module is left as it is: the values in
`parameters` are reset in place, as if they were loaded into the
module (see [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module)). Use this
for parameters kept outside the module, for example in a functional
call. Values that share their storage with the module parameters,
such as those of [`from_module()`](tensordict.TensorDict.html#tensordict.TensorDict.from_module), reset
the module too.

Parameters:

**parameters** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the parameters to reset,
with the structure of
[`from_module()`](tensordict.TensorDict.html#tensordict.TensorDict.from_module) and an empty batch
size. Defaults to `None`, which resets the parameters of the
module.

Returns:

`None` without `parameters`. Otherwise a new tensordict of
tensors that share their storage with `parameters` and require
gradients.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `parameters` has batch dimensions.

Without `parameters`, the call warns if no submodule has a
`reset_parameters` method.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torch import nn
>>> net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
>>> old_weight = net[0].weight.clone()
>>> module = TensorDictModule(net, in_keys=["x"], out_keys=["y"])
>>> module.reset_parameters_recursive()
>>> torch.equal(old_weight, net[0].weight)
False
```

The parameters can also live outside the module:

```
>>> params = TensorDict.from_module(module).clone()
>>> old_params = params.clone()
>>> _ = module.reset_parameters_recursive(params)
>>> (old_params == params).any()
False
>>> torch.equal(net[0].weight, old_params["module", "0", "weight"])
True
```

select_out_keys(**out_keys*) → [TensorDictModuleBase](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Keeps only some of the `out_keys` in the output of the module, and returns the module.

Use it to drop intermediate outputs, for example those of a
[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential). The module changes in
place: `out_keys` becomes the selected keys, and
`out_keys_source` keeps the full list. A call still computes all
the outputs, and then removes the ones that are not selected. It
removes nothing else, so the input entries stay in the output. When
you call the module with tensors (see [`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch)),
it returns only the selected outputs. `reset_out_keys()` undoes
the selection.

Parameters:

***out_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys to keep. Each one must be in
`out_keys`.

Returns:

`self`.

Raises:

- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if a single key is given and it is not in
 `out_keys`.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if several keys are given and one of them is not in
 `out_keys`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> mod = TensorDictModule(
... lambda x, y: (x + 2, y + 2), in_keys=["a", "b"], out_keys=["c", "d"]
... )
>>> _ = mod.select_out_keys("d")
>>> mod.out_keys, mod.out_keys_source
(['d'], ['c', 'd'])
>>> td = mod(TensorDict(a=torch.zeros(()), b=torch.ones(())))
>>> sorted(td.keys())
['a', 'b', 'd']
>>> mod(torch.zeros(()), torch.ones(()))
tensor(3.)
>>> _ = mod.reset_out_keys()
>>> sorted(mod(TensorDict(a=torch.zeros(()), b=torch.ones(()))).keys())
['a', 'b', 'c', 'd']
```

It works on a [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) too:

```
>>> from tensordict.nn import TensorDictSequential
>>> seq = TensorDictSequential(
... TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"]),
... TensorDictModule(lambda y: y + 1, in_keys=["y"], out_keys=["z"]),
... )
>>> _ = seq.select_out_keys("z")
>>> sorted(seq(TensorDict(x=torch.zeros(()))).keys())
['x', 'z']
```

## Tutorials that use `TensorDictModule`

![](../../_images/sphx_glr_export_thumb.png)

[Exporting tensordict modules](../../tutorials/export.html)

Exporting tensordict modules
![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule