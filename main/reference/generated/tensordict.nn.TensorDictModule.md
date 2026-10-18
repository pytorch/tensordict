# TensorDictModule

*class*tensordict.nn.TensorDictModule(*module: Callable*, *in_keys*, *out_keys: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | List[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*, ***, *out_to_in_map: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) = True*, *method: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *method_kwargs: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *get_kwargs: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Wraps a callable, such as a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), so that it reads its inputs from a tensordict and writes its outputs to it.

When you call the module with a tensordict, it reads the entries at
`in_keys`, passes them to `module`, and writes the results at
`out_keys`. By default it writes into the input tensordict and returns
it. You can also call it with tensors as positional or keyword arguments
(see [`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch)). It then returns the output tensors.
See [Modules: TensorDictModule and tensordict.nn](../../modules.html#modules-guide) for a guide.

Parameters:

- **module** (*Callable**[**[**Any**]**,**Any**]*) - a callable, typically a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module),
used to map the input to the output parameter space. Its forward method
can return a single tensor, a tuple of tensors or even a dictionary.
In the latter case, the output keys of the `TensorDictModule`
will be used to populate the output tensordict (ie. the keys present
in `out_keys` should be present in the dictionary returned by the
`module` forward method).
Dictionary outputs stay matched by key when `select_out_keys()`
filters or reorders the outputs.
- **in_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*, or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the entries that the
module reads from the input tensordict. If a list, the values are
passed as positional arguments, in this order. If a dict, the values
are passed as keyword arguments. By default, the dict maps an
argument name to a tensordict key: `in_keys={"x": "obs"}` calls
`module(x=td["obs"])`. One tensordict key can feed several
arguments. `out_to_in_map=False` reverses the direction.
- **out_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys where the outputs of
`module` are written. The length of `out_keys` must match the
number of tensors returned by `module`. The output at the key
`"_"` is not written.

Keyword Arguments:

- **out_to_in_map** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - the direction of a dict `in_keys`.
`True` reads it as `{argument: tensordict_key}`, `False` as
`{tensordict_key: argument}`. Only used when `in_keys` is a
dict. If you pass it with another type of `in_keys`, the
constructor warns. Defaults to `True` for a dict `in_keys`.
- **inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - 

where the module writes its outputs.

- `True` (default): into the input tensordict, which `forward` returns.
- `False`: into a new [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) that holds only
the outputs, with an empty batch size and no device, whatever
the type of the input.
- `"empty"`: into `tensordict.empty()`, which keeps the type,
batch size and device of the input.

If you pass `tensordict_out` to `forward()`, the outputs are
written into it, whatever the value of `inplace`.
- **method** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the method to be called in the module, if any. Defaults to __call__.
- **method_kwargs** (*Dict**[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**Any**]**,**optional*) - additional keyword arguments to be passed to the module's method being called.
- **strict** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the module will raise an exception if any of the inputs is missing from
the input tensordict. Otherwise, a None value will be used as placeholder. Defaults to `False`.
- **get_kwargs** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**Any**]**,**optional*) - additional keyword arguments to be passed to the [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get)
method. This is particularly useful when dealing with ragged tensors (see [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get)).
Defaults to `{}`.

To wrap a neural network, give it the input and output keys:

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> # one can wrap regular nn.Module
>>> module = TensorDictModule(nn.Transformer(128), in_keys=["input", "tgt"], out_keys=["out"])
>>> input = torch.ones(2, 3, 128)
>>> tgt = torch.zeros(2, 3, 128)
>>> data = TensorDict({"input": input, "tgt": tgt}, batch_size=[2, 3])
>>> data = module(data)
>>> print(data)
TensorDict(
 fields={
 input: Tensor(shape=torch.Size([2, 3, 128]), device=cpu, dtype=torch.float32, is_shared=False),
 out: Tensor(shape=torch.Size([2, 3, 128]), device=cpu, dtype=torch.float32, is_shared=False),
 tgt: Tensor(shape=torch.Size([2, 3, 128]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2, 3]),
 device=None,
 is_shared=False)
```

We can also pass directly the tensors

Examples

```
>>> out = module(input, tgt)
>>> assert out.shape == input.shape
>>> # we can also wrap regular functions
>>> module = TensorDictModule(lambda x: (x-1, x+1), in_keys=[("input", "x")], out_keys=[("output", "x-1"), ("output", "x+1")])
>>> module(TensorDict({("input", "x"): torch.zeros(())}, batch_size=[]))
TensorDict(
 fields={
 input: TensorDict(
 fields={
 x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 output: TensorDict(
 fields={
 x+1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 x-1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

We can use TensorDictModule to populate a tensordict:

Examples

```
>>> module = TensorDictModule(lambda: torch.randn(3), in_keys=[], out_keys=["x"])
>>> print(module(TensorDict({}, batch_size=[])))
TensorDict(
 fields={
 x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

A dict `in_keys` passes the values as keyword arguments. By default
(`out_to_in_map=True`), its keys are the argument names and its values
are the tensordict keys. One tensordict key can then feed several
arguments:

Examples

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

Examples

```
>>> module = TensorDictModule(lambda x, *, y: x+y,
... in_keys={'1': 'x', '2': 'y'}, out_keys=['z'], out_to_in_map=False
... )
>>> td = module(TensorDict({'1': torch.ones(()), '2': torch.ones(())*2}, []))
>>> td['z']
tensor(3.)
```

We can specify the method to be called within a module. Compared to using a lambda function or similar around the
module's method, this has the advantage that the module attributes (params, buffers, submodules) will be exposed.

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
>>> from torch import nn
>>> import torch
>>>
>>> class MyNet(nn.Module):
... def my_func(self, tensor: torch.Tensor, *, an_integer: int):
... return tensor + an_integer
...
>>> s = Seq(
... {
... "a": lambda td: td+1,
... "b": lambda td: td * 2,
... "c": Mod(MyNet(), in_keys=["a"], out_keys=["b"], method="my_func", method_kwargs={"an_integer": 2}),
... }
... )
>>> td = s(TensorDict(a=0))
>>> print(td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> assert td["b"] == 4
```

To call the module with other parameters, put them in a tensordict and
use [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) as a context manager:

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> td = TensorDict({"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3,])
>>> module = torch.nn.GRUCell(4, 8)
>>> td_module = TensorDictModule(
... module=module, in_keys=["input", "hidden"], out_keys=["output"]
... )
>>> params = TensorDict.from_module(td_module)
>>> # functional API
>>> with params.to_module(td_module):
... td_functional = td_module(td.clone())
>>> print(td_functional)
TensorDict(
 fields={
 hidden: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 output: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

In the stateful case:

```
>>> module = torch.nn.GRUCell(4, 8)
>>> td_module = TensorDictModule(
... module=module, in_keys=["input", "hidden"], out_keys=["output"]
... )
>>> td_stateful = td_module(td.clone())
>>> print(td_stateful)
TensorDict(
 fields={
 hidden: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 output: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

A deprecated approximation of the module device.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17 because modules may
contain parameters and buffers on multiple devices.

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *args=None*, ***, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Reads `in_keys` from `tensordict`, calls the wrapped module, and writes `out_keys`.

If you call the module with tensors instead of a tensordict, the
tensors are put in a new tensordict first (see
[`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch)).

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input tensordict.

Keyword Arguments:

- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - if given, the outputs
are written into it and it is returned, whatever the value of
`inplace`. Defaults to `None`.
- ****kwargs** - passed to the wrapped module.

Returns:

the tensordict that holds the outputs (see `inplace`).

*property*in_keys

The keys that the module reads from the input tensordict.

*property*is_functional*: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the wrapped module is a legacy functorch functional module.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17. It only checks
for the `FunctionalModule` classes of the legacy
`functorch.make_functional` API and has no replacement.

*static*is_tdmodule_compatible(*module*)

Checks if a module is compatible with TensorDictModule API.

Parameters:

**module** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) - the object to check.

Returns:

`True` if `module` has both an `in_keys` and an `out_keys`
attribute.

*property*out_keys

The keys that the module writes to the output tensordict.

After `select_out_keys()`, only the selected keys. The full list
is in `out_keys_source`.

*property*out_keys_source

All the keys that the module writes, before `select_out_keys()`.

reset_out_keys()

Resets the `out_keys` attribute to its original value.

Returns: the same module, with its original `out_keys` values.

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule, TensorDictSequential
>>> import torch
>>> mod = TensorDictModule(lambda x, y: (x+2, y+2), in_keys=["a", "b"], out_keys=["c", "d"])
>>> _ = mod.select_out_keys("d")
>>> td = TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, [])
>>> mod(td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> _ = mod.reset_out_keys()
>>> mod(td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

reset_parameters_recursive(*parameters: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None)

Recursively reset the parameters of the module and its children.

Parameters:

**parameters** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*of**parameters**,**optional*) - If set to None, the module will reset using self.parameters().
Otherwise, we will reset the parameters in the tensordict in-place. This is
useful for functional modules where the parameters are not stored in the module itself.

Returns:

A tensordict of the new parameters, only if parameters was not None.

Examples

```
>>> from tensordict.nn import TensorDictModule
>>> from torch import nn
>>> net = nn.Sequential(nn.Linear(2,3), nn.ReLU())
>>> old_param = net[0].weight.clone()
>>> module = TensorDictModule(net, in_keys=['bork'], out_keys=['dork'])
>>> module.reset_parameters_recursive()
>>> (old_param == net[0].weight).any()
tensor(False)
```

This method also supports functional parameter sampling:

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torch import nn
>>> net = nn.Sequential(nn.Linear(2,3), nn.ReLU())
>>> module = TensorDictModule(net, in_keys=['bork'], out_keys=['dork'])
>>> params = TensorDict.from_module(module)
>>> old_params = params.clone(recurse=True)
>>> _ = module.reset_parameters_recursive(params)
>>> (old_params == params).any()
False
```

select_out_keys(**out_keys*) → [TensorDictModuleBase](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Selects the keys that will be found in the output tensordict.

This is useful whenever one wants to get rid of intermediate keys in a
complicated graph, or when the presence of these keys may trigger unexpected
behaviours. The module removes only the out-keys that are not selected:
the other entries of the input tensordict stay in place.

The original `out_keys` can still be accessed via `module.out_keys_source`.

Parameters:

***out_keys** (*a sequence**of**strings**or**tuples**of**strings*) - the
out_keys that should be found in the output tensordict.

Returns: the same module, modified in-place with updated `out_keys`.

The simplest usage is with `TensorDictModule`:

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule, TensorDictSequential
>>> import torch
>>> mod = TensorDictModule(lambda x, y: (x+2, y+2), in_keys=["a", "b"], out_keys=["c", "d"])
>>> td = TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, [])
>>> mod(td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> _ = mod.select_out_keys("d")
>>> td = TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, [])
>>> mod(td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This feature will also work with dispatched arguments:

```
>>> mod(torch.zeros(()), torch.ones(()))
tensor(3.)
```

This change will occur in-place (ie the same module will be returned
with an updated list of out_keys). It can be reverted using the
[`TensorDictModuleBase.reset_out_keys()`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase.reset_out_keys) method.

```
>>> _ = mod.reset_out_keys()
>>> mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This will work with other classes too, such as Sequential:

```
>>> from tensordict.nn import TensorDictSequential
>>> seq = TensorDictSequential(
... TensorDictModule(lambda x: x+1, in_keys=["x"], out_keys=["y"]),
... TensorDictModule(lambda x: x+1, in_keys=["y"], out_keys=["z"]),
... )
>>> td = TensorDict({"x": torch.zeros(())}, [])
>>> seq(td)
TensorDict(
 fields={
 x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 y: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 z: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> _ = seq.select_out_keys("z")
>>> td = TensorDict({"x": torch.zeros(())}, [])
>>> seq(td)
TensorDict(
 fields={
 x: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 z: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
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