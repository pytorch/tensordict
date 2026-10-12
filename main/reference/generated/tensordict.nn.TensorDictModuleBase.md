# TensorDictModuleBase

*class*tensordict.nn.TensorDictModuleBase(**args: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*)

Bases: [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Base class to TensorDict modules.

TensorDictModule subclasses are characterized by `in_keys` and `out_keys`
key-lists that indicate what input entries are to be read and what output
entries should be expected to be written.

The forward method input/output signature should always follow the
convention:

```
tensordict_out = module.forward(tensordict_in)
```

Unlike [`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule), TensorDictModuleBase is typically used via subclassing:
you can wrap any python function in a TensorDictModuleBase subclass, as long as the subclass forward reads and
writes tensordict (or related types) instances.

The in_keys and out_keys should be properly specified. For example, out_keys can be dynamically reduced using
`select_out_keys()`.

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModuleBase
>>> class Mod(TensorDictModuleBase):
... in_keys = ["a"] # can also be specified during __init__
... out_keys = ["b", "c"]
... def forward(self, tensordict):
... b = tensordict["a"].clone()
... c = b + 1
... return tensordict.replace({"b": b, "c": c})
>>> mod = Mod()
>>> td = mod(TensorDict(a=0))
>>> td["b"]
tensor(0)
>>> td["c"]
tensor(1)
>>> _ = mod.select_out_keys("c")
>>> td = mod(TensorDict(a=0))
>>> td["c"]
tensor(1)
>>> assert "b" not in td
```

*property*in_keys

The keys that the module reads from the input tensordict.

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

select_out_keys(**out_keys*) → TensorDictModuleBase

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

The simplest usage is with [`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule):

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
`TensorDictModuleBase.reset_out_keys()` method.

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

## Tutorials that use `TensorDictModuleBase`

![](../../_images/sphx_glr_export_thumb.png)

[Exporting tensordict modules](../../tutorials/export.html)

Exporting tensordict modules
![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule