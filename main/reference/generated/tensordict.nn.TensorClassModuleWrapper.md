# TensorClassModuleWrapper

*class*tensordict.nn.TensorClassModuleWrapper(*module: [TensorClassModuleBase](tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase)[InputTensorClass, OutputTensorClass]*)

Bases: [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Wrapper class for TensorClassModuleBase objects.

The wrapper takes a TensorDict, converts it to the input TensorClass of the
module, calls the module, and converts the output TensorClass back to a
TensorDict. [`TensorClassModuleBase.as_td_module()`](tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase.as_td_module) returns one.

The wrapper returns a new TensorDict that holds only the output fields. It
does not write into the input. In a
[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential), the modules after it cannot
read the input keys, and by default the sequence returns only the
outputs.

Parameters:

**module** ([*TensorClassModuleBase*](tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase)) - The TensorClassModuleBase instance to wrap.

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.tensorclass import TensorClass
>>> from tensordict.nn import TensorClassModuleBase
>>> import torch
>>>
>>> class InputTC(TensorClass):
... x: torch.Tensor
...
>>> class OutputTC(TensorClass):
... y: torch.Tensor
...
>>> class MyModule(TensorClassModuleBase[InputTC, OutputTC]):
... def forward(self, input: InputTC) -> OutputTC:
... return OutputTC(y=input.x + 1, batch_size=input.batch_size)
...
>>> module = MyModule()
>>> td_module = module.as_td_module()
>>> td = TensorDict({"x": torch.zeros(3)}, batch_size=[3])
>>> result = td_module(td)
>>> list(result.keys())
['y']
>>> "y" in td
False
```

forward(*tensordict: [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict) = None*, *args=None*, ***kwargs*) → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Forward pass converting TensorDict to TensorClass and back.

Parameters:

- **tensordict** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)) - Input tensordict. Its entries that are not
fields of the input class are ignored.
- ***args** - Additional positional arguments.
- ****kwargs** - Additional keyword arguments.

Returns:

Output tensordict.

Return type:

[*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)

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