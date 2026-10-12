# EnsembleModule

*class*tensordict.nn.EnsembleModule(*module: [TensorDictModuleBase](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*, *num_copies: [int](https://docs.python.org/3/builtins/functions.html#int)*, *expand_input: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Bases: [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Module that wraps a module and repeats it to form an ensemble.

The constructor makes `num_copies` sets of parameters, each one reset
with the `reset_parameters` method of the submodules. It stacks them in
`ensemble.params_td`, a `TensorDictParams` with
batch size `[num_copies]`. A call runs the module once per copy with
[`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap), so every output gets a leading dimension of size
`num_copies`.

`ensemble.parameters()` returns the stacked copies, which an optimizer
can train. It also returns the parameters of the original `module`,
which the call does not use.

Parameters:

- **module** (*nn.Module*) - The nn.module to duplicate and wrap.
- **num_copies** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The number of copies of module to make.
- **expand_input** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, every copy receives the
same input. If `False`, the input must already have a leading
dimension of size `num_copies`, and copy `i` receives entry
`i`. Use `False` for an EnsembleModule placed after another
one, for example `EnsembleModule(cnn)` followed by
`EnsembleModule(mlp)`. Defaults to `True`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict.nn import TensorDictModule, EnsembleModule
>>> from tensordict import TensorDict
>>> net = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))
>>> mod = TensorDictModule(net, in_keys=['a'], out_keys=['b'])
>>> ensemble = EnsembleModule(mod, num_copies=3)
>>> data = TensorDict({'a': torch.randn(10, 4)}, batch_size=[10])
>>> ensemble(data)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([3, 10, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 10]),
 device=None,
 is_shared=False)
```

To chain EnsembleModules, set `expand_input=False` on the second one and on those after it.

Examples

```
>>> import torch
>>> from tensordict.nn import TensorDictModule, TensorDictSequential, EnsembleModule
>>> from tensordict import TensorDict
>>> module = TensorDictModule(torch.nn.Linear(2,3), in_keys=['bork'], out_keys=['dork'])
>>> next_module = TensorDictModule(torch.nn.Linear(3,1), in_keys=['dork'], out_keys=['spork'])
>>> e0 = EnsembleModule(module, num_copies=4, expand_input=True)
>>> e1 = EnsembleModule(next_module, num_copies=4, expand_input=False)
>>> seq = TensorDictSequential(e0, e1)
>>> data = TensorDict({'bork': torch.randn(5,2)}, batch_size=[5])
>>> seq(data)
TensorDict(
 fields={
 bork: Tensor(shape=torch.Size([4, 5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 dork: Tensor(shape=torch.Size([4, 5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 spork: Tensor(shape=torch.Size([4, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 5]),
 device=None,
 is_shared=False)
```

forward(*tensordict: [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)*) → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

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

reset_parameters_recursive(*parameters: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Resets the parameters of all the copies of the module.

Parameters:

**parameters** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)) - A TensorDict of parameters for self.module. The batch dimension(s) of the tensordict
denote the number of module copies to reset.

Returns:

A TensorDict of pointers to the reset parameters.

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

## Tutorials that use `EnsembleModule`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict