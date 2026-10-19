# EnsembleModule

*class*tensordict.nn.EnsembleModule(*module: [TensorDictModuleBase](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*, *num_copies: [int](https://docs.python.org/3/builtins/functions.html#int)*, *expand_input: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Bases: [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Runs `num_copies` copies of a module, each with its own parameters, on the same input.

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

- **module** ([*TensorDictModuleBase*](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)) - the module to copy. Its `in_keys` and
`out_keys` become those of the ensemble.
- **num_copies** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of copies.
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

```
>>> import torch
>>> from tensordict.nn import TensorDictModule, TensorDictSequential, EnsembleModule
>>> from tensordict import TensorDict
>>> module = TensorDictModule(torch.nn.Linear(2,3), in_keys=['x'], out_keys=['hidden'])
>>> next_module = TensorDictModule(torch.nn.Linear(3,1), in_keys=['hidden'], out_keys=['y'])
>>> e0 = EnsembleModule(module, num_copies=4, expand_input=True)
>>> e1 = EnsembleModule(next_module, num_copies=4, expand_input=False)
>>> seq = TensorDictSequential(e0, e1)
>>> data = TensorDict({'x': torch.randn(5,2)}, batch_size=[5])
>>> seq(data)
TensorDict(
 fields={
 hidden: Tensor(shape=torch.Size([4, 5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 x: Tensor(shape=torch.Size([4, 5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 y: Tensor(shape=torch.Size([4, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 5]),
 device=None,
 is_shared=False)
```

forward(*tensordict: [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)*) → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Calls every copy on `tensordict` and returns the outputs, stacked along a new first dimension.

*property*in_keys

The keys that the module reads from the input tensordict.

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

reset_parameters_recursive(*parameters: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Resets the values of `parameters` in place, with one set of values per copy of the module.

The constructor calls it to give each copy its own initial values. To
reset the parameters of the ensemble, pass `ensemble.params_td`.

Parameters:

**parameters** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the parameters of `module`, with
the structure of [`from_module()`](tensordict.TensorDict.html#tensordict.TensorDict.from_module).
Each index of the batch dimensions is one copy.

Returns:

A lazy stack of the reset parameters, with the batch size of
`parameters`, or for an empty batch size what
[`reset_parameters_recursive()`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase.reset_parameters_recursive)
of `module` returns.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `parameters` is `None`.

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

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if a key is not in `out_keys`. The error names the
 missing keys, and is also a `RuntimeError`.

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

## Tutorials that use `EnsembleModule`

![](../../_images/sphx_glr_functional_thumb.png)

[Functional calls and ensembles with TensorDict](../../tutorials/functional.html)

Functional calls and ensembles with TensorDict