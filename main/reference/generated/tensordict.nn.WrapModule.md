# WrapModule

*class*tensordict.nn.WrapModule(*func: Callable[[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, ***, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *in_keys: List[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out_keys: List[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Wraps a function that takes and returns a tensordict as a tensordict module.

Use it to put a function that works on the whole tensordict, such as a
reshape, in a [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential). A sequence
wraps the callables that have no `in_keys` and `out_keys` in a
`WrapModule` itself.

Parameters:

**func** (*Callable**[**[*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*]**,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*]*) - the function to
wrap. It takes a tensordict and returns a tensordict.

Keyword Arguments:

- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the call writes the entries of
the result of `func` into the input with
[`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) and returns the input.
If `False`, the call returns what `func` returns. Defaults to
`False`.
- **in_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the keys that `func` reads.
They are not checked: they tell a
[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) what the module
reads. Defaults to `None`, which sets `[]`.
- **out_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the keys that `func`
writes. They are not checked, like `in_keys`. Defaults to
`None`, which sets `[]`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod, WrapModule
>>> seq = Seq(
... Mod(lambda x: x * 2, in_keys=["x"], out_keys=["y"]),
... WrapModule(lambda td: td.reshape(-1)),
... )
>>> td = TensorDict(x=torch.ones(3, 4, 5), batch_size=[3, 4])
>>> td = seq(td)
>>> td.batch_size
torch.Size([12])
>>> td["y"].shape
torch.Size([12, 5])
```

forward(*data: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Calls `func` on `data` and returns the result, or `data` updated with it if `inplace=True`.

*static*is_tdmodule_compatible(*module*)

Returns `True` if `module` has both an `in_keys` and an `out_keys` attribute.

[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) wraps the callables that
fail this check in a `WrapModule`.

Parameters:

**module** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) - the object to check.

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