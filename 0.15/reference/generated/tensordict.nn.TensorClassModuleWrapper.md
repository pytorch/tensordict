# TensorClassModuleWrapper

*class*tensordict.nn.TensorClassModuleWrapper(*module: [TensorClassModuleBase](tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase)[InputTensorClass, OutputTensorClass]*)

Bases: [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

Wraps a [`TensorClassModuleBase`](tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase) so that it takes and returns a tensordict.

The wrapper converts the input tensordict to the input class of the
module, calls the module, and converts the output tensorclass back to a
TensorDict. Its `in_keys` and `out_keys` are the fields of the input
and output classes, with nested keys for nested tensorclasses.
[`TensorClassModuleBase.as_td_module()`](tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase.as_td_module) returns one.

The wrapper returns a new TensorDict that holds only the output fields. It
does not write into the input. In a
[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential), the modules after it cannot
read the input keys, and by default the sequence returns only the
outputs.

Parameters:

**module** ([*TensorClassModuleBase*](tensordict.nn.TensorClassModuleBase.html#tensordict.nn.TensorClassModuleBase)) - the module to wrap. Its input and
output types must be tensorclasses.

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

Converts `tensordict` to the input class, calls the module, and returns its output as a TensorDict.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input tensordict. Its entries that
are not fields of the input class are ignored.
- ***args** - ignored, with a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning).
- ****kwargs** - ignored, `tensordict_out` included, with a
[`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning). Starting with TensorDict 0.17, the
default will change to writing the outputs into
`tensordict_out` and returning it, as
[`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) does, and to raising
a `TypeError` for the other arguments.

Returns:

A new TensorDict with the fields of the output tensorclass.

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