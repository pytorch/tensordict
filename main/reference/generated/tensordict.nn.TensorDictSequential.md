# TensorDictSequential

*class*tensordict.nn.TensorDictSequential(**modules: Callable[[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *partial_tolerant: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *selected_out_keys: List[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)

Runs TensorDictModules one after the other on the same tensordict.

Like [`torch.nn.Sequential`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential), each module receives the output of the
previous one. Here the data is a tensordict, so a module can read any entry
that an earlier module wrote.

The `in_keys` of the sequence are the keys that a module reads and that no
earlier module writes. The input tensordict must contain them. The
`out_keys` are all the keys that the modules write, each listed once.
To call the sequence with other parameters, use
[`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module) (see the second example).
See [Modules: TensorDictModule and tensordict.nn](../../modules.html#modules-guide) for a guide.

Parameters:

**modules** (*OrderedDict**[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**Callable**[**[*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*]**,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*]**]**|**List**[**Callable**[**[*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*]**,*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*]**]*) - ordered sequence of callables that take a TensorDictBase as input and return a TensorDictBase.
These can be instances of TensorDictModuleBase or any other function that matches this signature.
Note that if a non-TensorDictModuleBase callable is used, its input and output keys will not be tracked,
and thus will not affect the in_keys and out_keys attributes of the TensorDictSequential.
Regular `dict` inputs will be converted to `OrderedDict` if necessary.

Keyword Arguments:

- **partial_tolerant** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if True, the input tensordict can miss some of the input keys.
If so, the only module that will be executed are those who can be executed given the keys that
are present.
Also, if the input tensordict is a lazy stack of tensordicts AND if partial_tolerant is [`True`](https://docs.python.org/3/builtins/constants.html#True) AND if the
stack does not have the required keys, then TensorDictSequential will scan through the sub-tensordicts
looking for those that have the required keys, if any. Defaults to False.
- **selected_out_keys** (*iterable**of**NestedKeys**,**optional*) - the list of out-keys to select. If not provided, all
`out_keys` will be written.
- **inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - 

where the sequence writes its `out_keys`
after the last module ran.

- `True`: into the input tensordict, which is returned.
- `False`: into a new, empty [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).
- `"empty"`: into `input.empty()`, which keeps the type, device
and batch size of the input.
- `None` (default): the sequence returns what the last module
returns, so the `inplace` value of each module decides.

With `False` or `"empty"`, the modules run on a shallow copy
of the input, so the input does not change. See the last example.

Note

A `TensorDictSequential` instance may have a long list of output keys, and one may wish to remove
some of them after execution for clarity or memory purposes. If this is the case, the method `select_out_keys()`
can be used after instantiation, or selected_out_keys may be passed to the constructor.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torch import nn
>>> from tensordict.nn import TensorDictModule, TensorDictSequential
>>> _ = torch.manual_seed(0)
>>> module = TensorDictSequential(
... TensorDictModule(lambda x: x+1, in_keys=["x"], out_keys=["x+1"]),
... TensorDictModule(nn.Linear(3, 4), in_keys=["x+1"], out_keys=["w*(x+1)+b"]),
... )
>>> # with tensordict input
>>> print(module(TensorDict({"x": torch.zeros(3)}, [])))
TensorDict(
 fields={
 w*(x+1)+b: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.float32, is_shared=False),
 x+1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> module.in_keys, module.out_keys
(['x'], ['x+1', 'w*(x+1)+b'])
>>> # with tensor input: returns all the output keys in the order of the modules, ie "x+1" and "w*(x+1)+b"
>>> module(x=torch.zeros(3))
(tensor([1., 1., 1.]), tensor([-0.7214, -0.8748, 0.1571, -0.1138], grad_fn=<ViewBackward0>))
>>> module(torch.zeros(3))
(tensor([1., 1., 1.]), tensor([-0.7214, -0.8748, 0.1571, -0.1138], grad_fn=<ViewBackward0>))
```

TensorDictSequential supports functional, modular and vmap coding.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import (
... ProbabilisticTensorDictModule,
... ProbabilisticTensorDictSequential,
... TensorDictModule,
... TensorDictSequential,
... )
>>> from tensordict.nn.distributions import NormalParamExtractor
>>> from torch.distributions import Normal
>>> td = TensorDict({"input": torch.randn(3, 4)}, [3,])
>>> net1 = torch.nn.Linear(4, 8)
>>> module1 = TensorDictModule(net1, in_keys=["input"], out_keys=["params"])
>>> normal_params = TensorDictModule(
... NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
... )
>>> td_module1 = ProbabilisticTensorDictSequential(
... module1,
... normal_params,
... ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"],
... out_keys=["hidden"],
... distribution_class=Normal,
... return_log_prob=True,
... )
... )
>>> module2 = torch.nn.Linear(4, 8)
>>> td_module2 = TensorDictModule(
... module=module2, in_keys=["hidden"], out_keys=["output"]
... )
>>> td_module = TensorDictSequential(td_module1, td_module2)
>>> params = TensorDict.from_module(td_module)
>>> with params.to_module(td_module):
... _ = td_module(td)
>>> print(td)
TensorDict(
 fields={
 hidden: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 hidden_log_prob: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 output: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 params: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

In the vmap case:

```
>>> from torch import vmap
>>> params = params.expand(4)
>>> def func(td, params):
... with params.to_module(td_module):
... return td_module(td)
>>> td_vmap = vmap(func, (None, 0))(td, params)
>>> print(td_vmap)
TensorDict(
 fields={
 hidden: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 hidden_log_prob: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 input: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 output: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 params: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 3]),
 device=None,
 is_shared=False)
```

The `inplace` argument controls where the outputs go. In this example the
inner modules use `inplace=False`, so each one passes only its outputs to
the next. The sequence uses `inplace=True`, so it writes the final
`output` into the input object. The intermediate entries are not written
into it.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
>>>
>>> class MyClass(TensorClass):
... input: torch.Tensor
... output: torch.Tensor | None = None
>>>
>>> obj = MyClass(torch.randn(2, 3), batch_size=(2,))
>>>
>>> model = Seq(
... Mod(
... lambda x: (x + 1, x - 1),
... in_keys=["input"],
... out_keys=[("intermediate", "0"), ("intermediate", "1")],
... inplace=False
... ),
... Mod(
... lambda y0, y1: y0 * y1,
... in_keys=[("intermediate", "0"), ("intermediate", "1")],
... out_keys=["output"],
... inplace=False
... ),
... inplace=True, )
>>> print(model(obj))
MyClass(
 input=Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 output=Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
```

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

A deprecated approximation of the module device.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17 because modules may
contain parameters and buffers on multiple devices.

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

select_out_keys(**selected_out_keys*) → TensorDictSequential

Selects the keys within the `out_keys` that will be found in the output tensordict.

This change occurs in-place and can be reverted using `reset_out_keys()`.

Parameters:

***selected_out_keys** (*a sequence**of**strings**or**tuples**of**strings*) - the
out_keys that should be found in the output tensordict. Each of them
must be part of the module's `out_keys`.

Returns: the same module, modified in-place with updated `out_keys`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if a key in `selected_out_keys` is not part of `out_keys`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
>>> seq = Seq(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["y"]),
... Mod(lambda y: y + 1, in_keys=["y"], out_keys=["z"]),
... )
>>> _ = seq.select_out_keys("z")
>>> seq.out_keys
['z']
>>> td = seq(TensorDict(x=torch.zeros(())))
>>> assert "y" not in td
>>> td["z"]
tensor(2.)
>>> _ = seq.reset_out_keys()
>>> seq.out_keys
['y', 'z']
```

select_subsequence(*in_keys: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out_keys: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → TensorDictSequential

Returns a new TensorDictSequential with only the modules that are necessary to compute the given output keys with the given input keys.

Parameters:

- **in_keys** - input keys of the subsequence we want to select.
All the keys absent from `in_keys` will be considered as
non-relevant, and modules that *just* take these keys as inputs
will be discarded.
The resulting sequential module will follow the pattern "all
the modules which output will be affected by a different value
for any in <in_keys>".
If none is provided, the module's `in_keys` are assumed.
- **out_keys** - output keys of the subsequence we want to select.
Only the modules that are necessary to get the `out_keys`
will be found in the resulting sequence.
The resulting sequential module will follow the pattern "all
the modules that condition the value or <out_keys> entries."
If none is provided, the module's `out_keys` are assumed.

Returns:

A new TensorDictSequential with only the modules that are necessary according to the given input and output keys.

Examples

```
>>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
>>> idn = lambda x: x
>>> module = Seq(
... Mod(idn, in_keys=["a"], out_keys=["b"]),
... Mod(idn, in_keys=["b"], out_keys=["c"]),
... Mod(idn, in_keys=["c"], out_keys=["d"]),
... Mod(idn, in_keys=["a"], out_keys=["e"]),
... )
>>> # select all modules whose output depend on "a"
>>> module.select_subsequence(in_keys=["a"])
TensorDictSequential(
 module=ModuleList(
 (0): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['a'],
 out_keys=['b'])
 (1): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['b'],
 out_keys=['c'])
 (2): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['c'],
 out_keys=['d'])
 (3): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['a'],
 out_keys=['e'])
 ),
 in_keys=['a'],
 out_keys=['b', 'c', 'd', 'e'])
>>> # select all modules whose output depend on "c"
>>> module.select_subsequence(in_keys=["c"])
TensorDictSequential(
 module=ModuleList(
 (0): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['c'],
 out_keys=['d'])
 ),
 in_keys=['c'],
 out_keys=['d'])
>>> # select all modules that affect the value of "c"
>>> module.select_subsequence(out_keys=["c"])
TensorDictSequential(
 module=ModuleList(
 (0): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['a'],
 out_keys=['b'])
 (1): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['b'],
 out_keys=['c'])
 ),
 in_keys=['a'],
 out_keys=['b', 'c'])
>>> # select all modules that affect the value of "e"
>>> module.select_subsequence(out_keys=["e"])
TensorDictSequential(
 module=ModuleList(
 (0): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['a'],
 out_keys=['e'])
 ),
 in_keys=['a'],
 out_keys=['e'])
```

This method propagates to nested sequential:

```
>>> module = Seq(
... Seq(
... Mod(idn, in_keys=["a"], out_keys=["b"]),
... Mod(idn, in_keys=["b"], out_keys=["c"]),
... ),
... Seq(
... Mod(idn, in_keys=["b"], out_keys=["d"]),
... Mod(idn, in_keys=["d"], out_keys=["e"]),
... ),
... )
>>> # select submodules whose output will be affected by a change in "b" or "d" AND which output is "e"
>>> module.select_subsequence(in_keys=["b", "d"], out_keys=["e"])
TensorDictSequential(
 module=ModuleList(
 (0): TensorDictSequential(
 module=ModuleList(
 (0): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['b'],
 out_keys=['d'])
 (1): TensorDictModule(
 module=<function <lambda> at ...>,
 in_keys=['d'],
 out_keys=['e'])
 ),
 in_keys=['b'],
 out_keys=['d', 'e'])
 ),
 in_keys=['b'],
 out_keys=['d', 'e'])
```

## Tutorials that use `TensorDictSequential`

![](../../_images/sphx_glr_export_thumb.png)

[Exporting tensordict modules](../../tutorials/export.html)

Exporting tensordict modules
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule