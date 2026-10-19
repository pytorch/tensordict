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

The modules are given as positional arguments, as one list, or as one
dict (or `OrderedDict`) that maps names to modules. `seq[i]` and
`seq["name"]` return a module, and a slice returns a new sequence with
the same options. Assigning or deleting `seq[i]` or `seq["name"]`
edits the sequence (a new name appends the module), and `in_keys` and
`out_keys` are recomputed. A module registered under several names
runs once for each name.

Parameters:

***modules** ([*TensorDictModuleBase*](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*or**Callable*) - the modules to run, in
order. A callable without `in_keys` and `out_keys`, such as a
function that takes and returns a tensordict, is wrapped in a
[`WrapModule`](tensordict.nn.WrapModule.html#tensordict.nn.WrapModule). Its keys are then not part of
the `in_keys` and `out_keys` of the sequence.

Keyword Arguments:

- **partial_tolerant** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the input can lack
some of the `in_keys`: a module runs only if all its inputs are
present, and is skipped otherwise. On a lazy stack, a module that
cannot run on the whole stack runs on each tensordict of the stack
that has its inputs, and its outputs are written into that
tensordict. Defaults to `False`.
- **selected_out_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the `out_keys` to
keep in the output, as with `select_out_keys()`. Each one must
be in `out_keys`. Defaults to `None`, which keeps all of them.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - 

where the sequence writes its `out_keys`
after the last module ran.

- `True`: into the input tensordict, which is returned.
- `False`: into a new, empty [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).
- `"empty"`: into `input.empty()`, which keeps the type, device
and batch size of the input.
- `None`: the sequence returns what the last module returns,
so the `inplace` value of each module decides.

With `False` or `"empty"`, the modules run on a shallow copy
of the input, so the input does not change. See the last example.
Defaults to `None`.

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

A sequence can hold probabilistic modules, and be called with other
parameters through [`to_module()`](tensordict.TensorDictBase.to_module.html#tensordict.TensorDictBase.to_module), also
under [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap):

```
>>> from tensordict.nn import (
... ProbabilisticTensorDictModule,
... ProbabilisticTensorDictSequential,
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
>>> sorted(td.keys())
['hidden', 'hidden_log_prob', 'input', 'loc', 'output', 'params', 'scale']
>>> td["output"].shape
torch.Size([3, 8])
```

With [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) over 4 copies of the parameters, each entry gets
a leading dimension of size 4:

```
>>> from torch import vmap
>>> params = params.expand(4)
>>> def func(td, params):
... with params.to_module(td_module):
... return td_module(td)
>>> td_vmap = vmap(func, (None, 0))(td, params)
>>> td_vmap.batch_size
torch.Size([4, 3])
>>> td_vmap["output"].shape
torch.Size([4, 3, 8])
```

The `inplace` argument controls where the outputs go. In this example the
inner modules use `inplace=False`, so each one passes only its outputs to
the next. The sequence uses `inplace=True`, so it writes the final
`output` into the input object. The intermediate entries are not written
into it.

```
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

The device of the first parameter of the module, or `cpu` if it has no parameter.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17 because modules may
contain parameters and buffers on multiple devices.

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Runs the modules in order on `tensordict` and returns the result (see `inplace`).

If you call the sequence with tensors instead of a tensordict, the
call returns the output tensors (see [`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch)).

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input tensordict.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - if given, the
`out_keys` are written into it and it is returned, whatever
the value of `inplace`. Defaults to `None`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if other keyword arguments are given.

*property*in_keys

The keys that the module reads from the input tensordict.

*property*is_functional*: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the wrapped module is a legacy functorch functional module.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17. It only checks
for the `FunctionalModule` classes of the legacy
`functorch.make_functional` API and has no replacement.

*static*is_tdmodule_compatible(*module*)

Returns `True` if `module` has both an `in_keys` and an `out_keys` attribute.

`TensorDictSequential` wraps the callables that
fail this check in a [`WrapModule`](tensordict.nn.WrapModule.html#tensordict.nn.WrapModule).

Parameters:

**module** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) - the object to check.

*property*out_keys

The keys that the modules write, each listed once, or those chosen with `select_out_keys()`.

*property*out_keys_source

All the keys that the module writes, before `select_out_keys()`.

reset_out_keys()

Undoes `select_out_keys()`: the sequence writes all its `out_keys` again, and returns `self`.

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

select_out_keys(**selected_out_keys*) → TensorDictSequential

Keeps only some of the `out_keys` in the output of the sequence, and returns the sequence.

The sequence changes in place, and `reset_out_keys()` undoes the
change. The modules still compute all their outputs; the call then
writes only the selected ones, and the entries of the input, into the
output.

Parameters:

***selected_out_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys to keep. Each one must be
in `out_keys`.

Returns:

`self`.

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
>>> "y" in td
False
>>> td["z"]
tensor(2.)
>>> _ = seq.reset_out_keys()
>>> seq.out_keys
['y', 'z']
```

select_subsequence(*in_keys: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out_keys: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → TensorDictSequential

Returns a new sequence with only the modules that can run from `in_keys` and that `out_keys` need.

Two filters are applied, in this order:

- `in_keys`: a module is kept if each of its inputs is in
`in_keys` or is written by a module kept before it. The other
modules, and those that read their outputs, are dropped.
- `out_keys`: of the modules left, a module is kept if it writes one
of `out_keys`, or an input of a module kept after it.

A nested `TensorDictSequential` is filtered the same way. The
modules are not copied: the new sequence holds the same module
objects. Modules of a named sequence keep their names, and the new
sequence keeps `partial_tolerant` and `inplace`.

Parameters:

- **in_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the keys that the modules
can start from. Defaults to `None`, which uses the
`in_keys` of the sequence.
- **out_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the keys to compute.
Defaults to `None`, which uses the `out_keys` of the
sequence.

Returns:

A new sequence of the same type.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if no module is left.

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
>>> # the modules that can run from "c"
>>> module.select_subsequence(in_keys=["c"]).out_keys
['d']
>>> # the modules that "c" depends on
>>> sub = module.select_subsequence(out_keys=["c"])
>>> sub.in_keys, sub.out_keys
(['a'], ['b', 'c'])
>>> sub[0] is module[0]
True
```

Nested sequences are filtered too:

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
>>> sub = module.select_subsequence(in_keys=["b", "d"], out_keys=["e"])
>>> len(sub), sub[0].out_keys
(1, ['d', 'e'])
```

## Tutorials that use `TensorDictSequential`

![](../../_images/sphx_glr_export_thumb.png)

[Exporting tensordict modules](../../tutorials/export.html)

Exporting tensordict modules
![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule