# ProbabilisticTensorDictSequential

*class*tensordict.nn.ProbabilisticTensorDictSequential(**modules: [TensorDictModuleBase](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [ProbabilisticTensorDictModule](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule)*, *partial_tolerant: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *return_composite: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *selected_out_keys: List[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)

A [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) with probabilistic modules, which can return their distribution.

Usually the last module is a
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) and the modules
before it compute its parameters. A call runs every module and writes the
sampled values (see [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule)
for the keys of the log-probabilities). `get_dist()` returns the
distribution, and `log_prob()` returns the log-probability of a
sample. See [Probabilistic modules](../../modules.html#modules-probabilistic) for a guide.

When a module samples from a distribution whose parameters depend on the
sample of an earlier module, its log-probability is conditional: for a
value `z` drawn from a distribution whose parameters are computed from
earlier samples `x` and `y`, it is `log p(z | x, y)`.

Parameters:

***modules** ([*TensorDictModuleBase*](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*or**Callable*) - the modules to run, in
order, usually ending with a
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule). You can
also pass one list, or one `OrderedDict` or `dict` that maps
names to modules. A callable that is not a
[`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) takes and returns a
tensordict, and its keys do not count in the `in_keys` and
`out_keys` of the sequence.

Keyword Arguments:

- **partial_tolerant** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - stored as an attribute, but
`forward()` does not use it: every module runs on the whole
input, as with `partial_tolerant=False`. To skip the modules whose
input keys are missing, or to run them on the members of a lazy
stack that have the keys, put this module in a
[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) with
`partial_tolerant=True`. Defaults to `False`.
- **return_composite** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - what `get_dist()` returns.
If `True`, one
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) that
holds the distributions of all the probabilistic modules. If
`False`, the distribution of the last module: the other
probabilistic modules only write their values, and the last module
must be probabilistic. Defaults to `None`, which gives `True`
if there is more than one probabilistic module or if the last
module is not probabilistic, and `False` otherwise.
- **selected_out_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the `out_keys` to
write. Defaults to `None`, which writes all the `out_keys`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - 

where the sequence writes its
`out_keys` after the last module ran.

- `True`: into the input tensordict, which is returned.
- `False`: into a new, empty [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).
- `"empty"`: into `input.empty()`, which keeps the type, device
and batch size of the input.
- `None` (default): the sequence returns what the last module
returns, so the `inplace` value of each module decides.

With `False` or `"empty"`, the modules run on a shallow copy
of the input, so the input does not change.

Where the output lacks the sub-tensordict of a nested out key, the
sequence writes that whole sub-tensordict, entries that are not
out keys included, and warns if it holds any. Starting with
TensorDict 0.17, the default will change to writing only the
`out_keys`.

Raises:

- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `modules` is empty.
- [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `return_composite=False` and the last module is not a
 [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) or a
 `ProbabilisticTensorDictSequential`.

Examples

```
>>> # Typical usage: a single distribution is computed last in the sequence
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import (
... ProbabilisticTensorDictModule as Prob,
... ProbabilisticTensorDictSequential as Seq,
... TensorDictModule as Mod,
... )
>>> _ = torch.manual_seed(0)
>>>
>>> module = Seq(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
... Prob(in_keys=["loc"], out_keys=["sample"], distribution_class=torch.distributions.Normal,
... distribution_kwargs={"scale": 1}),
... )
>>> input = TensorDict(x=torch.ones(3))
>>> td = module(input.copy())
>>> print(td)
TensorDict(batch_size=[], fields={loc: f32[3], sample: f32[3], x: f32[3]})
>>> print(module.get_dist(input))
Normal(loc: torch.Size([3]), scale: torch.Size([3]))
>>> print(module.log_prob(td))
tensor([-0.9189, -0.9189, -0.9189])
>>> # Intermediate distributions are ignored when return_composite=False
>>> module = Seq(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
... Prob(in_keys=["loc"], out_keys=["sample0"], distribution_class=torch.distributions.Normal,
... distribution_kwargs={"scale": 1}),
... Mod(lambda x: x + 1, in_keys=["sample0"], out_keys=["loc2"]),
... Prob(in_keys={"loc": "loc2"}, out_keys=["sample1"], distribution_class=torch.distributions.Normal,
... distribution_kwargs={"scale": 1}),
... return_composite=False,
... )
>>> td = module(TensorDict(x=torch.ones(3)))
>>> print(td)
TensorDict(batch_size=[], fields={
 loc2: f32[3],
 loc: f32[3],
 sample0: f32[3],
 sample1: f32[3],
 x: f32[3]})
>>> print(module.get_dist(input))
Normal(loc: torch.Size([3]), scale: torch.Size([3]))
>>> print(module.log_prob(td))
tensor([-0.9189, -0.9189, -0.9189])
>>> # Intermediate distributions produce a CompositeDistribution when return_composite=True
>>> module = Seq(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
... Prob(in_keys=["loc"], out_keys=["sample0"], distribution_class=torch.distributions.Normal,
... distribution_kwargs={"scale": 1}),
... Mod(lambda x: x + 1, in_keys=["sample0"], out_keys=["loc2"]),
... Prob(in_keys={"loc": "loc2"}, out_keys=["sample1"], distribution_class=torch.distributions.Normal,
... distribution_kwargs={"scale": 1}),
... return_composite=True,
... )
>>> input = TensorDict(x=torch.ones(3))
>>> td = module(input.copy())
>>> print(td)
TensorDict(batch_size=[], fields={
 loc2: f32[3],
 loc: f32[3],
 sample0: f32[3],
 sample1: f32[3],
 x: f32[3]})
>>> print(module.get_dist(input))
CompositeDistribution({'sample0': Normal(loc: torch.Size([3]), scale: torch.Size([3])), 'sample1': Normal(loc: torch.Size([3]), scale: torch.Size([3]))})
>>> print(module.log_prob(td))
TensorDict(batch_size=[], fields={
 sample0_log_prob: f32[3],
 sample1_log_prob: f32[3]})
>>> # Even a single intermediate distribution is wrapped in a CompositeDistribution when
>>> # return_composite=True
>>> module = Seq(
... Mod(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
... Prob(in_keys=["loc"], out_keys=["sample0"], distribution_class=torch.distributions.Normal,
... distribution_kwargs={"scale": 1}),
... Mod(lambda x: x + 1, in_keys=["sample0"], out_keys=["y"]),
... return_composite=True,
... )
>>> td = module(TensorDict(x=torch.ones(3)))
>>> print(td)
TensorDict(batch_size=[], fields={
 loc: f32[3],
 sample0: f32[3],
 x: f32[3],
 y: f32[3]})
>>> print(module.get_dist(input))
CompositeDistribution({'sample0': Normal(loc: torch.Size([3]), scale: torch.Size([3]))})
>>> print(module.log_prob(td))
TensorDict(batch_size=[], fields={sample0_log_prob: f32[3]})
```

build_dist_from_params(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Builds the distribution of the last probabilistic module from parameters, without running the other modules.

The method calls [`ProbabilisticTensorDictModule.get_dist()`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule.get_dist) of the
last [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) in the
sequence, nested sequences included. See the example of
`get_dist_params()`.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - a tensordict with the distribution
parameters at the `in_keys` of that module.

Returns:

A `torch.distributions.Distribution`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the sequence has no
 [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule).

*property*default_interaction_type

The `default_interaction_type` of the last module that has one.

If no module has one, the value of
[`interaction_type()`](tensordict.nn.interaction_type.html#tensordict.nn.interaction_type), which is `None` outside of a
[`set_interaction_type`](tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type) block.

*property*det_part

The modules that compute the distribution parameters.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17. Use
`get_dist_params()` to compute the distribution parameters.

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

The device of the first parameter of the module, or `cpu` if it has no parameter.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17 because modules may
contain parameters and buffers on multiple devices.

*property*dist_params_keys*: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

The keys of the distribution parameters of all the probabilistic modules, those of the last module first.

*property*dist_sample_keys*: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

The keys of the sampled values of all the probabilistic modules, those of the last module first.

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_requires_sample: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Runs the modules in order, and returns the tensordict with their outputs.

If the modules before the last one already write the sampled keys of
the last module, the last module does not sample again.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - if given, the modules
run on a shallow copy of `tensordict`, and the `out_keys`
are written into `tensordict_out`, which is returned (unless
`inplace` is set). Defaults to `None`.

Keyword Arguments:

****kwargs** - passed to the modules that are not
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule), with
either value of `return_composite`.

Returns:

The tensordict that holds the outputs (see `inplace`).

get_dist(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Returns the distribution that the sequence samples from, given the input `tensordict`.

If `return_composite` is `False`, this is the distribution of the
last module. The method runs the modules before it with
`get_dist_params()`, so the intermediate entries are written into
`tensordict`.

If `return_composite` is `True`, this is a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) that holds
the distributions of all the probabilistic modules, at their sample
keys. The method runs the modules on a shallow copy of
`tensordict`, so the input does not get new entries. Each
probabilistic module draws a value, which the modules after it read,
so their distributions are conditional on it. Only a probabilistic
module in the last position does not draw.
See the examples of `ProbabilisticTensorDictSequential`.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input, with the entries that the
first module reads.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - used only when
`return_composite` is `False`: the tensordict where the
intermediate entries are written, instead of `tensordict`.
Defaults to `None`.

Keyword Arguments:

****kwargs** - ignored when `return_composite` is `True`. When it
is `False`, passed to `get_dist_params()`.

Returns:

A `torch.distributions.Distribution`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the sequence has no probabilistic module.

get_dist_params(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Runs every module except the last one, and returns the tensordict with the distribution parameters.

The modules run with the interaction type of
[`interaction_type()`](tensordict.nn.interaction_type.html#tensordict.nn.interaction_type) if it is set, and otherwise
with the `default_interaction_type` of the last module that has
one, so that the probabilistic modules among them sample as in a
call to the sequence.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input, with the entries that the
first module reads.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the tensordict where
the parameters are written. If `None`, they are written into
`tensordict`, which is returned. Pass
`tensordict.copy()` to keep the input unchanged.
Defaults to `None`.

Keyword Arguments:

****kwargs** - passed to the modules that are not
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule), as in
`forward()` with `return_composite=True`.

Returns:

The tensordict that holds the distribution parameters.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import (
... ProbabilisticTensorDictModule,
... ProbabilisticTensorDictSequential,
... TensorDictModule,
... )
>>> seq = ProbabilisticTensorDictSequential(
... TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
... ProbabilisticTensorDictModule(
... in_keys=["loc"], out_keys=["y"], distribution_class=torch.distributions.Normal,
... distribution_kwargs={"scale": 1.0},
... ),
... )
>>> params = seq.get_dist_params(TensorDict(x=torch.zeros(2)))
>>> sorted(params.keys())
['loc', 'x']
>>> seq.build_dist_from_params(params)
Normal(loc: torch.Size([2]), scale: torch.Size([2]))
```

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

log_prob(*tensordict*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *dist: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the log-probability of the sampled values in `tensordict`.

If `return_composite` is `True`, the distribution is a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) and the
method returns its [`log_prob()`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution.log_prob):
while [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `False`, a
tensordict with one entry per sampled key, at the key where a call to
the sequence writes it (see `log_prob_keys`); while it is
`True`, the sum, as a tensor.

If `return_composite` is `False`, the method returns the
log-probability of the value of the last module, as
[`ProbabilisticTensorDictModule.log_prob()`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule.log_prob) does. The intermediate
entries are then written into `tensordict`, as in `get_dist()`.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input, with the sampled values.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - if given, the
distribution is computed on a copy of `tensordict`, so the
input does not get new entries. Defaults to `None`.

Keyword Arguments:

- **dist** (*torch.distributions.Distribution**,**optional*) - the
distribution. Defaults to `None`, which builds it with
`get_dist()`.
- ****kwargs** - if `dist` is `None`, `return_composite` is
`False` and the last module does not override
[`ProbabilisticTensorDictModule.log_prob()`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule.log_prob), passed to
`get_dist()`. Otherwise, passed to the `log_prob` method
of the distribution, or of the last module.

Returns:

A tensor, or a tensordict for a composite distribution while
[`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `False`.
See the examples of `ProbabilisticTensorDictSequential`.

*property*log_prob_key

The log-probability key of the sequence, when `log_prob_keys` has exactly one element.

See the [log-probability key table](tensordict.nn.ProbabilisticTensorDictModule.html#log-prob-keys).

Raises:

[**AttributeError**](https://docs.python.org/3/builtins/exceptions.html#AttributeError) - if `log_prob_keys` does not have exactly one
 element (use `log_prob_keys`), or if
 [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) changed since a
 module was built. The error message names the inner
 `ModuleList`.

*property*log_prob_keys

The log-probability keys of all the probabilistic modules, those of the last module first.

See the [log-probability key table](tensordict.nn.ProbabilisticTensorDictModule.html#log-prob-keys).

*property*num_samples

The leading dimensions that the probabilistic modules add to the output.

The `num_samples` of each probabilistic module, those of the last
module first, as they appear in the shape of the output. An empty
tuple if no module sets `num_samples`.

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

select_out_keys(**selected_out_keys*) → [TensorDictSequential](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)

Keeps only some of the `out_keys` in the output of the sequence, and returns the sequence.

The sequence changes in place, and `reset_out_keys()` undoes the
change. The modules still compute all their outputs; the call then
writes only the selected ones, and the entries of the input, into the
output.
Where the output lacks the sub-tensordict of a selected nested key, the
call writes that whole sub-tensordict, and warns if it holds other
entries. Starting with TensorDict 0.17, the default will change to
writing only the selected keys.

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

select_subsequence(*in_keys: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out_keys: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictSequential](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)

Returns a new sequence with only the modules that can run from `in_keys` and that `out_keys` need.

Two filters are applied, in this order:

- `in_keys`: a module is kept if each of its inputs is in
`in_keys` or is written by a module kept before it. The other
modules, and those that read their outputs, are dropped.
- `out_keys`: of the modules left, a module is kept if it writes one
of `out_keys`, or an input of a module kept after it.

A nested [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) is filtered the same way. The
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

## Tutorials that use `ProbabilisticTensorDictSequential`

![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule