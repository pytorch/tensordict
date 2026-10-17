# ProbabilisticTensorDictSequential

*class*tensordict.nn.ProbabilisticTensorDictSequential(**modules: [TensorDictModuleBase](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [ProbabilisticTensorDictModule](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule)*, *partial_tolerant: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *return_composite: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *selected_out_keys: List[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)

A [`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) that contains one or more [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) instances, and can return their distribution.

Usually the last module is probabilistic and the modules before it
compute its parameters. `get_dist()` returns the distribution, and
`log_prob()` returns the log-probability of a sample. A call to the
module runs every module and writes the sampled values (see
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) for the keys of the
log-probabilities).

`return_composite` decides what `get_dist()` returns when there is
more than one probabilistic module:

- `False`: the distribution of the last module. The other probabilistic
modules only write their values. The last module must be
probabilistic, otherwise the constructor raises `TypeError`.
- `True`: one [`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution)
that holds the distributions of all probabilistic modules.

The default chooses `True` if there is more than one probabilistic
module or if the last module is not probabilistic, and `False`
otherwise.

Resulting log-probabilities will be conditional probabilities if samples are interdependent:
whenever

> \[Z = F(X, Y)\]

then the log-probability of Z will be

> \[log(p(z | x, y))\]

Parameters:

***modules** (*sequence**or**OrderedDict**of*[*TensorDictModuleBase*](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*or*[*ProbabilisticTensorDictModule*](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule)) - An ordered sequence of
[`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) instances, usually terminating in a [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule),
to be run sequentially.
The modules can be instances of TensorDictModuleBase or any other function that matches this signature.
Note that if a non-TensorDictModuleBase callable is used, its input and output keys will not be tracked,
and thus will not affect the in_keys and out_keys attributes of the TensorDictSequential.

Keyword Arguments:

- **partial_tolerant** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - stored as an attribute, but
`forward()` does not use it: every module runs on the whole
input, as with `partial_tolerant=False`. To skip the modules whose
input keys are missing, or to run them on the members of a lazy
stack that have the keys, put this module in a
[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) with
`partial_tolerant=True`. Defaults to `False`.
- **return_composite** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If True and multiple
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) or
`ProbabilisticTensorDictSequential` instances are found,
a [`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) instance will be used.
Otherwise, only the last module will be used to build the distribution.
Defaults to `True` whenever there are more than one probabilistic modules or the last module is not probabilistic.
Errors if return_composite is False and the last module is not probabilistic.
- **selected_out_keys** (*iterable**of**NestedKeys**,**optional*) - the list of out-keys to select. If not provided, all
`out_keys` will be written.
- **inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if True, the input tensordict is modified in-place. If False, a new empty
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) instance is created. If "empty", input.empty() is used instead (ie, the
output preserves type, device and batch-size). Defaults to None (relies on sub-modules).

Raises:

- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - If the input sequence of modules is empty.
- [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - If `return_composite=False` and the final module is not an instance of
 [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) or
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
TensorDict(
 fields={
 loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 sample: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
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
TensorDict(
 fields={
 loc2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 sample0: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 sample1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
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
TensorDict(
 fields={
 loc2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 sample0: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 sample1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(module.get_dist(input))
CompositeDistribution({'sample0': Normal(loc: torch.Size([3]), scale: torch.Size([3])), 'sample1': Normal(loc: torch.Size([3]), scale: torch.Size([3]))})
>>> print(module.log_prob(td))
TensorDict(
 fields={
 sample0_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 sample1_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
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
TensorDict(
 fields={
 loc: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 sample0: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 x: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 y: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(module.get_dist(input))
CompositeDistribution({'sample0': Normal(loc: torch.Size([3]), scale: torch.Size([3]))})
>>> print(module.log_prob(td))
TensorDict(
 fields={
 sample0_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

build_dist_from_params(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Constructs a distribution from the input parameters without evaluating other modules in the sequence.

This method searches for the last [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) in the sequence and uses it to build the distribution.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input tensordict containing the distribution parameters.

Returns:

The constructed distribution object.

Return type:

D.Distribution

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - If no [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) is found in the sequence.

*property*default_interaction_type

Returns the default_interaction_type of the module using an iterative heuristic.

This property iterates over all modules in reverse order, attempting to retrieve the
default_interaction_type attribute from any child module. The first non-None value
encountered is returned. If no such value is found, a default interaction_type() is returned.

*property*det_part

The modules that compute the distribution parameters.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17. Use
`get_dist_params()` to compute the distribution parameters.

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

A deprecated approximation of the module device.

Deprecated since version 0.15: This property will be removed in TensorDict 0.17 because modules may
contain parameters and buffers on multiple devices.

*property*dist_params_keys*: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

Returns all the keys pointing at the distribution params.

*property*dist_sample_keys*: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

Returns all the keys pointing at the distribution samples.

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_requires_sample: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

get_dist(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Returns the distribution resulting from passing the input tensordict through the sequence.

If return_composite is `False`, this method will only consider the last probabilistic module in the sequence.
It runs the modules before it with `get_dist_params()`, so the
intermediate entries are written into `tensordict`.

Otherwise, it will return a [`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) instance containing the distributions of all probabilistic modules.
It runs the modules on a shallow copy of `tensordict`, so the input
does not get new entries.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input tensordict.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - Used only when
`return_composite` is `False`: the tensordict where the
intermediate entries are written, instead of `tensordict`.
Defaults to `None`.

Keyword Arguments:

****kwargs** - Additional keyword arguments passed to the underlying modules.

Returns:

The resulting distribution object.

Return type:

D.Distribution

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - If no probabilistic module is found in the sequence.

Note

When return_composite is `True`, the distributions are conditioned on the previous samples in the sequence.
This means that if a module depends on the output of a previous probabilistic module, its distribution will be conditional.

get_dist_params(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution), [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)]

Runs every module except the last one, and returns the tensordict with the distribution parameters.

This method runs the deterministic part of the `ProbabilisticTensorDictSequential`
module to obtain the distribution parameters. The interaction type is set to the current global
interaction type if available, otherwise it defaults to the interaction type of the last module.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input tensordict.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - The tensordict where
the parameters are written. If `None`, they are written into
`tensordict`, which is returned. Pass
`tensordict.copy()` to keep the input unchanged.
Defaults to `None`.

Keyword Arguments:

****kwargs** - Additional keyword arguments passed to the deterministic part of the module.

Returns:

the output tensordict containing the distribution parameters.

Return type:

[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Note

The interaction type is temporarily set to the specified value during the execution of this method.

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

log_prob(*tensordict*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *dist: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the log-probability of the input tensordict.

If self.return_composite is `True` and the distribution is a [`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution),
this method will return the log-probability of the entire composite distribution.

Otherwise, it will only consider the last probabilistic module in the sequence.
In that case the intermediate entries are written into `tensordict`,
as in `get_dist()`.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input tensordict, with the
sampled values.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - If given, the
distribution is computed on a copy of `tensordict`, so the
input does not get new entries. Defaults to `None`.

Keyword Arguments:

**dist** (*torch.distributions.Distribution**,**optional*) - The distribution object. If `None`, it will be computed using get_dist.
Defaults to `None`.

Returns:

The log-probability of the input tensordict.

Return type:

[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase) or [*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

*property*log_prob_key

The key where the log-probability is written, when there is one such key.

See the [log-probability key table](tensordict.nn.ProbabilisticTensorDictModule.html#log-prob-keys).

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate)
 changed since the module was built, or if it is `False` and
 the module has several sampled keys (use `log_prob_keys`).

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

select_out_keys(**selected_out_keys*) → [TensorDictSequential](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)

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

select_subsequence(*in_keys: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out_keys: [Iterable](https://docs.python.org/3/library/typing.html#typing.Iterable)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictSequential](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)

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
Modules in a named sequence keep their names, including nested sequences.
The new sequence keeps `partial_tolerant` and `inplace`.
Nested sequences keep their own options the same way.

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

## Tutorials that use `ProbabilisticTensorDictSequential`

![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule