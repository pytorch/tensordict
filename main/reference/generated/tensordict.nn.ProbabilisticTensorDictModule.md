# ProbabilisticTensorDictModule

*class*tensordict.nn.ProbabilisticTensorDictModule(*in_keys: NestedKey | List[NestedKey] | Dict[str, NestedKey], out_keys: NestedKey | List[NestedKey] | None = None, *, default_interaction_type: InteractionType = InteractionType.DETERMINISTIC, distribution_class: type = <class 'tensordict.nn.distributions.continuous.Delta'>, distribution_kwargs: dict | None = None, return_log_prob: bool = False, log_prob_keys: List[NestedKey] | None = None, log_prob_key: NestedKey | None = None, cache_dist: bool = False, n_empirical_estimate: int = 1000, num_samples: int | torch.Size | None = None, generator: torch.Generator | int | NestedKey | None = None*)

Bases: [`TensorDictModuleBase`](tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)

A module that builds a distribution from entries of a tensordict and writes a value from it.

The module has no parameters. It reads the distribution parameters at
`in_keys`, builds `distribution_class(**params, **distribution_kwargs)`,
and writes one value at each of the `out_keys`. To compute the parameters
with a network, put a [`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) before this
module in a [`ProbabilisticTensorDictSequential`](tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential). To
get the distribution without a value, call `get_dist()`. To get the
log-probability of a value, call `log_prob()`. See
[Probabilistic modules](../../modules.html#modules-probabilistic) for a guide.

The value depends on the interaction type: the value of
[`interaction_type()`](tensordict.nn.interaction_type.html#tensordict.nn.interaction_type) if it is set (see
[`set_interaction_type`](tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type)), `default_interaction_type`
otherwise. See [`InteractionType`](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) for what each type
reads from the distribution.

The default `distribution_class` is
[`Delta`](tensordict.nn.distributions.Delta.html#tensordict.nn.distributions.Delta), which always returns its
parameter. With it, the module copies its input to `out_keys`.

If every key in `out_keys` is already in the input and
[`skip_existing()`](tensordict.nn.skip_existing.html#tensordict.nn.skip_existing) is `True` (see
[`set_skip_existing`](tensordict.nn.set_skip_existing.html#tensordict.nn.set_skip_existing)), the module returns the input
unchanged. Otherwise it computes a new value and overwrites the entry.

With `return_log_prob=True` and the `RANDOM` interaction type, the
module draws and scores the value in one call if the distribution defines
the method: `dist.rsample_and_log_prob(sample_shape)` if
`dist.has_rsample`, `dist.sample_and_log_prob(sample_shape)`
otherwise. The method returns a `(sample, log_prob)` tuple, with
`log_prob` in the layout of `dist.log_prob`. Define it when
`dist.log_prob(sample)` cannot recover the draw exactly, for example
for a tanh-squashed normal that saturates. Otherwise the module calls
`dist.log_prob` on the sample. The [`torch.distributions`](https://docs.pytorch.org/docs/stable/distributions.html#module-torch.distributions) classes
do not define these methods;
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) defines both.

Log-probability keys

With `return_log_prob=True`, the module also writes the log-probability
of the value. The key depends on [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate)
(`False` by default, see [`set_composite_lp_aggregate`](tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate))
and on the number of sampled keys:

| Case | `composite_lp_aggregate()` is `False` (default) | `composite_lp_aggregate()` is `True` |
| --- | --- | --- |
| One sampled key, `"action"` | `"action_log_prob"` | `"sample_log_prob"` |
| One nested key, `("agent", "action")` | `("agent", "action_log_prob")` | `"sample_log_prob"` |
| Several sampled keys from a [`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution), `"cont"` and `("nested", "disc")` | `"cont_log_prob"` and `("nested", "disc_log_prob")` | The same two entries, and `"sample_log_prob"` with their sum |
| Several probabilistic modules in a [`ProbabilisticTensorDictSequential`](tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential) | One `<key>_log_prob` entry per sampled key | Each module writes `"sample_log_prob"`, so the entry holds the log-probability of the last module only. Call [`ProbabilisticTensorDictSequential.log_prob()`](tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential.log_prob) to get the sum. |

The suffix `_log_prob` goes on the last element of a nested key. Pass
`log_prob_key` (one sampled key) or `log_prob_keys` (several sampled
keys) to choose other keys.

`log_prob()` returns a tensor, except for a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) while
`composite_lp_aggregate()` is `False`: then it returns a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with the `<key>_log_prob` entries of the
table. With `True`, the tensor is the sum over the sampled keys and their
event dimensions.

Set [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) before you build the
module. If the value changed since then, `log_prob_key`,
`log_prob_keys`, `log_prob()` and a call with
`return_log_prob=True` can raise a `RuntimeError`.

Parameters:

- **in_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*, or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the entries that hold
the parameters of the distribution. With a key or a list of keys,
the last element of each key is the keyword argument of
`distribution_class` that it feeds: `["loc", "scale"]` for a
`Normal`, and `("agent", "loc")`
feeds `loc`. With a dict, the keys are the keyword arguments and
the values are the tensordict keys:
`{"loc": "mu", "scale": "sigma"}`.
- **out_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the keys where the
values are written, one per value that the distribution returns.
If the input already has these keys, the module overwrites them,
unless [`skip_existing()`](tensordict.nn.skip_existing.html#tensordict.nn.skip_existing) is `True`. Defaults
to `None`: for a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution), the
values of the `name_map` of `distribution_kwargs` if it has
one, and the keys of its `distribution_map` otherwise. A
component that `name_map` leaves out is written under its own
name but is not in `out_keys`, and the module warns. Starting
with TensorDict 0.17, the default will change to `out_keys` that
include these components. For another distribution, `None`
gives the key `"_"`: the module writes the sample there, and
warns the first time it does.
Starting with TensorDict 0.17, the default will change to not
writing the sample at `"_"`, which marks an output that is not
written, as in [`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule). Pass
`out_keys` to keep the current keys.

Keyword Arguments:

- **default_interaction_type** ([*InteractionType*](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the
interaction type that the module uses when
[`interaction_type()`](tensordict.nn.interaction_type.html#tensordict.nn.interaction_type) returns `None`, that is
outside of a [`set_interaction_type`](tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type) block.
`MODE` and `MEDIAN` raise `NotImplementedError` if the
distribution does not have them. `MEAN` averages
`n_empirical_estimate` samples if the distribution has no mean.
TorchRL collectors sample with `InteractionType.RANDOM` by
default (their `exploration_type` argument). Defaults to
`InteractionType.DETERMINISTIC`.
- **distribution_class** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*or**Callable**,**optional*) - the distribution
class, or a callable that takes the parameters as keyword arguments
and returns a `torch.distributions.Distribution`. For a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution),
`distribution_kwargs` holds its `distribution_map`, and
`out_keys` can be left out. Defaults to
[`Delta`](tensordict.nn.distributions.Delta.html#tensordict.nn.distributions.Delta).
- **distribution_kwargs** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - extra keyword arguments passed
to `distribution_class`, for example constants. To move the
tensors in it with the module (`module.to(device)`,
`module.to(dtype)`), pass a [`TensorDictParams`](tensordict.TensorDictParams.html#tensordict.TensorDictParams).
Defaults to `None`.
- **return_log_prob** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the module also writes
the log-probability of the value, at the keys of the table above.
Defaults to `False`.
- **log_prob_keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the keys of the
log-probabilities, one per sampled key, in place of the
`<key>_log_prob` defaults, including for a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution). If
[`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `True`, passing
it raises a `RuntimeError`. Defaults to `None`.
- **log_prob_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - 

the key of the log-probability:
of the one sampled key, or, while
[`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `True`, of their
sum. With several sampled keys while it is `False`, passing it
raises a `RuntimeError`. Defaults to `None`, which gives the
key of the table above.

With `return_log_prob=True`, a custom `log_prob_key` or
`log_prob_keys` entry equal to a sample key, inside one, or
holding one raises a `ValueError`, because the log-probability
would overwrite the sample. The check runs when
`distribution_class` is a class (and, for a
`CompositeDistribution`, when `distribution_kwargs` holds its
`distribution_map`).
- **cache_dist** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - has no effect. Deprecated: passing
`True` warns, and the argument will be removed in TensorDict
0.17. Defaults to `False`.
- **n_empirical_estimate** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of samples that
`MEAN` averages when the distribution has no mean. Defaults to
`1000`.
- **num_samples** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the number of values per
call. The output tensordict is expanded to
`num_samples + batch_size`, and each value gets these leading
dimensions. With the `RANDOM` interaction type, each value is a
new draw. With another interaction type, the module repeats its
one deterministic value. Defaults to `None`: one value, no extra
dimension.
- **generator** ([*Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - 

the random
number generator of the sampling, used by the `RANDOM`
interaction type and by the sampled estimate of `MEAN`.

- `None` (default): the global PyTorch generator.
- a [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator): used for every call. Its state
advances in place, so consecutive calls draw different values.
It must be on the device of the distribution parameters.
- an `int`: the seed of a CPU generator that the constructor
creates, then used as above.
- a `NestedKey`: the key of an entry of the input, read at each
call. The entry holds a [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator), used as above,
or an integer seed (a scalar tensor or an `int`). A seed gives
a new generator, and after sampling the module writes a new seed,
drawn from that generator, back at the key. If the input has
no such entry, the global generator is used.

Sampling with a generator does not change the state of the global
generator, and works with every distribution. Under
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile), the argument is ignored, without a warning.

Examples

Sample from a normal distribution, and write the log-probability of
the sample:

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import InteractionType, ProbabilisticTensorDictModule
>>> module = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"],
... out_keys=["action"],
... distribution_class=d.Normal,
... default_interaction_type=InteractionType.RANDOM,
... return_log_prob=True,
... )
>>> _ = torch.manual_seed(0)
>>> td = module(TensorDict(loc=torch.zeros(3), scale=torch.ones(3)))
>>> td["action"]
tensor([ 1.5410, -0.2934, -2.1788])
>>> td["action_log_prob"]
tensor([-2.1063, -0.9620, -3.2925])
```

Compute the parameters with a network, in a
[`ProbabilisticTensorDictSequential`](tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential):

```
>>> from tensordict.nn import (
... ProbabilisticTensorDictModule,
... ProbabilisticTensorDictSequential,
... TensorDictModule,
... )
>>> from tensordict.nn.distributions import NormalParamExtractor
>>> from torch.distributions import Normal, Independent
>>> td = TensorDict(
... {"input": torch.randn(3, 4), "hidden": torch.randn(3, 8)}, [3]
... )
>>> net = torch.nn.GRUCell(4, 8)
>>> module = TensorDictModule(
... net, in_keys=["input", "hidden"], out_keys=["params"]
... )
>>> normal_params = TensorDictModule(
... NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
... )
>>> def IndepNormal(**kwargs):
... return Independent(Normal(**kwargs), 1)
>>> prob_module = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"],
... out_keys=["action"],
... distribution_class=IndepNormal,
... return_log_prob=True,
... )
>>> td_module = ProbabilisticTensorDictSequential(
... module, normal_params, prob_module
... )
>>> params = TensorDict.from_module(td_module)
>>> with params.to_module(td_module):
... _ = td_module(td)
>>> print(td)
TensorDict(batch_size=[3], fields={
 action: f32[4],
 action_log_prob: f32[],
 hidden: f32[8],
 input: f32[4],
 loc: f32[4],
 params: f32[8],
 scale: f32[4]})
>>> with params.to_module(td_module):
... dist = td_module.get_dist(td)
>>> print(dist)
Independent(Normal(loc: torch.Size([3, 4]), scale: torch.Size([3, 4])), 1)
>>> # we can also apply the module to the TensorDict with vmap
>>> from torch import vmap
>>> params = params.expand(4)
>>> def func(td, params):
... with params.to_module(td_module):
... return td_module(td)
>>> td_vmap = vmap(func, (None, 0))(td, params)
>>> print(td_vmap)
TensorDict(batch_size=[4,3], fields={
 action: f32[4],
 action_log_prob: f32[],
 hidden: f32[8],
 input: f32[4],
 loc: f32[4],
 params: f32[8],
 scale: f32[4]})
```

build_dist_from_params(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Creates a `torch.distributions.Distribution` instance with the parameters provided in the input tensordict.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input, with the distribution
parameters at `in_keys`.

Returns:

A `torch.distributions.Distribution` instance created from the input tensordict.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if the entries of the input do not match the keyword
 arguments of the distribution.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import ProbabilisticTensorDictModule
>>> module = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"], out_keys=["action"], distribution_class=d.Normal
... )
>>> td = TensorDict(loc=torch.zeros(3), scale=torch.ones(3))
>>> dist = module.get_dist(td)
>>> dist
Normal(loc: torch.Size([3]), scale: torch.Size([3]))
>>> dist.entropy()
tensor([1.4189, 1.4189, 1.4189])
```

*property*dist_params_keys*: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

the `in_keys`.

Type:

The keys of the distribution parameters in the input

*property*dist_sample_keys*: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

the `out_keys` without the log-probability keys.

Type:

The keys of the sampled values

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_requires_sample: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Builds the distribution from `tensordict` and writes a value from it at `out_keys`.

With `return_log_prob=True`, the module also writes the
log-probability of the value. You can also call the module with the
parameters as keyword arguments, `module(loc=..., scale=...)`, and
get the outputs back (see [`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch)).

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input, with the distribution
parameters at `in_keys`.
- **tensordict_out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the tensordict where
the outputs are written. Defaults to `None`, which writes
them into `tensordict`.

Returns:

The tensordict that holds the outputs, `tensordict_out` or
`tensordict`. With `num_samples`, a new tensordict, which is
that tensordict expanded to `num_samples + batch_size`. The
input does not get the outputs then.

get_dist(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Creates a `torch.distributions.Distribution` instance with the parameters provided in the input tensordict.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input, with the distribution
parameters at `in_keys`.

Returns:

A `torch.distributions.Distribution` instance created from the input tensordict.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if the entries of the input do not match the keyword
 arguments of the distribution.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import ProbabilisticTensorDictModule
>>> module = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"], out_keys=["action"], distribution_class=d.Normal
... )
>>> td = TensorDict(loc=torch.zeros(3), scale=torch.ones(3))
>>> dist = module.get_dist(td)
>>> dist
Normal(loc: torch.Size([3]), scale: torch.Size([3]))
>>> dist.entropy()
tensor([1.4189, 1.4189, 1.4189])
```

*property*in_keys

The keys that the module reads from the input tensordict.

*static*is_tdmodule_compatible(*module*)

Returns `True` if `module` has both an `in_keys` and an `out_keys` attribute.

[`TensorDictSequential`](tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential) wraps the callables that
fail this check in a [`WrapModule`](tensordict.nn.WrapModule.html#tensordict.nn.WrapModule).

Parameters:

**module** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) - the object to check.

log_prob(*tensordict*, ***, *dist: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Computes the log-probability of the value stored in `tensordict`.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the input, with the distribution
parameters at `in_keys` and the value at `out_keys`.

Keyword Arguments:

**dist** (*torch.distributions.Distribution**,**optional*) - the
distribution. Defaults to `None`, which builds it from
`tensordict` with `get_dist()`.

Returns:

A tensor. For a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) while
[`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `False`, a
tensordict with one `<key>_log_prob` entry per sampled key (see
the log-probability key table). For a
composite distribution while it is `True`, the sum of these
entries, which are also written into `tensordict`.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import ProbabilisticTensorDictModule
>>> module = ProbabilisticTensorDictModule(
... in_keys=["loc", "scale"], out_keys=["action"], distribution_class=d.Normal
... )
>>> td = module(TensorDict(loc=torch.zeros(3), scale=torch.ones(3)))
>>> td["action"]
tensor([0., 0., 0.])
>>> module.log_prob(td)
tensor([-0.9189, -0.9189, -0.9189])
```

*property*log_prob_key

The key where the log-probability is written, when there is one such key.

See the log-probability key table.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate)
 changed since the module was built, or if it is `False` and
 the module has several sampled keys (use `log_prob_keys`).

*property*log_prob_keys

The list of keys where the log-probabilities are written.

One key per sampled key when [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate)
is `False`, and `[log_prob_key]` when it is `True`. See the
log-probability key table.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate)
 changed since the module was built.

*property*out_keys*: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

the sampled keys, then the log-probability keys if `return_log_prob=True`.

See the log-probability key table.

Type:

The keys that the module writes

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

## Tutorials that use `ProbabilisticTensorDictModule`

![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule