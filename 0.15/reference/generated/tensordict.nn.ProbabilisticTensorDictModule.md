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
log-probability of a value, call `log_prob()`.

The value depends on the interaction type. The module uses
[`interaction_type()`](tensordict.nn.probabilistic.interaction_type.html#tensordict.nn.probabilistic.interaction_type) if it is set (see
[`set_interaction_type`](tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type)). Otherwise it uses
`default_interaction_type`. See [`InteractionType`](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType)
for what each type reads from the distribution.

With `return_log_prob=True` and the `RANDOM` interaction type, the
module draws and scores the value in one call if the distribution class
defines the method: `dist.rsample_and_log_prob(sample_shape)` if
`dist.has_rsample`, `dist.sample_and_log_prob(sample_shape)`
otherwise. The method returns a `(sample, log_prob)` tuple, with
`log_prob` in the layout of `dist.log_prob`. Define it when
`dist.log_prob(sample)` cannot recover the draw exactly, for example
for a tanh-squashed normal that saturates. Otherwise the module calls
`dist.log_prob` on the sample. The [`torch.distributions`](https://docs.pytorch.org/docs/stable/distributions.html#module-torch.distributions) classes
do not define these methods;
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) defines both,
and calls them on its components when they define them.

The default `distribution_class` is
[`Delta`](tensordict.nn.distributions.Delta.html#tensordict.nn.distributions.Delta), which always returns its
parameter. With it, the module copies its input to `out_keys`.

If every key in `out_keys` is already in the input and
[`skip_existing()`](tensordict.nn.skip_existing.html#tensordict.nn.skip_existing) is `True` (see
[`set_skip_existing`](tensordict.nn.set_skip_existing.html#tensordict.nn.set_skip_existing)), the module returns the input
unchanged. Otherwise it computes a new value and overwrites the entry.
See [Modules: TensorDictModule and tensordict.nn](../../modules.html#modules-guide) for a guide.

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

- **in_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*|**List**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**|**Dict**[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]*) - key(s) that will be read from the input TensorDict
and used to build the distribution.
If it is a list of NestedKey or a NestedKey, the leaf (last element) of those keys must match the keywords used by
the distribution class of interest, e.g. `"loc"` and `"scale"` for
the `Normal` distribution and similar.
If in_keys is a dictionary, the keys are the keys of the distribution and the values are the keys in the
tensordict that will get match to the corresponding distribution keys.
- **out_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*|**List**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**|**None*) - key(s) where the sampled values will be written.
If the input already has these keys, the module overwrites them,
unless [`skip_existing()`](tensordict.nn.skip_existing.html#tensordict.nn.skip_existing) is `True`.

Keyword Arguments:

- **default_interaction_type** ([*InteractionType*](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType)*,**optional*) - 

keyword-only argument.
Default method to be used to retrieve
the output value. Should be one of InteractionType: DETERMINISTIC, MODE, MEDIAN, MEAN or RANDOM
(in which case the value is sampled randomly from the distribution). Default
is DETERMINISTIC.

Note

When a value is computed, the module first reads
[`interaction_type()`](tensordict.nn.probabilistic.interaction_type.html#tensordict.nn.probabilistic.interaction_type). If it
returns `None` (outside of [`set_interaction_type`](tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type)),
the module uses `default_interaction_type`. TorchRL
collectors sample with `InteractionType.RANDOM` by default
(their `exploration_type` argument).

Note

`MODE` and `MEDIAN` read `dist.mode` and `dist.median`,
and raise `NotImplementedError` if the distribution does
not have them. `MEAN` reads `dist.mean`. If the
distribution does not have a mean, the module averages
`n_empirical_estimate` samples.
- **distribution_class** (*Type**or**Callable**[**[**Any**]**,**Distribution**]**,**optional*) - 

keyword-only argument.
A `torch.distributions.Distribution` class to
be used for sampling.
Default is [`Delta`](tensordict.nn.distributions.Delta.html#tensordict.nn.distributions.Delta).

Note

If the distribution class is of type
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution), the `out_keys`
can be inferred directly from the `"distribution_map"` or `"name_map"`
keyword arguments provided through this class' `distribution_kwargs`
keyword argument, making the `out_keys` optional in such cases.
- **distribution_kwargs** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - 

keyword-only argument.
Keyword-argument pairs to be passed to the distribution.

Note

if your kwargs contain tensors that you would like to transfer to device with the module, or
tensors that should see their dtype modified when calling module.to(dtype), you can wrap the kwargs
in a [`TensorDictParams`](tensordict.TensorDictParams.html#tensordict.TensorDictParams) to do this automatically.
- **return_log_prob** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - keyword-only argument.
If `True`, the log-probability of the
distribution sample will be written in the tensordict, at the key
given in the table above. Default is `False`.
- **log_prob_keys** (*List**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**,**optional*) - 

keys where to write the log_prob if `return_log_prob=True`.
Defaults to '<sample_key_name>_log_prob', where <sample_key_name> is each of the `out_keys`.

Note

This is only available when [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is set to `False`.
- **log_prob_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - 

key where to write the log_prob if `return_log_prob=True`.
Defaults to 'sample_log_prob' when [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is set to True
or '<sample_key_name>_log_prob' otherwise.

Note

When there is more than one sampled key, this is only available when [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is set to `True`.
- **cache_dist** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - 

keyword-only argument. Has no effect.
Default is `False`.

Deprecated since version 0.15: Passing `cache_dist=True` warns, and the argument will be
removed in TensorDict 0.17. Do not pass it.
- **n_empirical_estimate** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - keyword-only argument.
Number of samples to compute the empirical
mean when it is not available. Defaults to 1000.
- **num_samples** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*or*[*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - keyword-only argument.
The number of values per call. The output tensordict is expanded
to `num_samples + batch_size`, and each sampled value gets these
leading dimensions. With the `RANDOM` interaction type, each
value is a new draw. With another interaction type, the module
repeats its one deterministic value `num_samples` times.
Defaults to `None`: one value, no extra dimension.
- **generator** ([*Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,*[*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*, or*[*None*](https://docs.python.org/3/builtins/constants.html#None)*,**optional*) - 

keyword-only argument.
Controls the random number generator used when sampling from the distribution
(relevant only for `RANDOM` and `MEAN` interaction types).

- `None` (default): the global PyTorch RNG is used (current behaviour).
- [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator): this generator is used for every sample. Its state
advances in place across calls, so consecutive forward passes draw distinct
samples. The generator must live on the same device as the distribution
parameters.
- [`int`](https://docs.python.org/3/builtins/functions.html#int): shorthand for `torch.Generator().manual_seed(int)`; a CPU
generator is created at construction time and used statefully thereafter.
- `NestedKey` (`str` or `tuple`): the generator is fetched from the
input tensordict at this key on every call. The value may be a
[`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) (used in place, state mutates) or an integer scalar
(read as a stream-key, then a fresh `next_seed` is written back to the
same key after sampling, JAX-PRNG-style).

Note

PyTorch's `Distribution` API does not
accept a `generator` argument. Internally, sampling is wrapped in
[`torch.random.fork_rng()`](https://docs.pytorch.org/docs/stable/random.html#torch.random.fork_rng) and the global RNG state is swapped with the
provided generator's state for the duration of the call. This makes the
approach portable across all distributions (including reparameterized
`rsample` and discrete `sample`). Under `torch.compile`,
the argument is ignored, without a warning.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
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
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 action_log_prob: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 hidden: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 input: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 params: Tensor(shape=torch.Size([3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
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
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 action_log_prob: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 hidden: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 input: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 loc: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 params: Tensor(shape=torch.Size([4, 3, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 scale: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 3]),
 device=None,
 is_shared=False)
```

build_dist_from_params(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Creates a `torch.distributions.Distribution` instance with the parameters provided in the input tensordict.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input tensordict containing the distribution parameters.

Returns:

A `torch.distributions.Distribution` instance created from the input tensordict.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - If the input tensordict does not match the distribution keywords.

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

Returns all the keys pointing at the distribution params.

*property*dist_sample_keys*: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*

Returns all the keys pointing at the distribution samples.

forward(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *tensordict_out: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_requires_sample: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

get_dist(*tensordict: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Creates a `torch.distributions.Distribution` instance with the parameters provided in the input tensordict.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input tensordict containing the distribution parameters.

Returns:

A `torch.distributions.Distribution` instance created from the input tensordict.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - If the input tensordict does not match the distribution keywords.

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

Checks if a module is compatible with TensorDictModule API.

Parameters:

**module** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) - the object to check.

Returns:

`True` if `module` has both an `in_keys` and an `out_keys`
attribute.

log_prob(*tensordict*, ***, *dist: [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Computes the log-probability of the value stored in `tensordict`.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input tensordict containing the
distribution parameters and the value at `out_keys`.

Keyword Arguments:

**dist** (*torch.distributions.Distribution**,**optional*) - The distribution instance. Defaults to `None`.
If `None`, the distribution will be computed using the get_dist method.

Returns:

A tensor. For a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) while
[`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `False`, a
TensorDict with one `<key>_log_prob` entry per sampled key (see
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

## Tutorials that use `ProbabilisticTensorDictModule`

![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule