# CompositeDistribution

*class*tensordict.nn.distributions.CompositeDistribution(*params: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *distribution_map: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)*, ***, *name_map: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *extra_kwargs=None*, *log_prob_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *entropy_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`Distribution`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A distribution made of several named distributions, which samples and scores tensordicts.

Each component is built from one entry of `params`, as
`distribution_map[name](**params[name], **extra_kwargs[name])`.
`sample()`, `rsample()`, `mode`, `mean` and
`deterministic_sample` return a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with
one entry per component, at its name or at its `name_map` key.
`log_prob()`, `cdf()` and `icdf()` take such a tensordict.
The batch size of `params` is the `batch_shape` of the distribution,
and a sample of shape `sample_shape` has the batch size
`sample_shape + batch_shape`. The `event_shape` is `torch.Size([])`:
the event dimensions of each component stay in its entry of the sample.

The distribution is also a read-only mapping from the sample keys to the
components: `dist["cont"]`, `list(dist)`, `len(dist)`.

`log_prob()` and `entropy()` return a tensordict with one entry
per component while [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is
`False` (the default), and one summed tensor while it is `True`. See
the [log-probability key table](tensordict.nn.ProbabilisticTensorDictModule.html#log-prob-keys) of
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) for the keys that
probabilistic modules write.

Parameters:

- **params** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the parameters of the components. The entry
at each name of `distribution_map` is a tensordict of keyword
arguments for that distribution, for example
`{"loc": ..., "scale": ...}` for a `Normal`.
- **distribution_map** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - maps each name (a `NestedKey`) to a
distribution class, or to any callable that takes the parameters
as keyword arguments and returns a distribution.

Keyword Arguments:

- **name_map** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - maps a name of `distribution_map` to the
key where its sample is written. Names that are not in it keep
their own key. Keys are compared after
[`unravel_key()`](tensordict.unravel_key.html#tensordict.unravel_key). Defaults to `None`.
- **extra_kwargs** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - maps a name of `distribution_map` to a
dict of extra keyword arguments for that distribution, for
example constants that are not in `params`. Defaults to `None`.
- **log_prob_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the key of the summed
log-probability that `log_prob_composite()` writes with
`include_sum=True`. The per-component entries are always at
`<sample key>_log_prob`. Defaults to `"sample_log_prob"`.
- **entropy_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the key of the summed entropy that
`entropy_composite()` writes with `include_sum=True`. The
per-component entries are always at `<sample key>_entropy`.
Defaults to `"entropy"`.

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if `params` has no entry for a name of `distribution_map`.

See also

[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) and
[`ProbabilisticTensorDictSequential`](tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential) to use this
class in a model, and [`set_composite_lp_aggregate`](tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate)
to control the aggregation of the log-probabilities.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution
>>> params = TensorDict({
... "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
... ("nested", "disc"): {"logits": torch.zeros(3, 10)},
... }, [3])
>>> dist = CompositeDistribution(params,
... distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
>>> dist["cont"]
Normal(loc: torch.Size([3, 4]), scale: torch.Size([3, 4]))
>>> sample = dist.sample((4,))
>>> sample.batch_size, sample["cont"].shape, sample["nested", "disc"].shape
(torch.Size([4, 3]), torch.Size([4, 3, 4]), torch.Size([4, 3]))
>>> lp = dist.log_prob(sample)
>>> lp["cont_log_prob"].shape, lp["nested", "disc_log_prob"].shape
(torch.Size([4, 3, 4]), torch.Size([4, 3]))
```

cdf(*sample: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the cumulative distribution function (CDF) of each component and writes it into `sample`.

The CDF of the component at `key` is written at `<key>_cdf`, with
the suffix on the last element of a nested key.

Parameters:

**sample** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the sample, with the value of each
component at its sample key.

Returns:

`sample`, with the `<key>_cdf` entries.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution
>>> params = TensorDict({
... "a": {"loc": torch.zeros(2), "scale": torch.ones(2)},
... "b": {"loc": torch.ones(2), "scale": torch.ones(2)},
... }, [2])
>>> dist = CompositeDistribution(params, distribution_map={"a": d.Normal, "b": d.Normal})
>>> sample = TensorDict(a=torch.zeros(2), b=torch.zeros(2), batch_size=[2])
>>> out = dist.cdf(sample)
>>> out is sample
True
>>> sample["a_cdf"], sample["b_cdf"]
(tensor([0.5000, 0.5000]), tensor([0.1587, 0.1587]))
>>> _ = dist.icdf(sample) # reads the "_cdf" entries
>>> sample["a_icdf"]
tensor([0., 0.])
```

*property*deterministic_sample*: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*

A tensordict with a deterministic value of each component, of batch size `batch_shape`.

This is the value that a probabilistic module writes with the
`DETERMINISTIC` interaction type (see
[`InteractionType`](tensordict.nn.InteractionType.html#tensordict.nn.InteractionType)). Each entry is the
`deterministic_sample` attribute of the component. A component
without one gives its mode if its support is enumerable
(`Categorical`,
`Bernoulli`, ...) and its mean otherwise.
For a distribution class that tensordict does not know, the choice
follows the support of the distribution, with a warning.

Raises:

[**NotImplementedError**](https://docs.python.org/3/builtins/exceptions.html#NotImplementedError) - if a component has no
 `deterministic_sample` and the mean or mode it falls back on
 is not implemented.

entropy(*samples_mc: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*, ***, *aggregate_probabilities: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *include_sum: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the entropy of each component, or their sum.

A component whose `entropy()` raises `NotImplementedError` gets a
Monte Carlo estimate from `samples_mc` reparameterized samples
(`rsample`).

Parameters:

**samples_mc** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of samples of a Monte Carlo
estimate. Defaults to `1`.

Keyword Arguments:

- **aggregate_probabilities** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - deprecated and ignored,
will be removed in TensorDict 0.17. A value other than
`None` emits a `DeprecationWarning`. Use
[`set_composite_lp_aggregate()`](tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate) instead.
- **include_sum** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - used while
[`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `False`: if
`True`, the tensordict also holds the summed entropy at
`entropy_key`. Defaults to `None`, which uses the value
of [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate).

Returns:

While [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `False`
(the default), a tensordict with one `<name>_entropy` entry per
component (see `entropy_composite()`). While it is `True`, one
tensor, the sum of the entropies over the components and their
event dimensions.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution, set_composite_lp_aggregate
>>> params = TensorDict({
... "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
... ("nested", "disc"): {"logits": torch.zeros(3, 10)}
... }, [3])
>>> dist = CompositeDistribution(params,
... distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
>>> sorted(dist.entropy().keys(True, True), key=str)
[('nested', 'disc_entropy'), 'cont_entropy']
>>> with set_composite_lp_aggregate(True):
... dist.entropy()
tensor([7.9783, 7.9783, 7.9783])
```

entropy_composite(*samples_mc=1*, ***, *include_sum: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the entropy of each component and returns them in a new tensordict.

The entropy of the component at `key` is at `<key>_entropy`, with
the suffix on the last element of a nested key. `entropy()` calls
this method while [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is
`False`. A component without a closed-form entropy gets a Monte Carlo
estimate, as in `entropy()`.

Parameters:

**samples_mc** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of samples of a Monte Carlo
estimate. Defaults to `1`.

Keyword Arguments:

**include_sum** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the result also holds
the sum of the entropies over the components and their event
dimensions, at `entropy_key`. Defaults to `None`, which
uses the value of [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate).

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) of batch size `batch_shape`.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution
>>> params = TensorDict({
... "a": {"loc": torch.zeros(2), "scale": torch.ones(2)},
... "b": {"probs": torch.full((2,), 0.5)},
... }, [2])
>>> dist = CompositeDistribution(params, distribution_map={"a": d.Normal, "b": d.Bernoulli})
>>> ent = dist.entropy_composite(include_sum=True)
>>> sorted(ent.keys())
['a_entropy', 'b_entropy', 'entropy']
>>> ent["entropy"]
tensor([2.1121, 2.1121])
```

*property*entropy_key

The key of the summed entropy that `entropy_composite()` writes with `include_sum=True`.

Defaults to `"entropy"`. Setting it to `None` restores the default.

*classmethod*from_distributions(*params*, *distributions: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)]*, ***, *name_map: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *log_prob_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *entropy_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → CompositeDistribution

Builds a composite distribution from distribution instances, instead of classes and parameters.

The distributions are used as they are. [`ProbabilisticTensorDictSequential`](tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential)
uses this method to combine the distributions of its modules.

Parameters:

- **params** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - a tensordict whose batch size is the
`batch_shape` of the composite distribution. Its entries are
not read.
- **distributions** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - maps each name (a `NestedKey`) to a
`Distribution` instance. Each
batch shape must start with the batch size of `params`.

Keyword Arguments:

- **name_map** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,**optional*) - maps a name of `distributions` to the
key where its sample is written. Names that are not in it keep
their own key, and keys of `name_map` that are not in
`distributions` are ignored. Keys are compared after
[`unravel_key()`](tensordict.unravel_key.html#tensordict.unravel_key). Defaults to `None`.
- **log_prob_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - as in
`CompositeDistribution`. Defaults to `"sample_log_prob"`.
- **entropy_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - as in
`CompositeDistribution`. Defaults to `"entropy"`.

Returns:

A new `CompositeDistribution`.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution
>>> dist = CompositeDistribution.from_distributions(
... TensorDict(batch_size=[3]),
... {"a": d.Normal(torch.zeros(3), torch.ones(3)),
... "b": d.Bernoulli(torch.full((3,), 0.5))},
... )
>>> sample = dist.sample()
>>> sample.batch_size, sorted(sample.keys())
(torch.Size([3]), ['a', 'b'])
```

icdf(*sample: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the inverse CDF of each component and writes it into `sample`.

For the component at `key`, the method reads the probability at
`<key>_cdf` (as written by `cdf()`) and writes its inverse CDF at
`<key>_icdf`. If `<key>_cdf` is missing, it uses the CDF of the
value at `key`, so `<key>_icdf` is that value again, up to rounding.
See the example of `cdf()`.

Parameters:

**sample** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - a tensordict with a `<key>_cdf` or a
`key` entry for each component.

Returns:

`sample`, with the `<key>_icdf` entries.

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if neither `key` nor `<key>_cdf` is in `sample`
 for a component.

log_prob(*sample: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the log-probability of a sample.

Parameters:

**sample** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the sample, with the value of each
component at its sample key, as returned by `sample()`.

Returns:

The result depends on [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate).

- `False` (default): a new tensordict with one `<name>_log_prob`
entry per distribution, for example `cont_log_prob` and
`("nested", "disc_log_prob")` (see `log_prob_composite()`).
`sample` is not modified.
- `True`: one tensor, the sum of the log-probabilities over all
distributions and their event dimensions.

In both cases the batch shape is the batch shape of `sample`.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution, set_composite_lp_aggregate
>>> params = TensorDict({
... "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
... ("nested", "disc"): {"logits": torch.zeros(3, 10)}
... }, [3])
>>> dist = CompositeDistribution(params,
... distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
>>> sample = TensorDict({"cont": torch.zeros(3, 4), ("nested", "disc"): torch.zeros(3, dtype=torch.long)}, [3])
>>> lp = dist.log_prob(sample)
>>> sorted(lp.keys(True, True), key=str)
[('nested', 'disc_log_prob'), 'cont_log_prob']
>>> with set_composite_lp_aggregate(True):
... dist.log_prob(sample)
tensor([-5.9783, -5.9783, -5.9783])
```

log_prob_composite(*sample: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *include_sum: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the log-probability of each component and returns them in a tensordict.

The log-probability of the component at `key` is at
`<key>_log_prob`, with the suffix on the last element of a nested
key: `("nested", "disc_log_prob")`. `log_prob()` calls this
method while [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is
`False`.

Parameters:

**sample** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the sample, with the value of each
component at its sample key.

Keyword Arguments:

**include_sum** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the result also holds
the sum of the log-probabilities over the components and their
event dimensions, at `log_prob_key`. Defaults to
`None`, which uses the value of
[`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate).

Returns:

While [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `False`, a
new tensordict with the log-probabilities. While it is `True`,
`sample`, with the log-probabilities written into it.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution
>>> params = TensorDict({
... "a": {"loc": torch.zeros(2), "scale": torch.ones(2)},
... "b": {"probs": torch.full((2,), 0.5)},
... }, [2])
>>> dist = CompositeDistribution(params, distribution_map={"a": d.Normal, "b": d.Bernoulli})
>>> sample = TensorDict(a=torch.zeros(2), b=torch.ones(2), batch_size=[2])
>>> lp = dist.log_prob_composite(sample, include_sum=True)
>>> sorted(lp.keys())
['a_log_prob', 'b_log_prob', 'sample_log_prob']
>>> lp["sample_log_prob"]
tensor([-1.6121, -1.6121])
```

*property*log_prob_key

The key of the summed log-probability that `log_prob_composite()` writes with `include_sum=True`.

Defaults to `"sample_log_prob"`. Setting it to `None` restores the
default.

*property*mean*: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*

A tensordict with the mean of each component, of batch size `batch_shape`.

*property*mode*: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*

A tensordict with the mode of each component, of batch size `batch_shape`.

rsample(*shape=None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Draws a reparameterized sample from each component and returns them in a tensordict.

Every component must support `rsample`.

Parameters:

**shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the leading sample dimensions.
Defaults to `torch.Size([])`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with one entry per component
and the batch size `shape + batch_shape`.

rsample_and_log_prob(*shape=None*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)]

Draws a reparameterized sample and returns it with the log-probability of the same draw.

Each component is sampled and scored with its own
`rsample_and_log_prob` method if it has one, and with `rsample` then
`log_prob` otherwise. See `sample_and_log_prob()`.

Parameters:

**shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the leading sample dimensions.
Defaults to `torch.Size([])`.

Returns:

A `(sample, log_prob)` tuple, with the sample as returned by
`rsample()` and its log-probability in the layout of
`log_prob()`.

sample(*shape=None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Draws a sample from each component and returns them in a tensordict.

Parameters:

**shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the leading sample dimensions.
Defaults to `torch.Size([])`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with one entry per component
and the batch size `shape + batch_shape`.

sample_and_log_prob(*shape=None*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)]

Draws a sample and returns it with the log-probability of the same draw.

Each component is sampled and scored with its own `sample_and_log_prob`
method if it has one, and with `sample` then `log_prob` otherwise.
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) calls this method
when it samples with `return_log_prob=True`.

Parameters:

**shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the leading sample dimensions.
Defaults to `torch.Size([])`.

Returns:

A `(sample, log_prob)` tuple, with the sample as returned by
`sample()` and its log-probability in the layout of
`log_prob()`.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution
>>> params = TensorDict({
... "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
... ("nested", "disc"): {"logits": torch.zeros(3, 10)}
... }, [3])
>>> dist = CompositeDistribution(params,
... distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
>>> sample, lp = dist.sample_and_log_prob()
>>> sorted(lp.keys(True, True), key=str)
[('nested', 'disc_log_prob'), 'cont_log_prob']
```