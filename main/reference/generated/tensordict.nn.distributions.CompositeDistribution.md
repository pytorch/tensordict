# CompositeDistribution

*class*tensordict.nn.distributions.CompositeDistribution(*params: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *distribution_map: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)*, ***, *name_map: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *extra_kwargs=None*, *log_prob_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *entropy_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`Distribution`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution), [`Mapping`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)

A composite distribution that groups multiple distributions together using the TensorDict interface.

This class allows for operations such as log_prob_composite, entropy_composite, cdf, icdf, rsample, and sample
to be performed on a collection of distributions, returning a TensorDict. `cdf()` and `icdf()`
write their results into the input TensorDict.

`log_prob()` and `entropy()` return a TensorDict with one entry
per distribution while [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is
`False` (the default), and one summed tensor while it is `True`. See
the [log-probability key table](tensordict.nn.ProbabilisticTensorDictModule.html#log-prob-keys) of
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) for the keys that
probabilistic modules write.

Parameters:

- **params** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - A nested key-tensor map where the root entries correspond to sample names, and the leaves
are the distribution parameters. Entry names must match those specified in distribution_map.
- **distribution_map** (*Dict**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**Type**[**torch.distributions.Distribution**]**]*) - Specifies the distribution types to be used.
The names of the distributions should match the sample names in the TensorDict.

Keyword Arguments:

- **name_map** (*Dict**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**,**optional*) - A mapping of where each sample should be written. If not provided,
the key names from distribution_map will be used.
- **extra_kwargs** (*Dict**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**Dict**]**,**optional*) - A dictionary of additional keyword arguments for constructing the distributions.
- **log_prob_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - 

The key where the aggregated log probability will be stored.
Defaults to 'sample_log_prob'.

Note

if [`tensordict.nn.composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) returns `False`, the log-probabilities will
be written under ("path", "to", "leaf", "<sample_name>_log_prob")
where ("path", "to", "leaf", "<sample_name>") is the `NestedKey` corresponding to
the leaf tensor being sampled. In that case, the `log_prob_key` argument will be ignored.
- **entropy_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - 

The key where the entropy will be stored. Defaults to 'entropy'

Note

if [`tensordict.nn.composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) returns `False`, the entropies will
be written under ("path", "to", "leaf", "<sample_name>_entropy")
where ("path", "to", "leaf", "<sample_name>") is the `NestedKey` corresponding to
the leaf tensor being sampled. In that case, the `entropy_key` argument will be ignored.

Note

The batch size of the input TensorDict containing the parameters (params) determines the batch shape of
the distribution. For example, the "sample_log_prob" entry resulting from a call to log_prob will have the
shape of the parameters plus any additional batch dimensions.

See also

[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) and [`ProbabilisticTensorDictSequential`](tensordict.nn.ProbabilisticTensorDictSequential.html#tensordict.nn.ProbabilisticTensorDictSequential)
to learn how to use this class as part of a model.

See also

[`set_composite_lp_aggregate`](tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate) to control the aggregation of the log-probabilities.

Examples

```
>>> import torch
>>> from torch import distributions as d
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution, set_composite_lp_aggregate
>>> _ = torch.manual_seed(0)
>>> params = TensorDict({
... "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
... ("nested", "disc"): {"logits": torch.randn(3, 10)}
... }, [3])
>>> dist = CompositeDistribution(params,
... distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
>>> sample = dist.sample((4,))
>>> with set_composite_lp_aggregate(False):
... sample = dist.log_prob(sample)
... print(sample)
TensorDict(
 fields={
 cont_log_prob: Tensor(shape=torch.Size([4, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 nested: TensorDict(
 fields={
 disc_log_prob: Tensor(shape=torch.Size([4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 3]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([4, 3]),
 device=None,
 is_shared=False)
```

cdf(*sample: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the cumulative distribution function (CDF) for each component distribution in the composite distribution.

This method calculates the CDF for each component distribution and updates the input TensorDict with the results.

Parameters:

**sample** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - A TensorDict containing samples for which to compute the CDF.

Returns:

The input TensorDict updated with <sample_name>_cdf entries for each component distribution.

Return type:

[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

entropy(*samples_mc: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*, ***, *aggregate_probabilities: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *include_sum: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes and returns the entropy of the composite distribution.

This method calculates the entropy for each component distribution and optionally sums them.

Parameters:

**samples_mc** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The number of samples to draw if the entropy does not have a closed-form solution.
Defaults to 1.

Keyword Arguments:

- **aggregate_probabilities** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - Deprecated and ignored. A value other than `None`
emits a [`DeprecationWarning`](https://docs.python.org/3/builtins/exceptions.html#DeprecationWarning). Use [`set_composite_lp_aggregate()`](tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate)
instead.
- **include_sum** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - Whether to include the summed entropy in the output TensorDict.
Defaults to composite_lp_aggregate(), which defaults to False.

Returns:

If [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate)
is `True`, a single tensor with the summed entropies. If it is
`False` (the default), a TensorDict with one `<name>_entropy`
entry per distribution (see `entropy_composite()`).

Return type:

[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) or [*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Note

If a distribution does not implement a closed-form solution for entropy, Monte Carlo sampling is used
to estimate it.

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

Computes the entropy for each component distribution and returns a TensorDict with individual entropies.

This method is used by the entropy method when [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is `False`.

Parameters:

**samples_mc** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The number of samples to draw if the entropy does not have a closed-form solution.
Defaults to 1.

Keyword Arguments:

**include_sum** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - Whether to include the summed entropy in the output TensorDict.
Defaults to composite_lp_aggregate(), which defaults to False.

Returns:

A TensorDict containing the individual entropies for each component distribution,
along with an "entropy" entry containing the summed entropies if include_sum is True.

Return type:

[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Note

If a distribution does not implement a closed-form solution for entropy, Monte Carlo sampling is used
to estimate it.

*classmethod*from_distributions(*params*, *distributions: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Distribution](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)]*, ***, *name_map: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *log_prob_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *entropy_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → CompositeDistribution

Create a CompositeDistribution instance from existing distribution objects.

This class method allows for the creation of a CompositeDistribution by directly providing
a dictionary of distribution instances, rather than specifying distribution types and parameters separately.

Parameters:

- **params** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - A TensorDict that defines the batch shape for the composite distribution.
Only its batch size is used.
- **distributions** (*Dict**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**d.Distribution**]*) - A dictionary mapping nested keys to distribution instances.
These distributions will be used directly in the composite distribution.

Keyword Arguments:

- **name_map** (*Dict**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**,**optional*) - A mapping of where each sample should be written. If not provided,
the keys of `distributions` are used.
- **log_prob_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - The key where the log probability will be stored.
Defaults to 'sample_log_prob'.
- **entropy_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - The key where the entropy will be stored. Defaults to 'entropy'.

Returns:

An instance of CompositeDistribution initialized with the provided distributions.

Return type:

*CompositeDistribution*

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - If a key in name_map cannot be found in the provided distributions.

Note

The batch size of the params TensorDict determines the batch shape of the composite distribution.

Example

```
>>> from tensordict.nn import CompositeDistribution, ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule, TensorDictModule
>>> import torch
>>> from tensordict import TensorDict
>>>
>>> # Values are not used to build the dists
>>> params = TensorDict({("0", "loc"): None, ("1", "loc"): None, ("0", "scale"): None, ("1", "scale"): None})
>>> d0 = torch.distributions.Normal(0, 1)
>>> d1 = torch.distributions.Normal(torch.zeros(1, 2), torch.ones(1, 2))
>>>
>>> d = CompositeDistribution.from_distributions(params, {"0": d0, "1": d1})
>>> print(d.sample())
TensorDict(
 fields={
 0: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 1: Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

icdf(*sample: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the inverse cumulative distribution function (inverse CDF) for each component distribution.

This method requires the input TensorDict to have either a <sample_name>_cdf entry or a <sample_name> entry
for each component distribution. It calculates the inverse CDF and updates the TensorDict with the results.

Parameters:

**sample** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - A TensorDict containing either <sample_name>_cdf or <sample_name> entries
for each component distribution.

Returns:

The input TensorDict updated with <sample_name>_icdf entries for each component distribution.

Return type:

[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - If neither <sample_name> nor <sample_name>_cdf can be found in the input TensorDict for a component distribution.

log_prob(*sample: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Computes the log-probability of a sample.

Parameters:

**sample** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input sample to compute the log probability for.

Returns:

- `False` (default): a new TensorDict with one `<name>_log_prob`
entry per distribution, for example `cont_log_prob` and
`("nested", "disc_log_prob")` (see `log_prob_composite()`).
`sample` is not modified.
- `True`: one tensor, the sum of the log-probabilities over all
distributions and their event dimensions.

In both cases the batch shape is the batch shape of `sample`.

Return type:

The result depends on [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate)

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

Computes the log-probability of each component in the input sample and return a TensorDict with individual log-probabilities.

Parameters:

**sample** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The input sample to compute the log probabilities for.

Keyword Arguments:

**include_sum** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - Whether to include the summed log-probability in the output TensorDict.
Defaults to `composite_lp_aggregate()` (`False` by default).

Returns:

A TensorDict containing the individual log-probabilities for each component in the input sample,

along with a "sample_log_prob" entry containing the summed log-probability if include_sum is True.
While [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) is
`True`, the entries are written into `sample`, which is
returned. Otherwise the result is a new TensorDict.

Return type:

[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

*property*mean*: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*

Returns the mean of the distribution.

*property*mode*: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*

Returns the mode of the distribution.

rsample(*shape=None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Generates a sample_shape shaped reparameterized sample or sample_shape
shaped batch of reparameterized samples if the distribution parameters
are batched.

sample(*shape=None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.