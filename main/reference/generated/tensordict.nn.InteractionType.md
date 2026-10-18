# InteractionType

*class*tensordict.nn.InteractionType(*value*, *names=<not given>*, **values*, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)

Bases: [`StrEnum`](https://docs.python.org/3/library/enum.html#enum.StrEnum)

The rule that a probabilistic module uses to get a value from a distribution.

- `RANDOM`: a sample, from `rsample()` if the distribution has it,
else from `sample()`.
- `MODE`, `MEDIAN`, `MEAN`: the attribute of the same name. If the
distribution has no `mean`, `MEAN` averages `n_empirical_estimate`
samples (see [`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule)).
- `DETERMINISTIC`: the `deterministic_sample` attribute. If the
distribution does not have one, the mode for distributions with
enumerable support (Categorical, Bernoulli,
[`OneHotCategorical`](tensordict.nn.distributions.OneHotCategorical.html#tensordict.nn.distributions.OneHotCategorical)) and the mean
for the others. For a distribution class that tensordict does not
know, the module chooses the mean or the mode from the support of the
distribution, and warns.

Set the type for a block of code with [`set_interaction_type`](tensordict.nn.set_interaction_type.html#tensordict.nn.set_interaction_type), or
for one module with its `default_interaction_type` argument. Strings
are accepted, in any case: `InteractionType("RANDOM")`.

Examples

```
>>> from tensordict.nn import InteractionType
>>> InteractionType("RANDOM")
<InteractionType.RANDOM: 'random'>
>>> InteractionType.from_str("Mode") is InteractionType.MODE
True
```

*classmethod*from_str(*type_str: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*) → InteractionType

Returns the interaction type whose name matches a string, in any case.

`InteractionType(type_str)` does the same.

Parameters:

**type_str** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - the name, for example `"random"` or `"MODE"`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if no interaction type has this name.

## Tutorials that use `InteractionType`

![](../../_images/sphx_glr_tensordict_module_thumb.png)

[Building models with TensorDictModule](../../tutorials/tensordict_module.html)

Building models with TensorDictModule