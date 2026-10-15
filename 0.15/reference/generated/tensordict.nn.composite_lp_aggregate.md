# composite_lp_aggregate

tensordict.nn.composite_lp_aggregate(*nowarn: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None)

Returns whether a [`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) log-probabilities and entropies will be aggregated in a single tensor.

Returns `False` unless [`set_composite_lp_aggregate`](tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate)
or the `COMPOSITE_LP_AGGREGATE` environment variable set it to `True`.
See the [log-probability key table](tensordict.nn.ProbabilisticTensorDictModule.html#log-prob-keys) of
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule) for what each value
changes.

Parameters:

**nowarn** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - has no effect. Defaults to False.

See also

[`set_composite_lp_aggregate()`](tensordict.nn.set_composite_lp_aggregate.html#tensordict.nn.set_composite_lp_aggregate)