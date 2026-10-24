# set_composite_lp_aggregate

*class*tensordict.nn.set_composite_lp_aggregate(*mode: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***, *thread_local: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Bases: `_DecoratorContextManager`

Sets whether a [`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) sums its log-probabilities in one tensor.

When [`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) returns `True`, the
log-probabilities and entropies of a
[`CompositeDistribution`](tensordict.nn.distributions.CompositeDistribution.html#tensordict.nn.distributions.CompositeDistribution) are summed
into one tensor with the shape of the root tensordict.
When it returns `False` (the default), they are returned as a
TensorDict with one `<name>_log_prob` (or `<name>_entropy`) entry per
sampled key. The value also decides the keys where probabilistic modules
write their log-probabilities: see the [log-probability key table](tensordict.nn.ProbabilisticTensorDictModule.html#log-prob-keys) of
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule), and
[Modules: TensorDictModule and tensordict.nn](../../modules.html#modules-guide).

The `COMPOSITE_LP_AGGREGATE` environment variable sets the value too.
It works as a context manager and as a decorator, and `set()` and
`unset()` change it outside of a block.

Parameters:

**mode** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the value that
[`composite_lp_aggregate()`](tensordict.nn.composite_lp_aggregate.html#tensordict.nn.composite_lp_aggregate) returns inside the
block. Defaults to `True`.

Keyword Arguments:

**thread_local** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the mode applies to the
calling thread only, and blocks that other threads enter do not
change it. Inside the block, blocks without this flag that the
same thread enters stay in that thread too. Threads started inside
the block see the process-wide mode. If `False`, the mode applies
to the whole process. Defaults to `False`.

Note

Threads. The mode is process-wide by default, and is meant to
be set once for the whole process, for example with `set()`
before you build the modules: it decides the keys of the
log-probabilities, which the threads that write them and the threads
that read them must agree on. A `with` block in one thread also
changes the value in other threads at that time, unless it passes
`thread_local=True`. A thread started inside a `thread_local`
block reads the process-wide value, not the block's.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import CompositeDistribution, set_composite_lp_aggregate
>>> from torch import distributions as d
>>> _ = torch.manual_seed(0)
>>> params = TensorDict({
... "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
... ("nested", "disc"): {"logits": torch.randn(3, 10)}
... }, [3])
>>> dist = CompositeDistribution(params,
... distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical})
>>> sample = dist.sample((4,))
>>> with set_composite_lp_aggregate(False):
... lp = dist.log_prob(sample)
... print(lp)
TensorDict(batch_size=[4,3], fields={
 cont_log_prob: f32[4],
 nested: {disc_log_prob: f32[]}})
>>> with set_composite_lp_aggregate(True):
... lp = dist.log_prob(sample)
... print(lp)
tensor([[-2.0886, -1.2155, -0.0414],
 [-2.8973, -5.5165, 2.4402],
 [-0.2806, -1.2799, 3.1733],
 [-3.0407, -4.3593, 0.5763]])
```

set()

Sets the mode, as entering the block does, until `unset()` is called.

unset()

Restores the mode that `set()` replaced.