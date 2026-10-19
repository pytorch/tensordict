# Delta

*class*tensordict.nn.distributions.Delta(*param: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *atol: [float](https://docs.python.org/3/builtins/functions.html#float) = 1e-06*, *rtol: [float](https://docs.python.org/3/builtins/functions.html#float) = 1e-06*, *batch_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *event_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`Distribution`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

A distribution with all its mass on one value, `param`.

`sample()` and `rsample()` return `param` (expanded to the
sample shape), and `mean`, `mode` and
`deterministic_sample` return it as it is. `log_prob()` returns
`inf` for a value equal to `param` (within `atol` and `rtol`, on
every element of the event) and `-inf` otherwise. It is the default
`distribution_class` of
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule), which then copies
its input to its output.

Parameters:

- **param** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the value that holds all the mass.
- **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the absolute tolerance with which
`log_prob()` compares a value with `param`. Defaults to
`1e-6`.
- **rtol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the relative tolerance with which
`log_prob()` compares a value with `param`. Defaults to
`1e-6`.
- **batch_shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch shape. Defaults to
`None`: see `event_shape`.
- **event_shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the shape of one value. If
neither `batch_shape` nor `event_shape` is given (or both
are empty), the last dimension of `param` is the event shape and
the others are the batch shape. Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict.nn.distributions import Delta
>>> dist = Delta(torch.zeros(4, 2))
>>> dist.batch_shape, dist.event_shape
(torch.Size([4]), torch.Size([2]))
>>> dist = Delta(torch.tensor([1.0, 2.0]))
>>> dist.sample()
tensor([1., 2.])
>>> dist.log_prob(torch.tensor([1.0, 2.0])), dist.log_prob(torch.tensor([1.0, 3.0]))
(tensor(inf), tensor(-inf))
```

log_prob(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the log of the probability density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

*property*mean*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mean of the distribution.

*property*mode*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mode of the distribution.

rsample(*sample_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Generates a sample_shape shaped reparameterized sample or sample_shape
shaped batch of reparameterized samples if the distribution parameters
are batched.

sample(*sample_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.

update(*param: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Replaces `param`. The batch and event shapes do not change.