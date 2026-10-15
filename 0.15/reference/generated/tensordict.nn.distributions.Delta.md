# Delta

*class*tensordict.nn.distributions.Delta(*param: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *atol: [float](https://docs.python.org/3/builtins/functions.html#float) = 1e-06*, *rtol: [float](https://docs.python.org/3/builtins/functions.html#float) = 1e-06*, *batch_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *event_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`Distribution`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Delta distribution.

A distribution with all its mass on `param`. `sample()`,
`rsample()`, `mean`, `mode` and
`deterministic_sample` return `param`. `log_prob()` returns
`inf` for a value equal to `param` (within `atol` and `rtol`) and
`-inf` otherwise. It is the default `distribution_class` of
[`ProbabilisticTensorDictModule`](tensordict.nn.ProbabilisticTensorDictModule.html#tensordict.nn.ProbabilisticTensorDictModule).

Parameters:

- **param** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - parameter of the delta distribution;
- **atol** (*number**,**optional*) - absolute tolerance to consider that a tensor matches the distribution parameter;
Default is 1e-6
- **rtol** (*number**,**optional*) - relative tolerance to consider that a tensor matches the distribution parameter;
Default is 1e-6
- **batch_shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - batch shape;
- **event_shape** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - shape of the outcome.
If neither `batch_shape` nor `event_shape` is given, the last
dimension of `param` is the event shape and the others are the
batch shape.

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