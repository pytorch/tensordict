# OneHotCategorical

*class*tensordict.nn.distributions.OneHotCategorical(*logits: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *probs: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*)

Bases: [`Categorical`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.categorical.Categorical)

One-hot categorical distribution.

This class behaves exactly as torch.distributions.Categorical except that it reads and produces one-hot encodings
of the discrete tensors. `sample()`, `mode` and
`deterministic_sample` return `torch.long` one-hot tensors, and
`log_prob()` takes one.

`rsample()` returns a straight-through sample: its values are one-hot,
and its gradient is the gradient of a
`RelaxedOneHotCategorical` sample with
temperature 1.

Parameters:

- **logits** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - the unnormalized log-probabilities,
with the categories along the last dimension. A last dimension of
size 1 is removed first.
- **probs** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - the probabilities, as for `logits`.
Pass `logits` or `probs`, not both.
- ****kwargs** - passed to `Categorical`, for
example `validate_args`.

Examples

```
>>> import torch
>>> from tensordict.nn.distributions import OneHotCategorical
>>> _ = torch.manual_seed(0)
>>> dist = OneHotCategorical(logits=torch.tensor([0.0, 2.0, 1.0]))
>>> dist.mode
tensor([0, 1, 0])
>>> dist.sample((2,))
tensor([[0, 0, 1],
 [0, 1, 0]])
>>> dist.log_prob(torch.tensor([0, 1, 0]))
tensor(-0.4076)
```

*property*deterministic_sample*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mode of the distribution.

log_prob(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the log of the probability density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

*property*mode*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mode of the distribution.

rsample(*sample_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Generates a sample_shape shaped reparameterized sample or sample_shape
shaped batch of reparameterized samples if the distribution parameters
are batched.

sample(*sample_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Generates a sample_shape shaped sample or sample_shape shaped batch of
samples if the distribution parameters are batched.