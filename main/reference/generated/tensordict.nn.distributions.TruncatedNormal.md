# TruncatedNormal

*class*tensordict.nn.distributions.TruncatedNormal(*loc: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *scale: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *a: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *b: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *validate_args: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`TruncatedStandardNormal`](tensordict.nn.distributions.truncated_normal.TruncatedStandardNormal.html#tensordict.nn.distributions.truncated_normal.TruncatedStandardNormal)

A normal distribution of mean `loc` and standard deviation `scale`, restricted to `[a, b]`.

Its samples are reparameterized (`rsample`).

Source: [https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf](https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf)

Parameters:

- **loc** (*Number**or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the mean of the normal distribution
before truncation.
- **scale** (*Number**or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the standard deviation of the normal
distribution before truncation. Values under `1e-6` are set to
`1e-6`.
- **a** (*Number**or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the lower bound of the support, in the
same units as `loc`.
- **b** (*Number**or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the upper bound of the support, in the
same units as `loc`.
- **validate_args** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - passed to
`Distribution`. Defaults to `None`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `a` is not smaller than `b` everywhere.

Examples

```
>>> import torch
>>> from tensordict.nn.distributions import TruncatedNormal
>>> _ = torch.manual_seed(0)
>>> dist = TruncatedNormal(loc=torch.zeros(3), scale=torch.ones(3), a=-1.0, b=1.0)
>>> dist.sample()
tensor([-0.0064, 0.4764, -0.7754])
>>> dist.mean
tensor([0., 0., 0.])
>>> TruncatedNormal(loc=5.0, scale=1.0, a=4.0, b=6.0).mean
tensor(5.)
```

cdf(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the cumulative density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

icdf(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the inverse cumulative density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

log_prob(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the log of the probability density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

*property*support*: [Constraint](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.constraints.Constraint)*

Returns a [`Constraint`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.constraints.Constraint) object
representing this distribution's support.