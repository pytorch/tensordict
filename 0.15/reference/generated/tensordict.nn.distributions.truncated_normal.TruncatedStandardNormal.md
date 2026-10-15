# TruncatedStandardNormal

*class*tensordict.nn.distributions.truncated_normal.TruncatedStandardNormal(*a: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *b: [Number](https://docs.python.org/3/library/numbers.html#numbers.Number) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *validate_args: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`Distribution`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.distribution.Distribution)

Truncated Standard Normal distribution.

Source: [https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf](https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf)

cdf(*value: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the cumulative density/mass function evaluated at
value.

Parameters:

**value** (*Tensor*) -

entropy() → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns entropy of distribution, batched over batch_shape.

Returns:

Tensor of shape batch_shape.

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

*property*mean*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the mean of the distribution.

rsample(*sample_shape: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Generates a sample_shape shaped reparameterized sample or sample_shape
shaped batch of reparameterized samples if the distribution parameters
are batched.

*property*support*: constraints.Constraints*

Returns a [`Constraint`](https://docs.pytorch.org/docs/stable/distributions.html#torch.distributions.constraints.Constraint) object
representing this distribution's support.

*property*variance*: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*

Returns the variance of the distribution.