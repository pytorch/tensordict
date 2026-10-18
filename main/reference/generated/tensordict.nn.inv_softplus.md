# inv_softplus

tensordict.nn.inv_softplus(*bias: [float](https://docs.python.org/3/builtins/functions.html#float) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [float](https://docs.python.org/3/builtins/functions.html#float) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the input of the softplus function that gives `bias`: `log(exp(bias) - 1)`.

`exp(bias) - 1` is clamped to `1e-6` or more, so that the result is
finite for `bias <= 0`.

Parameters:

**bias** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the output of the softplus function.

Returns:

A float for a float input, else a tensor.

Examples

```
>>> import torch
>>> from tensordict.nn import inv_softplus
>>> inv_softplus(torch.tensor([1.0, 2.0]))
tensor([0.5413, 1.8546])
>>> torch.nn.functional.softplus(inv_softplus(torch.tensor([1.0, 2.0])))
tensor([1., 2.])
```