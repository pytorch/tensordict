# rand_one_hot

tensordict.nn.rand_one_hot(*values: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *do_softmax: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Draws a one-hot sample along the last dimension of `values`.

Parameters:

- **values** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the logits, or the probabilities if
`do_softmax=False`, along the last dimension.
- **do_softmax** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, applies a softmax to
`values` first. Defaults to `True`.

Returns:

A `torch.long` tensor with the shape of `values`, with one `1`
per row of the last dimension.

Examples

```
>>> import torch
>>> from tensordict.nn import rand_one_hot
>>> _ = torch.manual_seed(0)
>>> rand_one_hot(torch.tensor([0.0, 2.0, 1.0]))
tensor([0, 1, 0])
>>> rand_one_hot(torch.tensor([[0.1, 0.9], [0.8, 0.2]]), do_softmax=False)
tensor([[0, 1],
 [1, 0]])
```