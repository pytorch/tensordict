# TensorDictBase.lerp

TensorDictBase.lerp(*end: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *weight: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*) → Self

Returns a new tensordict that interpolates linearly from every leaf (the start) to `end`.

\[\text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)\]

Parameters:

- **end** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the end points. A tensordict
is paired with the tensordict key by key and must have all its
keys. A tensor is aligned with the batch dimensions of each
leaf and expanded over its feature dimensions.
- **weight** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*float*](https://docs.python.org/3/builtins/functions.html#float)) - the interpolation
weight: a number for all the leaves, or a tensordict or a
tensor paired like `end`. One of `end` and `weight` can
be a tensordict and the other a tensor. Where a lazy stack
meets a tensordict that is not one, a `weight` tensor that
does not broadcast with the batch dims broadcasts with each
leaf from the right instead, with a `FutureWarning`.
Starting with TensorDict 0.17, the default will change to
raising the error of the other tensordicts.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> start = TensorDict(x=torch.zeros(2), batch_size=[2])
>>> end = TensorDict(x=torch.full((2,), 10.0), batch_size=[2])
>>> start.lerp(end, 0.25)["x"]
tensor([2.5000, 2.5000])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).