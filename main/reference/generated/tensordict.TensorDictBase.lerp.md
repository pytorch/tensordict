# TensorDictBase.lerp

TensorDictBase.lerp(*end: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *weight: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*) → Self

Returns a new tensordict that interpolates linearly from every leaf (the start) to `end`.

\[\text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)\]

Parameters:

- **end** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the end points, paired with the tensordict
key by key. It must have every key of the tensordict.
- **weight** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*float*](https://docs.python.org/3/builtins/functions.html#float)) - the interpolation weight: a
number for all the leaves, or a tensordict paired key by key.

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