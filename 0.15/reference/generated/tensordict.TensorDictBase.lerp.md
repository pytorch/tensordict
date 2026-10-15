# TensorDictBase.lerp

TensorDictBase.lerp(*end: TensorCollection | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *weight: TensorCollection | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*) → Self

Does a linear interpolation of two tensors `start` (given by `self`) and `end` based on a scalar or tensor `weight`.

\[\text{out}_i = \text{start}_i + \text{weight}_i \times (\text{end}_i - \text{start}_i)\]

The shapes of `start` and `end` must be
broadcastable. If `weight` is a tensor, then
the shapes of `weight`, `start`, and `end` must be broadcastable.

Parameters:

- **end** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)) - the tensordict with the ending points.
- **weight** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*,**tensor**or*[*float*](https://docs.python.org/3/builtins/functions.html#float)) - the weight for the interpolation formula.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).