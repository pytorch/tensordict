# TensorDictBase.lerp

TensorDictBase.lerp_(*end: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *weight: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*)

Interpolates linearly from every leaf to `end` in place and returns `self`.

The in-place version of [`lerp()`](tensordict.TensorDictBase.lerp.html#tensordict.TensorDictBase.lerp).

Parameters:

- **end** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the end points. A tensordict
is paired with the tensordict key by key and must have all its
keys. A tensor is aligned with the batch dimensions of each
leaf and expanded over its feature dimensions.
- **weight** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*float*](https://docs.python.org/3/builtins/functions.html#float)) - the interpolation
weight: a number for all the leaves, or a tensordict or a
tensor paired like `end`. One of `end` and `weight` can
be a tensordict and the other a tensor.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).