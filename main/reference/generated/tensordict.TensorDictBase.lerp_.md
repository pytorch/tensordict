# TensorDictBase.lerp

TensorDictBase.lerp_(*end: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*, *weight: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*)

Interpolates linearly from every leaf to `end` in place and returns `self`.

The in-place version of [`lerp()`](tensordict.TensorDictBase.lerp.html#tensordict.TensorDictBase.lerp).

Parameters:

- **end** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the end points, paired with the tensordict
key by key. It must have every key of the tensordict.
- **weight** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*float*](https://docs.python.org/3/builtins/functions.html#float)) - the interpolation weight: a
number for all the leaves, or a tensordict paired key by key.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).