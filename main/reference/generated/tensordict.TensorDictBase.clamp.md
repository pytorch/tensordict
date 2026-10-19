# TensorDictBase.clamp

TensorDictBase.clamp(*min: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *max: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *out=None*) → Self

Returns a new tensordict with every leaf clamped into the range `[min, max]`.

\[y_i = \min(\max(x_i, \text{min}_i), \text{max}_i)\]

Where `min` is greater than `max`, the result is `max`, as with
[`torch.clamp()`](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp). At least one of `min` and `max` must be given;
with only one bound, `clamp` calls [`clamp_min()`](tensordict.TensorDictBase.clamp_min.html#tensordict.TensorDictBase.clamp_min) or
[`clamp_max()`](tensordict.TensorDictBase.clamp_max.html#tensordict.TensorDictBase.clamp_max).

Parameters:

- **min** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**Number**or**None**,**optional*) - the
lower bound. A tensordict is paired with the tensordict key by
key. `min` and `max` must both be tensordicts, or neither.
Defaults to `None`, which sets no lower bound.
- **max** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**Number**or**None**,**optional*) - the
upper bound. `min` and `max` must both be tensordicts, or
neither. Defaults to `None`, which sets no upper bound.

Keyword Arguments:

**out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the tensordict to write the result
into, and the one returned. The result is copied into its
leaves, so `out` must have the keys, shapes and dtypes of
the result. Defaults to `None`.

Raises:

- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if one of `min` and `max` is a tensordict and the
 other is not, or if `out` is given and both `min` and
 `max` are `None`.
- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if `out` does not have the keys of the result.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a leaf of `out` does not have the dtype of the
 result, as with [`torch.clamp()`](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.tensor([-2.0, 0.5, 3.0]), batch_size=[3])
>>> td.clamp(0.0, 1.0)["x"]
tensor([0.0000, 0.5000, 1.0000])
>>> td.clamp(max=0.0)["x"]
tensor([-2., 0., 0.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).