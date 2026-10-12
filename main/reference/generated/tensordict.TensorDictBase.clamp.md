# TensorDictBase.clamp

TensorDictBase.clamp(*min: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *max: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *out=None*) → Self

Clamps all elements in `self` into the range [ [`min`](tensordict.TensorDictBase.min.html#tensordict.TensorDictBase.min), [`max`](tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max) ].

Letting min_value and max_value be [`min`](tensordict.TensorDictBase.min.html#tensordict.TensorDictBase.min) and [`max`](tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max), respectively, this returns:

> \[y_i = \min(\max(x_i, \text{min\_value}_i), \text{max\_value}_i)\]
> 
> 
> If [`min`](tensordict.TensorDictBase.min.html#tensordict.TensorDictBase.min) is `None`, there is no lower bound.
> Or, if [`max`](tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max) is `None` there is no upper bound.

Note

If [`min`](tensordict.TensorDictBase.min.html#tensordict.TensorDictBase.min) is greater than [`max`](tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max) [`torch.clamp(..., min, max)`](https://docs.pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp)
sets all elements in `input` to the value of [`max`](tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).