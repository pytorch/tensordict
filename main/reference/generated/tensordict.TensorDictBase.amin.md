# TensorDictBase.amin

TensorDictBase.amin(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | NO_DEFAULT = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the minimum values of all elements in the input tensordict.

Same as [`min()`](tensordict.TensorDictBase.min.html#tensordict.TensorDictBase.min) with `return_indices=False`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).