# TensorDictBase.amax

TensorDictBase.amax(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | NO_DEFAULT = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the maximum values of all elements in the input tensordict.

Same as [`max()`](tensordict.TensorDictBase.max.html#tensordict.TensorDictBase.max) with `return_indices=False`. A tuple `dim`
reduces only its first dim. Starting with TensorDict 0.17, the default
will change to reducing every dim of the tuple, as [`torch.amax()`](https://docs.pytorch.org/docs/stable/generated/torch.amax.html#torch.amax)
does.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).