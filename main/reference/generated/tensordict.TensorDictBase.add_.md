# TensorDictBase.add

TensorDictBase.add_(*other: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [float](https://docs.python.org/3/builtins/functions.html#float)*, ***, *alpha: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

In-place version of [`add()`](tensordict.TensorDictBase.add.html#tensordict.TensorDictBase.add).

Note

In-place `add` does not support `default` keyword argument.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).