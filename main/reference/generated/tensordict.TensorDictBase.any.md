# TensorDictBase.any

TensorDictBase.any(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [bool](https://docs.python.org/3/builtins/functions.html#bool) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)

Checks if any value is True/non-null in the tensordict.

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if `None`, returns a boolean indicating
whether all tensors return tensor.any() == True.
If integer, all is called upon the dimension specified if
and only if this dimension is compatible with
the tensordict shape.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).