# TensorDictBase.type

TensorDictBase.type(*dst_type: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*, ***, *cast_filter: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Casts all tensors to `dst_type`.

Parameters:

**dst_type** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*or**string*) - the desired type

Keyword Arguments:

**cast_filter** ([*callable*](https://docs.python.org/3/builtins/functions.html#callable)*,**optional*) - a predicate `fn(key, value) -> bool`
that selects the tensor leaves to cast, as in [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).