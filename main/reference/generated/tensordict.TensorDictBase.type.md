# TensorDictBase.type

TensorDictBase.type(*dst_type: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*, ***, *cast_filter: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Returns a tensordict whose tensor leaves are cast to `dst_type`.

Each leaf is cast with [`torch.Tensor.type()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.type.html#torch.Tensor.type).

Parameters:

**dst_type** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - the dtype, or the name of a tensor
type such as `"torch.IntTensor"`.

Keyword Arguments:

**cast_filter** (*Callable**,**optional*) - a predicate
`fn(key, value) -> bool` that selects the tensor leaves to
cast, as in [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to). Defaults to `None`, which casts every
tensor leaf (a tensorclass uses its own `cast_filter`, if it
declares one).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).