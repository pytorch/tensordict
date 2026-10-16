# TensorDictBase.complex128

TensorDictBase.complex128() → Self

Casts all tensors to `torch.complex128`.

Keyword Arguments:

**cast_filter** ([*callable*](https://docs.python.org/3/builtins/functions.html#callable)*,**optional*) - a predicate `fn(key, value) -> bool`
that selects the tensor leaves to cast, as in [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).