# TensorDictBase.float32

TensorDictBase.float32() → Self

Returns a tensordict whose tensor leaves are cast to `torch.float32`.

Keyword Arguments:

**cast_filter** (*Callable**,**optional*) - a predicate
`fn(key, value) -> bool` that selects the tensor leaves to
cast, as in [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to). Defaults to `None`, which casts every
tensor leaf (a tensorclass uses its own `cast_filter`, if it
declares one).

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).