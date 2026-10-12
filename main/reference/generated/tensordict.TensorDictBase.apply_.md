# TensorDictBase.apply

TensorDictBase.apply_(*fn: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)*, **others*, ***kwargs*) → Self

Applies a callable to all values stored in the tensordict and re-writes them in-place.

Parameters:

- **fn** (*Callable*) - function to be applied to the tensors in the
tensordict.
- ***others** (*sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the other
tensordicts to be used.

Keyword Args: See [`apply()`](tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply).

Returns:

self or a copy of self with the function applied

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).