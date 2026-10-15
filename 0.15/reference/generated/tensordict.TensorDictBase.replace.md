# TensorDictBase.replace

TensorDictBase.replace(**args*, ***kwargs*)

Creates a shallow copy of the tensordict where entries have been replaced.

Accepts one unnamed argument which must be a dictionary of a [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass.
Additionally, first-level entries can be updated with the named keyword arguments.

Returns:

a copy of `self` with updated entries if the input is non-empty. If an empty dict or no dict is provided
and the kwargs are empty, `self` is returned.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).