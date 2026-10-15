# TensorDictBase.get_item_shape

TensorDictBase.get_item_shape(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*)

Returns the shape of the entry, without reading it with [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) when possible.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).