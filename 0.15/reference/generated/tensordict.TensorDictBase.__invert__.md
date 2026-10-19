# TensorDictBase.__invert__

TensorDictBase.__invert__() → Self

Returns `~td`, a new tensordict with `~` applied to every leaf.

The operation is a logical NOT for boolean leaves and a bitwise NOT for
integer leaves.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).