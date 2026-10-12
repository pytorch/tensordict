# TensorDictBase.to_pytree

TensorDictBase.to_pytree()

Converts a tensordict to a PyTree.

If the tensordict was not created from a pytree, this method just returns `self` without modification.

See [`from_pytree()`](tensordict.TensorDictBase.from_pytree.html#tensordict.TensorDictBase.from_pytree) for more information and examples.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).