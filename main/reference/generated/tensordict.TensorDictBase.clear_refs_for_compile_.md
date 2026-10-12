# TensorDictBase.clear_refs_for_compile

TensorDictBase.clear_refs_for_compile_() → Self

Clears the weakrefs in order for the tensordict to get out of the compile region safely.

Use this whenever you hit torch._dynamo.exc.Unsupported: reconstruct: WeakRefVariable()
before returning a TensorDict.

Returns: self

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).