# TensorDictBase.clear_refs_for_compile

TensorDictBase.clear_refs_for_compile_() → Self

Clears the record of the last operation of the tensordict and its nested tensordicts.

The tensordict records its last operation, with a weak reference to its
result, for the `with` form of methods such as [`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_). Call
this method on a tensordict before you return it from a function
compiled with [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) when compiling fails with
`torch._dynamo.exc.Unsupported: reconstruct: WeakRefVariable()`.
See [torch.compile, torch.func, torch.export, and CUDA graphs](../../compile.html#compile).

Returns:

`self`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).