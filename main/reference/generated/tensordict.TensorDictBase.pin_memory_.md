# TensorDictBase.pin_memory

TensorDictBase.pin_memory_(*num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 0*) → Self

Calls [`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) on the stored tensors and returns the TensorDict modifies in-place.

Deprecated since version 0.15: Use `pin_memory(inplace=True)` instead. Pinning allocates new
storage, so it does not fit the trailing-underscore convention.
`pin_memory_()` pins on the main thread by default
(`num_threads=0`); pass `num_threads=0` to
[`pin_memory()`](tensordict.TensorDictBase.pin_memory.html#tensordict.TensorDictBase.pin_memory) to keep that behavior.

Parameters:

**num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - if provided, the number of threads to use
to call `pin_memory` on the leaves. If `"auto"` is passed, the
number of threads is automatically determined.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).