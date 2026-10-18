# TensorDictBase.pin_memory

TensorDictBase.pin_memory_(*num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → Self

Pins the tensor leaves of the tensordict and returns `self`.

Deprecated since version 0.15: Use `pin_memory(inplace=True)` instead. Pinning allocates new
storage, so it does not fit the trailing-underscore convention
(see [In-place operations](../../concepts/operations.html#inplace-conventions)). `pin_memory_` will be removed
in TensorDict 0.17.

Parameters:

**num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of threads that pin the
leaves, as in [`pin_memory()`](tensordict.TensorDictBase.pin_memory.html#tensordict.TensorDictBase.pin_memory). Defaults to `0`, which
pins them on the calling thread.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).