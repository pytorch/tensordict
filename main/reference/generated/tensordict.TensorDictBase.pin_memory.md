# TensorDictBase.pin_memory

TensorDictBase.pin_memory(*num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Calls [`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) on the stored tensors.

Parameters:

- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - if provided, the number of threads to use
to call `pin_memory` on the leaves. Defaults to `None`, which sets a high
number of threads in `ThreadPoolExecutor(max_workers=None)`.
To execute all the calls to [`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) on the main thread, pass
`num_threads=0`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict is modified in-place.
Defaults to `False`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).