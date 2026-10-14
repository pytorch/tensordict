# TensorDictBase.pin_memory

TensorDictBase.pin_memory(*num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Calls [`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) on the stored tensors.

Parameters:

- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if provided, the number of threads to
use to call `pin_memory` on the leaves, in a
[`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor). Defaults to
`None`, which, like `0`, calls
[`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) on the calling thread.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict is modified
in-place. When every node is an unlocked
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) or
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) (or a tensorclass
over one) and no leaf is a [`MemoryMappedTensor`](tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor), in
shared memory, sparse, jagged or an
[`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor), each leaf is replaced by
its pinned copy, and the tensordict and every nested node stay
the same objects; keys that hold the same tensor share its
pinned copy, and a leaf that is a view of another leaf gets its
own. Any other tree, such as a locked one or one that holds a
memory-mapped tensordict, has the pinned copies written back
into its leaves, as before, so they stay in pageable memory.
Defaults to `False`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).