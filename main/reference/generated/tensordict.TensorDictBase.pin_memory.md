# TensorDictBase.pin_memory

TensorDictBase.pin_memory(*num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Returns a tensordict whose tensor leaves are in pinned (page-locked) memory.

Calls [`pin_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.pin_memory.html#torch.Tensor.pin_memory) on every tensor leaf. Copies
from pinned memory to an accelerator are faster, and can run
asynchronously with `to(device, non_blocking=True)`. Pinning needs an
accelerator, such as CUDA. A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)
or a [`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) raises a
`RuntimeError`: call [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first. See
[Devices, dtypes, and memory](../../concepts/devices.html#devices).

Parameters:

- **num_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of threads that pin the
leaves, in a [`ThreadPoolExecutor`](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ThreadPoolExecutor).
Defaults to `None`, which, like `0`, pins them on the
calling thread.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict and its
nested tensordicts keep their identity, each leaf is replaced
by its pinned copy, and the call returns `self`. A locked
tensordict, one that holds a node other than a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), a lazy stack or a tensorclass,
or one with memory-mapped, shared, sparse, jagged or unbatched
leaves gets the pinned copies written back into its existing
leaves, which stay in pageable memory. Defaults to `False`.

Examples

The example needs an accelerator, so it is not run when the
documentation is tested.

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(obs=torch.zeros(1000, 3), batch_size=[1000])
>>> pinned = td.pin_memory() 
>>> pinned["obs"].is_pinned() 
True
>>> td_cuda = pinned.to("cuda", non_blocking=True)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).