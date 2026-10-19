# TensorDictBase.share_memory

TensorDictBase.share_memory_() → Self

Moves the leaves to shared memory, locks the tensordict, and returns `self`.

Processes that receive the tensordict, for example through
[`torch.multiprocessing`](https://docs.pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing), then read and write the same tensors.
The lock forbids adding, removing or replacing entries, but in-place
writes work (see [Locking and temporary changes](../../concepts/keys.html#keys-locking)). On a tensordict whose device is
CUDA, the call does nothing: CUDA tensors are shared already, and the
tensordict is not locked. After
[`unlock_()`](tensordict.TensorDictBase.unlock_.html#tensordict.TensorDictBase.unlock_), [`is_shared()`](tensordict.TensorDictBase.is_shared.html#tensordict.TensorDictBase.is_shared) returns `False`, although the
tensors stay in shared memory.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict is memory-mapped.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> td.share_memory_().is_shared(), td.is_locked, td["a"].is_shared()
(True, True, True)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).