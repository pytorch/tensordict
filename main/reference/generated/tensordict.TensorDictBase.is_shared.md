# TensorDictBase.is_shared

TensorDictBase.is_shared() → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the tensordict is in shared memory.

It is `True` after [`share_memory_()`](tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_), and always for a
tensordict on a CUDA device. A tensordict built from tensors that are
in shared memory is not marked as shared: only
[`share_memory_()`](tensordict.TensorDictBase.share_memory_.html#tensordict.TensorDictBase.share_memory_) marks it. A tensordict in shared memory is
locked: entries cannot be added, renamed or removed. See
[Locking and temporary changes](../../concepts/keys.html#keys-locking).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> td.is_shared()
False
>>> td.share_memory_().is_shared()
True
>>> td.is_locked
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).