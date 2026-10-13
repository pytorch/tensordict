# TensorDictBase.consolidate

TensorDictBase.consolidate(*filename: [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *num_threads=0*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *use_buffer: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_memory: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *pin_memory: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *metadata: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Copies all the leaves into one contiguous storage, and returns a tensordict of views of it.

A consolidated tensordict pickles and moves between devices faster,
because its leaves are one block of memory. The result is locked (see
[`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_)): you can write into its leaves in place, but you
cannot add or remove entries.

Parameters:

**filename** (*Path**,**optional*) - an optional file path for a memory-mapped tensor
to use as a storage for the tensordict.

Keyword Arguments:

- **num_threads** (*integer**,**optional*) - the number of threads to use for populating
the storage. The leaves are copied by contiguous chunks of
roughly equal byte size, one chunk per thread. When writing to
a memory-mapped file with all leaves already on the target
device, `num_threads` is ignored and a single fused copy is
used instead, as concurrent writes to a fresh file mapping are
slower than a sequential one on most filesystems.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - an optional device where the storage must be
instantiated.
- **non_blocking** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - `non_blocking` argument passed to [`copy_()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.copy_.html#torch.Tensor.copy_).
- **inplace** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the resulting tensordict is the same
as `self` with updated values. Defaults to `False`.
- **return_early** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and `num_threads>1`,
the method will return a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture)
while the copies keep running in the background. The consolidated
tensordict can be queried using `future.result()`. Incompatible
with `use_buffer=True` and `non_blocking=True`. Defaults to
`False`.
- **use_buffer** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and a filename is passed, an intermediate
local buffer will be created in shared memory, and the data will be copied at
the storage location as a last step. This may be faster than writing directly
to a distant physical memory (e.g., NFS).
Defaults to `False`.
- **share_memory** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the storage will be placed in shared memory.
Defaults to `False`.
- **pin_memory** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether the consolidated data should be placed in pinned
memory. Defaults to `False`.
- **metadata** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the metadata will be stored alongside the
common storage. If a filename is provided, this is without effect.
Storing the metadata can be useful when one wants to control how serialization
is achieved, as TensorDict handles the pickling/unpickling of consolidated TDs
differently if the metadata is or isn't available.

Note

If the tensordict is already consolidated, all arguments are ignored and `self`
is returned. Call [`contiguous()`](tensordict.TensorDictBase.contiguous.html#tensordict.TensorDictBase.contiguous) to re-consolidate.

Note

A pickle (or [`torch.save()`](https://docs.pytorch.org/docs/stable/generated/torch.save.html#torch.save), [`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send), [`broadcast()`](tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast)) of a
consolidated tensordict is rebuilt from metadata recorded here. Setting
`batch_size` or `names` on the consolidated tensordict updates that
metadata; setting them directly on a nested tensordict, as in
`td["nested"].batch_size = ...`, may not. Call `consolidate()` again
afterwards: on a consolidated tensordict, it writes the current batch sizes
and names into that metadata.

Examples

```
>>> import pickle
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict({"a": torch.zeros(()), "b": {"c": torch.zeros(())}})
>>> data_consolidated = data.consolidate()
>>> # All the leaves share one storage.
>>> assert torch.tensor([
... v.untyped_storage().data_ptr() for v in data_consolidated.values(True, True)
... ]).unique().numel() == 1
>>> data_consolidated.is_consolidated(), data_consolidated.is_locked
(True, True)
>>> # A consolidated tensordict pickles as one buffer.
>>> restored = pickle.loads(pickle.dumps(data_consolidated))
>>> (restored == data).all()
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `consolidate`

![](../../_images/sphx_glr_serialization_speed_thumb.png)

[Benchmarking TensorDict serialization speed](../../tutorials/serialization_speed.html)

Benchmarking TensorDict serialization speed