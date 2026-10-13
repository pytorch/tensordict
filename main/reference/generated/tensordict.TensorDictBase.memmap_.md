# TensorDictBase.memmap

TensorDictBase.memmap_(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *existsok: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*) → Self

Writes all tensors onto a corresponding memory-mapped Tensor, in-place.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - directory prefix where the memory-mapped tensors will
be stored. The directory tree structure will mimic the tensordict's.
- **copy_existing** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - If False (default), an exception will be raised if an
entry in the tensordict is already a tensor stored on disk
with an associated file, but is not saved in the correct
location according to prefix.
If `True`, any existing Tensor will be copied to the new location.

Keyword Arguments:

- **num_threads** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the number of threads used to write the memmap
tensors. Defaults to 0.
- **return_early** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and `num_threads>0`,
the method will return a future of the tensordict. The resulting
tensordict can be queried using future.result().
- **share_non_tensor** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the non-tensor data will be
shared between the processes and writing operation (such as inplace update
or set) on any of the workers within a single node will update the value
on all other workers. If the number of non-tensor leaves is high (e.g.,
sharing large stacks of non-tensor data) this may result in OOM or similar
errors. Defaults to `False`.
- **existsok** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `False`, an exception will be raised if a tensor already
exists in the same path. Defaults to `True`.
- **robust_key** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` (default), uses robust key encoding that safely
handles keys with path separators and special characters. If `False`,
uses legacy behavior (keys used as-is). If `None`, uses the default
robust behavior.

The TensorDict is then locked, meaning that any writing operations that
isn't in-place will throw an exception (eg, rename, set or remove an
entry).
Once the tensordict is unlocked, the memory-mapped attribute is turned to `False`,
because cross-process identity is not guaranteed anymore.

Returns:

self if `return_early=False`, otherwise a [`TensorDictFuture`](tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) instance.

Note

Serialising in this fashion might be slow with deeply nested tensordicts, so
it is not recommended to call this method inside a training loop.

Note

A [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict), a
[`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) or a
[`LazyStackedTensorDictStore`](tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore) (or one of its
elements) cannot be memory-mapped in-place: this method raises a
`RuntimeError`. Use [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap) to
build a memory-mapped copy instead.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `memmap_`

![](../../_images/sphx_glr_tensordict_memory_thumb.png)

[Devices and memory-mapped tensors](../../tutorials/tensordict_memory.html)

Devices and memory-mapped tensors