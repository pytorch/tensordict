# Which container should I use?

Most code needs one of two forms of the same container. Use a
[`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), the dict form, when your program chooses the
keys at run time. Use a [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass), the class form, when
you know the fields in advance and want methods on them. Both forms have the
same tensor operations, and `from_tensordict` wraps the dict form in the
class form without a copy:

```
>>> import torch
>>> from tensordict import TensorClass, TensorDict
>>> td = TensorDict({"pos": torch.zeros(4, 3), "vel": torch.zeros(4, 3)}, batch_size=[4])
>>> td["mass"] = torch.ones(4) # add a key at run time
>>> class Particles(TensorClass):
... pos: torch.Tensor
... vel: torch.Tensor
... mass: torch.Tensor
>>> particles = Particles.from_tensordict(td) # wraps td
>>> particles.pos += 1 # writes to td
>>> td["pos"][0]
tensor([1., 1., 1.])
```

`from_tensordict` raises `ValueError` if the tensordict has a key that is
not a field. `to_tensordict()` returns the data in the dict form, as a copy.

The other containers are special cases: items of different shapes, data on
disk or on a server, and typed fields on a `TensorDictBase`. All the
containers except the plain `dict` have a *batch size*: the leading
dimensions that all their entries share (see [Batch size, shapes, and indexing](concepts/batch_size.html#batch-size)). Indexing,
reshaping, and stacking apply to these dimensions.

## The containers

| Container | Use it for | Main constraint |
| --- | --- | --- |
| [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) | Tensors that share their batch dimensions, under string keys or nested keys that you add and remove at any time. The data is in memory, on any device. | Each entry must start with the batch dimensions. Any other value raises `RuntimeError: batch dimension mismatch`. |
| [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) | A fixed set of fields that you declare like a dataclass, with attribute access, your own methods, and plain Python values for fields that are not tensors. | It is not a [`TensorDictBase`](reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) or a `Mapping`: `tc["key"]` and `**tc` raise `ValueError`, and a function that tests `isinstance(x, TensorDictBase)` rejects it. |
| [`LazyStackedTensorDict`](reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) | A stack of tensordicts whose entries have different shapes, for example sequences of different lengths. [`lazy_stack()`](reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack) makes one. It keeps the stacked tensordicts, so a write to the stack changes them. | Each read of an entry stacks it into a new tensor. An entry with different shapes raises `RuntimeError` unless you ask for a padded tensor, a nested tensor, or a list (`get(key, as_padded_tensor=True)`). `keys()` lists only the keys that all the stacked tensordicts have. |
| [`PersistentTensorDict`](reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) | Data in an HDF5 or Zarr file, read on demand. The file can be larger than the memory. | Each read loads a new tensor from the file, so an in-place operation on that tensor does not change the file. The in-place methods of the tensordict, such as `add_()` and `update_()`, write to the file when it is open for writing (`mode="r+"`); `from_h5()` and `from_zarr()` open it read-only by default. Needs `h5py` or `zarr`. |
| [`TensorDictStore`](reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) | Data on a Redis, Dragonfly, or KeyDB server, which several processes or machines can read and write. | Needs a running server and the `redis` package. Each read goes over the network. The data stays on the server until you call `clear_redis()`. |
| [`TypedTensorDict`](reference/generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict) | Declared fields on a class that is a `TensorDictBase`, with `**state` unpacking and the full tensordict API. | Experimental: its API can change in a minor release without a deprecation cycle (see [TypedTensorDict](typed_tensordict.html#typed-tensordict-guide)). Attribute access returns a string field as a [`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) wrapper. No custom `__init__` and no `__post_init__`. |
| A plain `dict` of tensors | Any Python values, with no rule on their shapes. | No batch dimensions: you write a loop, or a `torch.utils._pytree.tree_map` call, for each operation on the values. |

On `PersistentTensorDict` and `TensorDictStore`, use `set_()` for each
key, or `update(d, inplace=True)`, instead of `update_()`.
[Containers on storage backends](compatibility.html#compatibility) lists this limit with the other limits of each backend.

## Choose by need

- If your program chooses the keys at run time, use a
[`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict). See [What is TensorDict?](overview.html#overview).
- If you know the fields and want methods of your own and completion in your
editor, use a [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass). See [TensorClass: a dataclass of tensors](tensorclass.html#tensorclass-guide).
- If you want declared fields on an object that is a `TensorDictBase`, use
a [`TypedTensorDict`](reference/generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict). It accepts `state["key"]` and
`**state`, and a function that takes a tensordict accepts it. See
[TypedTensorDict](typed_tensordict.html#typed-tensordict-guide).
- If you stack items of different shapes, use
[`lazy_stack()`](reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack), which returns a
[`LazyStackedTensorDict`](reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict). See [Stacking and variable-size data](concepts/stacking.html#stacking).
- If your dataset is larger than the memory and is on a local disk, save a
`TensorDict` with [`memmap_()`](reference/generated/tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_) and load it with
[`load_memmap()`](reference/generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap). Its entries are then
memory-mapped tensors. See [Save and load tensordicts](saving.html#saving).
- If your data is in HDF5 or Zarr files, use a
[`PersistentTensorDict`](reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict). See [Large datasets and storage backends](storage.html#storage).
- If several processes or machines must share the data through a server, use
a [`TensorDictStore`](reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore). See [Large datasets and storage backends](storage.html#storage).
- If you want typed fields on a file, a server, or a lazy stack, wrap it with
`from_tensordict` of a `TensorClass` or a `TypedTensorDict`. See
[Containers on storage backends](compatibility.html#compatibility).
- If your values have no common leading dimension, use a plain `dict`, or
put them in a tensordict with `batch_size=[]`.

## TensorClass or TypedTensorDict?

`TypedTensorDict` is experimental: its API can change in a minor release
without a deprecation cycle (see [TypedTensorDict](typed_tensordict.html#typed-tensordict-guide)).
`TensorClass` follows the normal deprecation rules.

Both declare typed fields with class annotations, and both can wrap a
tensordict of any backend. A [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) holds a
tensordict, and a [`TypedTensorDict`](reference/generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict) is one. The other
differences follow from this:

| Feature | `TensorClass` | `TypedTensorDict` |
| --- | --- | --- |
| Declaration | `class Foo(TensorClass)`, or the legacy `@tensorclass` decorator | `class Foo(TypedTensorDict)` |
| `isinstance(obj, TensorDictBase)` | `False`. To accept both, use [`is_tensor_collection()`](reference/generated/tensordict.is_tensor_collection.html#tensordict.is_tensor_collection). | `True` |
| `obj["key"]` | Raises `ValueError`. Use `obj.key` or `obj.get("key")`. | Returns the entry. |
| `**obj` | Raises `ValueError`. Use `**obj.to_dict()`, which also turns nested tensorclasses into dicts. | Works: a `TypedTensorDict` is a `MutableMapping`. |
| A `str` field | Attribute access returns the `str`. | Attribute access returns a `NonTensorData` wrapper. `obj["key"]` returns the `str`. |
| A Python number | Becomes a tensor, unless the class has the `nocast` flag. | Becomes a tensor. |
| Optional fields | `x: Tensor \| None = None`. A field that is not set is not in `keys()`. | The same as for `TensorClass`. |
| Class options | `autocast`, `nocast`, `tensor_only`, `frozen`, `shadow` | `shadow` and `frozen`. The other three have no effect, and are deprecated. |
| `frozen` | Attribute assignment raises `dataclasses.FrozenInstanceError`. | Locks the tensordict: attribute assignment raises `RuntimeError`. |
| Methods and properties | Supported | Supported |
| Custom `__init__`, `__post_init__` | Supported | Not supported |
| Wrap another backend | `Foo.from_tensordict(td)`. Writes go to `td`. | The same as for `TensorClass`. |
| Wrap the other class | `Foo.from_tensordict(ttd)` works. | `Foo.from_tensordict(tc)` reads the fields, but an attribute write raises `ValueError`. |
| Type checkers (pyright, mypy) | Report a field value of the wrong type, a missing field, and an unknown keyword argument in the constructor. | Type the fields. Only mypy reports a missing field. Neither reports an unknown keyword argument. |

Use `TensorClass` if:

- Your fields include strings or other Python objects that you read as plain
values.
- You need a custom `__init__`, `__post_init__`, or the `autocast` flag.

Use `TypedTensorDict` if:

- The object must be a `TensorDictBase`.
- Your pipeline builds its state in stages and passes it on with
`**state`.

## See also

- [What is TensorDict?](overview.html#overview): what a tensordict is.
- [TensorClass: a dataclass of tensors](tensorclass.html#tensorclass-guide) and [TypedTensorDict](typed_tensordict.html#typed-tensordict-guide): the two typed
containers.
- [Stacking and variable-size data](concepts/stacking.html#stacking): lazy stacks and data of variable size.
- [Large datasets and storage backends](storage.html#storage) and [Save and load tensordicts](saving.html#saving): the backends on disk and on servers.
- [Containers on storage backends](compatibility.html#compatibility): the typed containers on each backend.
- [Coming from JAX pytrees](coming_from_jax.html#coming-from-jax): the containers for users of JAX pytrees.
- [tensordict: containers and functions](reference/td.html#api-tensordict), [tensorclass and non-tensor data](reference/tc.html#api-tensorclass), and
[TypedTensorDict](reference/ttd.html#api-typed-tensordict): the API reference.