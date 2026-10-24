# Containers on storage backends

A [`TensorClass`](reference/generated/tensordict.TensorClass.html#tensordict.TensorClass) or a [`TypedTensorDict`](reference/generated/tensordict.TypedTensorDict.html#tensordict.TypedTensorDict)
can wrap a tensordict on any storage backend without a copy. The backends are
an in-memory [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), an HDF5 or zarr file
([`PersistentTensorDict`](reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict)), a Redis server
([`TensorDictStore`](reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore)), a lazy stack
([`LazyStackedTensorDict`](reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)), or a memory-mapped
`TensorDict`. Code that uses the typed fields then reads and writes the file
or the server directly. A few operations behave differently on some backends:
see Operations on each backend.

`TypedTensorDict` is experimental: its API can change in a minor release
without a deprecation cycle (see [TypedTensorDict](typed_tensordict.html#typed-tensordict-guide)). To choose
between `TensorClass` and `TypedTensorDict`, see
[Which container should I use?](choosing_a_container.html#choosing-a-container). For the backends themselves, see [Large datasets and storage backends](storage.html#storage)
and [Save and load tensordicts](saving.html#saving).

## How the classes relate

```
TensorDictBase
├── TensorDict (in memory)
├── TypedTensorDict (typed fields, wraps any TensorDictBase)
├── PersistentTensorDict (HDF5 or zarr file)
├── TensorDictStore (Redis, Dragonfly or KeyDB server)
└── LazyStackedTensorDict (lazy stack of tensordicts)

TensorClass (typed fields, has a TensorDictBase)
```

A `TypedTensorDict` is a [`TensorDictBase`](reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase). A
`TensorClass` is not: it keeps its fields in a tensordict that it holds.
Both have a `from_tensordict(td)` class method that wraps `td`. Reads and
writes through the wrapper go to the storage of `td`. If you call the
constructor instead, the storage is a new in-memory `TensorDict`.

## Operations on each backend

The following operations work on each backend, directly or through a
`TensorClass` or a `TypedTensorDict`, with the exceptions in the next
table:

- Build: `MyTC.from_tensordict(td)`.
- Read a field: `w.a`.
- Write a field: `w.a = value`.
- In-place write: `w.set_("a", value)` and `w.update(d, inplace=True)`.
- Update: `w.update(d)`.
- Index: `w[0]` returns the wrapper class, and `w[0] = other` writes
into the backend.
- Clone: `w.clone()`.
- Stack: `torch.stack([w, w])`.
- Iterate: `list(w)`.

| Backend | Operation | What happens, and what to do |
| --- | --- | --- |
| Memory-mapped `TensorDict` | Write a field, `update(d)` | `memmap_()` locks the tensordict, so these raise `RuntimeError: Cannot modify locked TensorDict`. Use `set_()`, `update_()`, or `update(d, inplace=True)`, which write into the existing files. |
| `PersistentTensorDict` | Write a field, `update(d)` | Each write replaces the dataset in the file and warns `UserWarning: Replacing an array with another one is inefficient`. To write into the existing dataset, use `set_()` or `update(d, inplace=True)`. |
| `PersistentTensorDict` | `clone()` | Without a destination, it warns `A destination should be provided when cloning a PersistentTensorDict`. An HDF5 clone goes to a temporary file that is not deleted. A zarr clone raises `FileExistsError`. Pass a destination: `w.clone(newfile=path)`. To copy the data into memory, use `to_tensordict()`. |
| `PersistentTensorDict` opened read-only (the default of `from_h5()` and `from_zarr()`) | In-place methods such as `add_()`, `update_(d)`, `copy_(d)` | Write nothing, and warn `FutureWarning: Calling add_() on a tensordict stored in a file opened read-only, which leaves the file unchanged, is deprecated and will be removed in TensorDict 0.17`. From 0.17 on, these calls raise, as `zero_()` and `set_()` already do. Open the file with `mode="r+"` to write, or call `to_tensordict()` to change a copy in memory. |
| `PersistentTensorDict`, `TensorDictStore`, `LazyStackedTensorDict` | In-place operation on a field, such as `w.a.add_(1)` | Reading an entry returns a new tensor with a copy of the data, so the operation does not change the storage. Write the result back with `w.a = ...`, `set_()`, or an indexed assignment. |

### Nested tensorclass fields

A field can have another tensorclass as its type (`a: Pose` or
`a: Pose | None`). If the input is a `TensorDict` and its entry for that
field is a tensordict, `from_tensordict` builds a `Pose` from that entry,
recursively. The `Pose` goes into a shallow copy of the input. The leaves
are shared, but the input keeps its tensordict entry, and it does not see the
entries set later on the tensorclass. An entry with keys that `Pose` does
not declare stays a tensordict. On the other backends, `from_tensordict`
wraps the input as it is, and the nested entry stays a tensordict, so that
the wrapper writes nothing to their storage.

## A TensorClass on each backend

The examples write their files to a temporary directory, and use this class:

```
>>> import tempfile
>>> tmpdir = tempfile.TemporaryDirectory()
>>> path = tmpdir.name
>>> import torch
>>> from torch import Tensor
>>> from tensordict import TensorDict, TensorClass
>>> class MyTC(TensorClass):
... a: Tensor
... b: Tensor
```

### In-memory `TensorDict`

```
>>> td = TensorDict({"a": torch.randn(4, 3), "b": torch.randn(4, 5)}, batch_size=[4])
>>> tc = MyTC.from_tensordict(td)
>>> tc.a.shape
torch.Size([4, 3])
```

### HDF5 (`PersistentTensorDict`)

```
>>> import os
>>> from tensordict import PersistentTensorDict
>>> h5 = PersistentTensorDict.from_dict(td, filename=os.path.join(path, "data.h5"))
>>> tc_h5 = MyTC.from_tensordict(h5)
>>> tc_h5.a.shape # reads from the HDF5 file
torch.Size([4, 3])
```

### Redis (`TensorDictStore`)

These examples need a Redis server on `localhost`, so the docs build does
not run them. To start a server for a test, run
`docker run --rm -d -p 6379:6379 redis:7-alpine`.

```
>>> from tensordict.store import TensorDictStore
>>> store = TensorDictStore.from_tensordict(td, host="localhost") 
>>> tc_redis = MyTC.from_tensordict(store) 
>>> tc_redis.a.shape # fetched from Redis 
torch.Size([4, 3])
```

The data stays on the server after the process ends. To delete it, call
`store.clear_redis()`. To close the connection, call `store.close()`:

```
>>> store.clear_redis() 
>>> store.close()
```

### Lazy stack

```
>>> from tensordict import lazy_stack
>>> tds = [TensorDict({"a": torch.randn(3), "b": torch.randn(5)}) for _ in range(4)]
>>> ls = lazy_stack(tds, dim=0)
>>> tc_lazy = MyTC.from_tensordict(ls)
>>> tc_lazy[0].a.shape
torch.Size([3])
```

### Memory-mapped `TensorDict`

`memmap_()` locks the tensordict, so write with `set_()`:

```
>>> td_mmap = td.clone().memmap_(os.path.join(path, "tc_memmap"))
>>> tc_mmap = MyTC.from_tensordict(td_mmap)
>>> tc_mmap.a.shape
torch.Size([4, 3])
>>> _ = tc_mmap.set_("a", torch.ones(4, 3))
>>> bool((td_mmap["a"] == 1).all())
True
```

### `TypedTensorDict`

A `TypedTensorDict` can be the backend of a `TensorClass`. Both declare
fields, and the two declarations do not conflict:

```
>>> from tensordict import TypedTensorDict
>>> class MyTTD(TypedTensorDict):
... a: Tensor
... b: Tensor
>>> ttd = MyTTD(a=torch.randn(4, 3), b=torch.randn(4, 5), batch_size=[4])
>>> tc_typed = MyTC.from_tensordict(ttd)
>>> tc_typed.a.shape
torch.Size([4, 3])
```

The other direction is not supported: `TypedTensorDict.from_tensordict`
accepts a tensorclass, but an attribute write through it raises
`ValueError`.

## A TypedTensorDict on each backend

The examples use this class:

```
>>> class State(TypedTensorDict):
... x: Tensor
... y: Tensor
```

### In-memory `TensorDict`

The constructor creates an in-memory `TensorDict`:

```
>>> state = State(x=torch.randn(4, 3), y=torch.randn(4, 5), batch_size=[4])
>>> state.x.shape
torch.Size([4, 3])
```

`from_tensordict` wraps an existing `TensorDict` without a copy:

```
>>> td = TensorDict({"x": torch.randn(4, 3), "y": torch.randn(4, 5)}, batch_size=[4])
>>> state = State.from_tensordict(td)
>>> state.x.shape # reads from td
torch.Size([4, 3])
>>> state.x = torch.ones(4, 3) # writes to td
>>> (td["x"] == 1).all()
tensor(True)
```

### HDF5 (`PersistentTensorDict`)

```
>>> h5_state = PersistentTensorDict.from_dict(td, filename=os.path.join(path, "state.h5"))
>>> state = State.from_tensordict(h5_state)
>>> state.x.shape # reads from the HDF5 file
torch.Size([4, 3])
```

### Redis (`TensorDictStore`)

As above, these examples need a Redis server on `localhost`:

```
>>> store = TensorDictStore.from_tensordict(td, host="localhost") 
>>> state = State.from_tensordict(store) 
>>> state.x.shape # fetched from Redis 
torch.Size([4, 3])
>>> store.clear_redis() 
>>> store.close()
```

### Lazy stack

```
>>> tds = [TensorDict({"x": torch.randn(3), "y": torch.randn(5)}) for _ in range(4)]
>>> ls = lazy_stack(tds, dim=0)
>>> state = State.from_tensordict(ls)
>>> state[0].x.shape
torch.Size([3])
```

### Memory-mapped `TensorDict`

```
>>> td_mmap = td.clone().memmap_(os.path.join(path, "state_memmap"))
>>> state = State.from_tensordict(td_mmap)
>>> state.x.shape
torch.Size([4, 3])
>>> _ = state.set_("x", torch.zeros(4, 3))
>>> bool((td_mmap["x"] == 0).all())
True
```

### Stacking TypedTensorDicts

`torch.stack` returns an instance of the `TypedTensorDict` subclass:

```
>>> s1 = State(x=torch.randn(3), y=torch.randn(3), batch_size=[3])
>>> s2 = State(x=torch.randn(3), y=torch.randn(3), batch_size=[3])
>>> stacked = torch.stack([s1, s2], dim=0)
>>> type(stacked).__name__, stacked.batch_size
('State', torch.Size([2, 3]))
```

A lazy stack also works. An index of a `LazyStackedTensorDict` of
`TypedTensorDict` instances returns the subclass:

```
>>> from tensordict import LazyStackedTensorDict
>>> ls = LazyStackedTensorDict(s1, s2, stack_dim=0)
>>> isinstance(ls[0], State)
True
```

When you no longer need the files, close the HDF5 files and delete the
directory:

```
>>> h5.close()
>>> h5_state.close()
>>> tmpdir.cleanup()
```

To allocate storage on a Redis server first and fill it one sample at a time,
see [pre-allocating on Redis](storage.html#compat-redis-prealloc).

## See also

- [TensorClass: a dataclass of tensors](tensorclass.html#tensorclass-guide) and [TypedTensorDict](typed_tensordict.html#typed-tensordict-guide): the two typed
containers.
- [Which container should I use?](choosing_a_container.html#choosing-a-container): which container to use, and the
`TensorClass` and `TypedTensorDict` comparison.
- [Large datasets and storage backends](storage.html#storage): HDF5, zarr, Redis, and the other storage backends.
- [Save and load tensordicts](saving.html#saving): memory-mapped tensordicts and the other formats on disk.