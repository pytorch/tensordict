# LazyStackedTensorDictStore

`LazyStackedTensorDictStore` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `LazyStackedTensorDictStore` adds, and the overrides that have their own
documentation.

*class*tensordict.store.LazyStackedTensorDictStore(***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *count: [int](https://docs.python.org/3/builtins/functions.html#int)*, *stack_dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *inner_batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)]*, *device=None*, *client=None*, *td_id: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *cache_metadata: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***redis_kwargs*)

Bases: `_StoreUnsupported`, [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

A lazy stack of tensordicts whose entries live on a Redis-compatible server.

Warning

`LazyStackedTensorDictStore` is experimental: its API can change in a
minor release without a deprecation cycle.

[`TensorDictStore.from_tensordict`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore.from_tensordict)
returns this class when you pass a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict),
so that the elements of the stack can have entries of different shapes.
You can also call `from_lazy_stack()`, and `from_store()` to
reconnect. The class behaves like a lazy stack, but it is a
[`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase) subclass, not a
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) subclass.

Supports [Redis](https://redis.io), [Dragonfly](https://dragonflydb.io),
[KeyDB](https://docs.keydb.dev), and any other Redis-wire-compatible
server. The data lifetime, `close` and `clear_redis` work as for
[`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) (see its "Lifecycle and
limits" section).

Warning

`store["nested"]` is an in-memory copy of the nested entry: writes
through it, for example `store["nested"].set_(...)` or
`store["nested"].zero_()`, change the copy only, not the store.
Write with the full key, `store.set(("nested", "b"), ...)`, or with
[`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update).

Stores each leaf key as a single concatenated blob in the store,
regardless of how many stack elements there are. For *N* elements and
*K* leaf keys this uses only *O(K)* keys (plus offset tables for
heterogeneous shapes).

Two storage modes are supported:

- Homogeneous: all stack elements have the same shape per key.
Byte offsets are computed arithmetically (no offset table stored).
- Heterogeneous: element shapes can differ per key. An offset
table (packed int64 array of *N+1* byte offsets) is stored alongside
the data blob, and per-element shapes are persisted in the metadata
hash.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Store backend (`"redis"`, `"dragonfly"`, etc.).
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Unix domain socket path.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Key namespace. Defaults to `"tensordict"`.
- **count** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Number of stack elements (*N*). Required.
- **stack_dim** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Dimension along which the stack was performed.
- **inner_batch_size** (*Sequence**[*[*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*]*) - Batch size of each element. Required.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device for retrieved tensors.
- **client** - Existing `redis.asyncio.Redis` client.
- **td_id** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - UUID for reconnecting to existing data.
- **cache_metadata** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - Cache (shape, dtype) locally.
- ****redis_kwargs** - Extra connection keyword arguments.

Examples

The example needs a Redis server on `localhost:6379`.

```
>>> import torch
>>> from tensordict import TensorDict, lazy_stack
>>> from tensordict.store import TensorDictStore
>>> local = lazy_stack([TensorDict(x=torch.zeros(3)), TensorDict(x=torch.zeros(5))])
>>> store = TensorDictStore.from_tensordict(local)
>>> type(store).__name__
'LazyStackedTensorDictStore'
>>> store[0]["x"].shape, store[1]["x"].shape
(torch.Size([3]), torch.Size([5]))
>>> store[1] = TensorDict(x=torch.ones(5))
>>> store[1]["x"]
tensor([1., 1., 1., 1., 1.])
>>> store.clear_redis()
>>> store.close()
```

clear_redis()

Delete all keys associated with this TensorDict from Redis.

flatten_keys(*separator='.'*, *inplace=False*)

Returns an in-memory copy where the nested keys are joined by `separator`.

The content is first loaded with [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict).

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the separator between the nested items. Defaults to `"."`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - must be `False`; `True` raises a `ValueError`.
Defaults to `False`.

See [`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys).

*classmethod*from_dict(*input_dict*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_size=None*, *device=None*, ***kwargs*)

Not directly supported -- use `from_lazy_stack()` instead.

*classmethod*from_lazy_stack(*lazy_td*, ***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *device=None*, ***kwargs*) → LazyStackedTensorDictStore

Upload a `LazyStackedTensorDict` to a key-value store.

Data is sent in chunks of 10,000 elements, so that the full stack is
not materialized in memory.

Parameters:

**lazy_td** ([*LazyStackedTensorDict*](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)) - The source lazy stack.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Store backend (`"redis"`, `"dragonfly"`, etc.).
A label only.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Unix socket path. If set,
`host` and `port` are ignored.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Key namespace. Defaults to `"tensordict"`.
- **device** - Device override for retrieved tensors.
- ****kwargs** - Extra connection keyword arguments.

Returns:

A `LazyStackedTensorDictStore` backed by the uploaded data.

*classmethod*from_store(***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *td_id: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *device=None*, ***kwargs*) → LazyStackedTensorDictStore

Reconnect to an existing LazyStackedTensorDictStore on a server.

Reads the number of elements, the stack dimension, the batch size of
the elements and the device from the server.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Store backend. A label only.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Unix socket path. If set,
`host` and `port` are ignored.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Key namespace. Defaults to `"tensordict"`.
- **td_id** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - The identifier of the store to reconnect to, from
the private attribute `_td_id` of the writer or from its
`repr`. Required.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device override for retrieved
tensors. If `None`, uses the device stored on the server.
- ****kwargs** - Extra connection keyword arguments.

Returns:

A `LazyStackedTensorDictStore` connected to the existing data.

to_local() → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Pulls the whole store into a local `TensorDict`.

Deprecated since version 0.15: Use [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) instead.