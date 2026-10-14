# TensorDictStore

`TensorDictStore` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TensorDictStore` adds, and the overrides that have their own
documentation.

*class*tensordict.store.TensorDictStore(***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *batch_size=None*, *device=None*, *client=None*, *td_id: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *cache_metadata: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***redis_kwargs*)

Bases: `_StoreConnection`, `_StoreUnsupported`, [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

A TensorDict whose entries live on a Redis-compatible server.

Supports [Redis](https://redis.io), [Dragonfly](https://dragonflydb.io),
[KeyDB](https://docs.keydb.dev), and any other Redis-wire-compatible server.

Use it to share data between processes or machines, or to keep data out of
the RAM of the training process. Each tensor entry is stored as one string
of raw bytes on the server. Its shape and dtype are stored next to it, so
that `keys()` and `shape` do not download the data. See
[Large datasets and storage backends](../../storage.html#storage) for a guide.

Each read or write blocks until the server answers. Internally, the client
is a `redis.asyncio.Redis` client that runs on a background event loop,
so that one call can send many commands in one round-trip.

Lifecycle and limits

- The store needs a running server. By default it connects to
`localhost:6379`. To start a local Redis server, run
`docker run -d -p 6379:6379 redis`.
- The data stays on the server after `close()`, after the object is
garbage collected, and after the process ends. The server keeps it in
its memory until you delete it.
- To reconnect from another process, pass the `td_id` of the store to
`from_store()`, which also reads the batch size and the device. The
id is in the private attribute `_td_id`, and is shown by `repr`.
There is no public accessor.
- `close()` closes the connection and stops the background loop. The
object cannot read or write after that, and the data stays on the
server. `clear_redis()` deletes the data and keeps the connection
open.
- Each tensor entry is one string on the server. Redis limits a string to
512 MB by default (`proto-max-bulk-len`), and a larger write raises
`redis.exceptions.ResponseError`. For larger entries, raise the limit
on the server, or split the entry.
- Non-tensor values are stored as JSON. A value that JSON cannot encode
is stored with [`pickle`](https://docs.python.org/3/library/pickle.html#module-pickle), and each read unpickles it. Any client
that can write to the server can then run code in the readers, so
connect only to a server that you trust.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Name of the server backend. Accepted values include
`"redis"` (default), `"dragonfly"`, `"keydb"`, or any
other string. It is a label, kept in the object and shown by
`repr`. All servers use the same wire protocol, so the value
does not change how the store connects.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Path to a Unix domain socket, for a
server on the same machine. If set, `host` and `port` are
ignored.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - A namespace prefix for all keys.
Defaults to `"tensordict"`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**compatible*) - The TensorDict batch size.
Defaults to `torch.Size(())`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible**,**optional*) - Target device for
retrieved tensors. Defaults to `None` (CPU).
- **client** - An existing `redis.asyncio.Redis` client instance. If
provided, `host`/`port`/`db`/`unix_socket_path` are ignored.
- **td_id** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - A unique identifier for this TensorDict in the
store. Defaults to a new UUID. Pass an existing ID to reconnect
to previously stored data. The constructor does not read the
stored batch size: use `from_store()` to reconnect.
- **cache_metadata** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - If `True` (default), cache tensor metadata
locally to reduce round-trips.
- ****redis_kwargs** - Additional keyword arguments passed to
`redis.asyncio.Redis`.

Examples

The example needs a Redis server on `localhost:6379`.

```
>>> import torch
>>> from tensordict.store import TensorDictStore
>>> # Redis backend (default)
>>> td = TensorDictStore(batch_size=[100])
>>> td["obs"] = torch.randn(100, 84)
>>> td["obs"].shape # fetched from the store
torch.Size([100, 84])
>>> local_td = td.to_tensordict()
>>> type(local_td).__name__
'TensorDict'
>>> td.clear_redis() # delete the data on the server
>>> td.close()
```

To connect to a Dragonfly server (same wire protocol as Redis) on
another host, pass its address, for example
`TensorDictStore(backend="dragonfly", host="dragonfly-host", batch_size=[100])`.

Note

Requires `redis` package: `pip install redis`.

Note

Tensors are stored as CPU contiguous bytes. The `device` attribute
controls the device tensors are cast to upon retrieval.

flatten_keys(*separator='.'*, *inplace=False*)

Returns an in-memory copy where the nested keys are joined by `separator`.

The content is first loaded with [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict).

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the separator between the nested items. Defaults to `"."`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - must be `False`; `True` raises a `ValueError`.
Defaults to `False`.

See [`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys).

*classmethod*from_dict(*input_dict*, ***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_size=None*, *device=None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*)

Create a TensorDictStore from a dictionary or TensorDict.

Parameters:

**input_dict** - A dictionary or TensorDict to store.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Store backend (`"redis"`, `"dragonfly"`, etc.).
A label only (see `TensorDictStore`).
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Unix socket path. If set,
`host` and `port` are ignored.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Key namespace. Defaults to `"tensordict"`.
- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if `True`, a dict `input_dict` without
`batch_size` gives the store the largest leading shape that
all its entries share, as
[`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_) gives.
Defaults to `False`.
- **batch_size** - Explicit batch_size. Defaults to the batch size of
`input_dict` if it is a TensorDict. For a dict, defaults to
`[]`, or to the inferred batch size with
`auto_batch_size=True`.
- **device** - Target device for tensor retrieval.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - for a dict `input_dict` without
`batch_size`, the number of leading dims of the inferred batch
size; implies `auto_batch_size=True`. Exclusive with
`batch_size`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the dimension names of the store.
- ****kwargs** - Additional connection kwargs.

Returns:

A new TensorDictStore.

Examples

The example needs a Redis server on `localhost:6379`.

```
>>> import torch
>>> from tensordict.store import TensorDictStore
>>> store = TensorDictStore.from_dict({"a": torch.zeros(5, 3)}, batch_size=[5])
>>> store.batch_size
torch.Size([5])
>>> store.clear_redis()
>>> store.close()
```

*classmethod*from_schema(*schema: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[list](https://docs.python.org/3/builtins/stdtypes.html#list)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size), [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)]]*, ***, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *device=None*, ***kwargs*) → TensorDictStore

Creates a store with zero-filled entries of known shapes, without sending any data from Python.

Creates a new `TensorDictStore` and registers each key in
`schema` with its shape and dtype. The server allocates each entry
with `SETRANGE` and fills it with zeros. No tensor data goes through
the Python process, but the server allocates the full size in its
memory at once.

Use it for a replay buffer or another shared buffer whose schema you
know before you fill it.

Each entry is one string on the server. Redis limits a string to 512
MB by default (`proto-max-bulk-len`). A larger entry raises
`redis.exceptions.ResponseError`: raise the limit on the server, or
split the entry.

Parameters:

**schema** - Mapping from field name (a str, or a tuple of str for a
nested entry) to `(element_shape, dtype)`.
`element_shape` is the per-element shape (excluding
`batch_size`). The full stored shape is
`[*batch_size, *element_shape]`.

Keyword Arguments:

- **batch_size** - Overall batch size of the store.
- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Store backend. Defaults to `"redis"`.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Unix socket path.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Key namespace. Defaults to `"tensordict"`.
- **device** - Device override for retrieved tensors.
- ****kwargs** - Extra connection kwargs passed to the Redis client.

Returns:

A new `TensorDictStore` with pre-allocated (zero-filled) keys.

Examples

The example needs a Redis server on `localhost:6379`. The
`obs` entry takes 21 MB on the server.

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.store import TensorDictStore
>>> store = TensorDictStore.from_schema(
... {"obs": ([84, 84, 3], torch.uint8),
... "action": ([4], torch.float32),
... "reward": ([], torch.float32)},
... batch_size=[1000],
... )
>>> store["obs"].shape
torch.Size([1000, 84, 84, 3])
>>> store[0] = TensorDict(
... obs=torch.ones(84, 84, 3, dtype=torch.uint8),
... action=torch.ones(4),
... reward=torch.ones(()),
... batch_size=[],
... )
>>> store["action"][:2]
tensor([[1., 1., 1., 1.],
 [0., 0., 0., 0.]])
>>> store.clear_redis()
>>> store.close()
```

*classmethod*from_store(***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *td_id: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *device=None*, *tensorclass_cls: [type](https://docs.python.org/3/builtins/functions.html#type) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*) → TensorDictStore

Connect to an existing TensorDictStore on a remote server.

This is the cross-node entry point: one process stores data with
`from_tensordict()` (or regular `__setitem__`), and another
process on any machine that can reach the same server reconstructs
the handle by passing the same `td_id`.

Batch size and device are read from the metadata already persisted
in the store by the original writer.

If the original writer was a TensorClass instance, the class path
is stored automatically. On retrieval, the TensorClass is
reconstructed by importing the class. You can also pass
`tensorclass_cls` explicitly to override or skip the auto-import.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Store backend (`"redis"`, `"dragonfly"`, etc.).
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Unix socket path.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Key namespace. Defaults to `"tensordict"`.
- **td_id** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - The unique identifier of the TensorDict to reconnect
to. Obtain this from a previously created instance via
`td._td_id` (a private attribute: there is no public
accessor), or read it in its `repr`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device override for retrieved
tensors. If `None`, uses the device stored in the server.
- **tensorclass_cls** ([*type*](tensordict.TensorDictBase.type.html#tensordict.TensorDictBase.type)*|*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*|*[*None*](https://docs.python.org/3/builtins/constants.html#None)) - TensorClass to wrap the
result with. If a class, used directly via
`cls._from_tensordict`. If a string, interpreted as a
fully-qualified class path (`"module.ClassName"`) and
imported. If `None` (default), the class path stored in
the server (if any) is used automatically.
- ****kwargs** - Extra keyword arguments for the constructor, such as
`cache_metadata`, and connection kwargs for the Redis client.

Returns:

A TensorDictStore (or TensorClass wrapping one) connected to the
existing data.

Examples

On node A (writer):

```
td = TensorDictStore(host="my-server", batch_size=[100])
td["obs"] = torch.randn(100, 84)
print(td._td_id) # e.g. "a1b2c3d4-..."
```

On node B (reader):

```
td = TensorDictStore.from_store(
 host="my-server",
 td_id="a1b2c3d4-...",
)
td["obs"] # fetched from the shared store
```

*classmethod*from_tensordict(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *device=None*, ***kwargs*) → TensorDictStore

Upload a TensorDict to a key-value store.

Creates a new `TensorDictStore` and copies all tensor data from
the provided TensorDict into the store.

Parameters:

**td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The source TensorDict whose data will be
stored.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Store backend (`"redis"`, `"dragonfly"`, etc.).
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Unix socket path.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Key namespace. Defaults to `"tensordict"`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device override for retrieved
tensors. If `None`, uses the source TensorDict's device.
- ****kwargs** - Extra connection kwargs.

Returns:

A new TensorDictStore backed by the uploaded data.

A [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) gives a
[`LazyStackedTensorDictStore`](tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore), which keeps the
entries of different shapes.

Examples

The example needs a Redis server on `localhost:6379`. For a
server on another machine, pass `host`.

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.store import TensorDictStore
>>> local = TensorDict({"obs": torch.randn(10, 84)}, [10])
>>> remote = TensorDictStore.from_tensordict(local)
>>> remote["obs"].shape
torch.Size([10, 84])
>>> remote.clear_redis()
>>> remote.close()
```

to_local() → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Pull the entire Redis-backed dict into local RAM.

Alias for [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict).

Deprecated since version 0.15: Use [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) instead.