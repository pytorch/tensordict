# TensorDictStore

`TensorDictStore` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `TensorDictStore` adds, and the overrides that have their own
documentation.

*class*tensordict.store.TensorDictStore(***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *batch_size=None*, *device=None*, *client=None*, *td_id: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *cache_metadata: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***redis_kwargs*)

Bases: `_StoreConnection`, `_StoreUnsupported`, [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

A tensordict whose entries live on a Redis-compatible server.

Supports [Redis](https://redis.io), [Dragonfly](https://dragonflydb.io),
[KeyDB](https://docs.keydb.dev), and any other Redis-wire-compatible server.

Use it to share data between processes or machines, or to keep data out of
the RAM of the training process. Each tensor entry is stored as one string
of raw bytes on the server. Its shape and dtype are stored next to it, so
that `keys()` and `shape` do not download the data. See
[Large datasets and storage backends](../../storage.html#storage) for a guide.

Each read or write blocks until the server answers. One call sends its
commands to the server in one round-trip.

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

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a label for the kind of server, kept in the
store and shown by its `repr`, for example `"redis"`,
`"dragonfly"` or `"keydb"`. Every server speaks the same
protocol, so the value does not change how the store connects.
Defaults to `"redis"`.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the host name of the server. Defaults to
`"localhost"`.
- **port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the port of the server. Defaults to `6379`.
- **db** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the database number on the server. Defaults to
`0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the path of a Unix socket, for a
server on the same machine. If set, `host` and `port` are
ignored. Defaults to `None`.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the namespace of the keys of the store on the
server. Defaults to `"tensordict"`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the store: the leading dimensions that all its entries share.
Defaults to `None`, which gives `torch.Size([])`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
tensors that reads return. The server holds CPU bytes. Defaults to
`None`: reads return CPU tensors, and the store has no device.
- **client** (*redis.asyncio.Redis**,**optional*) - an existing client. If given,
`host`, `port`, `db` and `unix_socket_path` are ignored.
Defaults to `None`, which creates a client.
- **td_id** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the identifier of the store on the server.
Defaults to `None`, which creates a new UUID. The constructor
does not read the batch size of an existing store: use
`from_store()` to reconnect.
- **cache_metadata** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, keeps the shape and
dtype of each entry in the process, to save round-trips to the
server. Defaults to `True`.
- ****redis_kwargs** - passed to `redis.asyncio.Redis`.

Raises:

[**ModuleNotFoundError**](https://docs.python.org/3/builtins/exceptions.html#ModuleNotFoundError) - if the `redis` package is not installed
 (`pip install redis`).

Examples

The example needs a Redis server on `localhost:6379`.

```
>>> import torch
>>> from tensordict.store import TensorDictStore
>>> td = TensorDictStore(batch_size=[100]) 
>>> td["obs"] = torch.zeros(100, 84) 
>>> td["obs"].shape 
torch.Size([100, 84])
>>> type(td.to_tensordict()).__name__ 
'TensorDict'
>>> td.clear_redis() 
>>> td.close()
```

To connect to a Dragonfly server on another host, pass its address,
for example
`TensorDictStore(backend="dragonfly", host="dragonfly-host", batch_size=[100])`.

flatten_keys(*separator='.'*, *inplace=False*, *is_leaf=None*)

Returns a local copy of the store whose nested keys are joined by `separator`.

The data is read from the server with [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict), and the
store is not changed.

Parameters:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the string that joins the parts of a
nested key. Defaults to `"."`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - must be `False`; `True` raises a
`ValueError`. Defaults to `False`.
- **is_leaf** (*Callable**,**optional*) - see
[`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys). Defaults to
`None`.

See [`flatten_keys()`](tensordict.TensorDictBase.flatten_keys.html#tensordict.TensorDictBase.flatten_keys).

*classmethod*from_dict(*input_dict*, ***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_size=None*, *device=None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*)

Returns a new store that holds the entries of a dict or a tensordict.

Parameters:

**input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the entries to write to the
server.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a label for the kind of server, as in
`TensorDictStore`. Defaults to `"redis"`.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the host name of the server. Defaults to
`"localhost"`.
- **port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the port of the server. Defaults to `6379`.
- **db** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the database number on the server. Defaults
to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the path of a Unix socket, for a
server on the same machine. If set, `host` and `port` are
ignored. Defaults to `None`.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the namespace of the keys of the store on
the server. Defaults to `"tensordict"`.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and `input_dict`
is a dict without `batch_size`, sets the batch size to the
largest leading shape that all the entries share. Defaults to
`False`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size
of the store. Defaults to `None`, which takes the batch size
of a tensordict `input_dict`, and `[]` (or the inferred
batch size with `auto_batch_size=True`) for a dict.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
tensors that reads return, as in `TensorDictStore`.
Defaults to `None`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions; giving it turns
`auto_batch_size` on. Exclusive with `batch_size`.
Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions,
one per dimension. Defaults to `None`.
- ****kwargs** - passed to the `TensorDictStore` constructor, such
as `client` or `td_id`, and options of the Redis client.

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

Returns a new store with zero-filled entries of the given shapes and dtypes.

Each key of `schema` is registered with its shape and dtype. The
server allocates each entry with `SETRANGE` and fills it with zeros.
No tensor data goes through the Python process, but the server
allocates the full size in its memory at once.

Use it for a replay buffer or another shared buffer whose schema you
know before you fill it.

Each entry is one string on the server. Redis limits a string to 512
MB by default (`proto-max-bulk-len`). A larger entry raises
`redis.exceptions.ResponseError`: raise the limit on the server, or
split the entry.

Parameters:

**schema** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - maps each key (a str, or a tuple of str for a
nested entry) to `(feature_shape, dtype)`, where
`feature_shape` is the shape after the batch dimensions. The
entry has shape `[*batch_size, *feature_shape]`.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size
of the store. Defaults to `None`, which gives
`torch.Size([])`.
- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a label for the kind of server, as in
`TensorDictStore`. Defaults to `"redis"`.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the host name of the server. Defaults to
`"localhost"`.
- **port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the port of the server. Defaults to `6379`.
- **db** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the database number on the server. Defaults
to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the path of a Unix socket, for a
server on the same machine. If set, `host` and `port` are
ignored. Defaults to `None`.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the namespace of the keys of the store on
the server. Defaults to `"tensordict"`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
tensors that reads return, as in `TensorDictStore`.
Defaults to `None`.
- ****kwargs** - passed to the `TensorDictStore` constructor, such
as `client` or `td_id`, and options of the Redis client.

Returns:

A new `TensorDictStore` whose entries are zeros.

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

Returns a store connected to data that another store wrote to the server.

One process writes the data, with `from_tensordict()` or with
`td[key] = value`; another process, on any machine that reaches the
server, passes the same `td_id` to connect to it. The batch size and
the device are read from the server.

If the writer was a tensorclass, the server holds the path of its
class, and the result is an instance of that class, which must be
importable. `tensorclass_cls` overrides it.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a label for the kind of server, as in
`TensorDictStore`. Defaults to `"redis"`.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the host name of the server. Defaults to
`"localhost"`.
- **port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the port of the server. Defaults to `6379`.
- **db** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the database number on the server. Defaults
to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the path of a Unix socket, for a
server on the same machine. If set, `host` and `port` are
ignored. Defaults to `None`.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the namespace of the keys of the store on
the server. Defaults to `"tensordict"`.
- **td_id** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - the identifier of the store to connect to. A store
shows it in its `repr`, and holds it in the private attribute
`_td_id`; there is no public accessor.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
tensors that reads return. Defaults to `None`, which uses the
device stored on the server.
- **tensorclass_cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the tensorclass to wrap
the result in, as a class or as a path `"module.ClassName"`
to import. Defaults to `None`, which uses the class stored on
the server, if any.
- ****kwargs** - passed to the `TensorDictStore` constructor, such
as `client` or `cache_metadata`, and options of the Redis
client.

Returns:

A `TensorDictStore`, or a tensorclass that wraps one.

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the server has no store with this `td_id` under
 `prefix`.

Examples

The example needs a Redis server on `localhost:6379`. The
second store could be in another process.

```
>>> import torch
>>> from tensordict.store import TensorDictStore
>>> writer = TensorDictStore(batch_size=[100]) 
>>> writer["obs"] = torch.zeros(100, 84) 
>>> reader = TensorDictStore.from_store(td_id=writer._td_id) 
>>> reader.batch_size, reader["obs"].shape 
(torch.Size([100]), torch.Size([100, 84]))
>>> writer.clear_redis() 
>>> writer.close() 
>>> reader.close()
```

*classmethod*from_tensordict(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *device=None*, ***kwargs*) → TensorDictStore

Returns a new store that holds a copy of the data of `td`.

A [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) gives a
[`LazyStackedTensorDictStore`](tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore), which keeps
entries of different shapes. For a tensorclass, the store records its
class, so that `from_store()` returns an instance of it.

Parameters:

**td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict or tensorclass to copy to the
server.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a label for the kind of server, as in
`TensorDictStore`. Defaults to `"redis"`.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the host name of the server. Defaults to
`"localhost"`.
- **port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the port of the server. Defaults to `6379`.
- **db** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the database number on the server. Defaults
to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the path of a Unix socket, for a
server on the same machine. If set, `host` and `port` are
ignored. Defaults to `None`.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the namespace of the keys of the store on
the server. Defaults to `"tensordict"`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
tensors that reads return. Defaults to `None`, which takes
the device of `td`.
- ****kwargs** - passed to the `TensorDictStore` constructor, such
as `client` or `td_id`, and options of the Redis client.

Returns:

A `TensorDictStore`, or a
[`LazyStackedTensorDictStore`](tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore) for a lazy
stack.

Examples

The example needs a Redis server on `localhost:6379`. For a
server on another machine, pass `host`.

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.store import TensorDictStore
>>> local = TensorDict(obs=torch.zeros(10, 84), batch_size=[10])
>>> remote = TensorDictStore.from_tensordict(local) 
>>> remote["obs"].shape 
torch.Size([10, 84])
>>> remote.clear_redis() 
>>> remote.close()
```

to_local() → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Returns a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with a copy of the data of the store.

Deprecated since version 0.15: Use [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) instead.