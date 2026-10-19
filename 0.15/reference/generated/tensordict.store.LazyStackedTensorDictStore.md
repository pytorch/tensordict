# LazyStackedTensorDictStore

`LazyStackedTensorDictStore` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `LazyStackedTensorDictStore` adds, and the overrides that have their own
documentation.

*class*tensordict.store.LazyStackedTensorDictStore(***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *count: [int](https://docs.python.org/3/builtins/functions.html#int)*, *stack_dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *inner_batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)]*, *device=None*, *client=None*, *td_id: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *cache_metadata: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***redis_kwargs*)

Bases: `_StoreConnection`, `_StoreUnsupported`, [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

Each leaf key is one string on the server for the whole stack, with a
table of offsets when the shapes of the elements differ, so the number of
keys on the server does not grow with the number of elements.

The constructor makes an empty stack of `count` elements of batch size
`inner_batch_size`. To copy a lazy stack to the server, use
`from_lazy_stack()`.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a label for the kind of server, as in
[`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore). Defaults to
`"redis"`.
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
- **count** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the number of elements of the stack.
- **stack_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension of the stack in the batch
size. Defaults to `0`.
- **inner_batch_size** (*sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the batch size of each element.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the tensors
that reads return. Defaults to `None`: reads return CPU tensors,
and the store has no device.
- **client** (*redis.asyncio.Redis**,**optional*) - an existing client. If given,
`host`, `port`, `db` and `unix_socket_path` are ignored.
Defaults to `None`, which creates a client.
- **td_id** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the identifier of the store on the server.
Defaults to `None`, which creates a new UUID. To reconnect to an
existing store, use `from_store()`.
- **cache_metadata** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, keeps the shape and
dtype of each entry in the process, to save round-trips to the
server. Defaults to `True`.
- ****redis_kwargs** - passed to `redis.asyncio.Redis`.

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

*classmethod*from_dict(*input_dict*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_size=None*, *device=None*, ***kwargs*)

Raises a `NotImplementedError`: build the store with `from_lazy_stack()`.

The arguments are those of [`from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict),
and are not used.

*classmethod*from_lazy_stack(*lazy_td*, ***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *device=None*, ***kwargs*) → LazyStackedTensorDictStore

Returns a new store that holds a copy of the data of a lazy stack.

The data is sent in chunks of 10,000 elements, so that the full stack
is not built in memory at once.

Parameters:

**lazy_td** ([*LazyStackedTensorDict*](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)) - the lazy stack to copy to the
server.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a label for the kind of server, as in
[`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore). Defaults to
`"redis"`.
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
the device of `lazy_td`.
- ****kwargs** - passed to the constructor, such as `client` or
`td_id`, and options of the Redis client.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `lazy_td` is not a
 [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict).

Examples

The example needs a Redis server on `localhost:6379`.

```
>>> import torch
>>> from tensordict import TensorDict, lazy_stack
>>> from tensordict.store import LazyStackedTensorDictStore
>>> local = lazy_stack([TensorDict(x=torch.zeros(2)), TensorDict(x=torch.zeros(4))])
>>> store = LazyStackedTensorDictStore.from_lazy_stack(local) 
>>> store.batch_size, store[1]["x"].shape 
(torch.Size([2]), torch.Size([4]))
>>> store.clear_redis() 
>>> store.close()
```

*classmethod*from_store(***, *backend: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['redis', 'dragonfly'] = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *td_id: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *device=None*, ***kwargs*) → LazyStackedTensorDictStore

Returns a store connected to a lazy stack that another store wrote to the server.

Reads the number of elements, the stack dimension, the batch size of
the elements and the device from the server.

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a label for the kind of server, as in
[`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore). Defaults to
`"redis"`.
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
- ****kwargs** - passed to the constructor, such as `client` or
`cache_metadata`, and options of the Redis client.

Returns:

A `LazyStackedTensorDictStore` connected to the existing data.

Raises:

[**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the server has no lazy stack with this `td_id`.

Examples

The example needs a Redis server on `localhost:6379`. The
second store could be in another process.

```
>>> import torch
>>> from tensordict import TensorDict, lazy_stack
>>> from tensordict.store import LazyStackedTensorDictStore
>>> local = lazy_stack([TensorDict(x=torch.zeros(2)), TensorDict(x=torch.zeros(4))])
>>> writer = LazyStackedTensorDictStore.from_lazy_stack(local) 
>>> reader = LazyStackedTensorDictStore.from_store(td_id=writer._td_id) 
>>> reader[1]["x"].shape 
torch.Size([4])
>>> writer.clear_redis() 
>>> writer.close() 
>>> reader.close()
```

to_local() → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Returns a local copy of the data of the store.

Deprecated since version 0.15: Use [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) instead.