# TensorDictBase.to_store

TensorDictBase.to_store(***, *backend: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *device=None*, ***kwargs*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Upload this TensorDict to a key-value store (Redis, Dragonfly, etc.).

Returns a [`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) (or
[`LazyStackedTensorDictStore`](tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore) for
lazy stacks) backed by the uploaded data.

For [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) inputs, data is streamed in chunks
to avoid materialising the full stack in memory.

Keyword Arguments:

- **backend** ([*STORE_BACKENDS*](tensordict.store.STORE_BACKENDS.html#tensordict.store.STORE_BACKENDS)) - Store backend -- `"redis"` (default)
or `"dragonfly"`.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Server hostname. Defaults to `"localhost"`.
- **port** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Server port. Defaults to `6379`.
- **db** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - Database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - Unix domain socket path.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - Key namespace. Defaults to `"tensordict"`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device override for retrieved
tensors. If `None`, uses this TensorDict's device.
- ****kwargs** - Extra connection keyword arguments.

Returns:

A store-backed TensorDict instance.

Examples

The example needs a Redis server on `localhost:6379`, and the
last line a Dragonfly server on the host `dragonfly-host`, so it
is not run when the documentation is tested.

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"obs": torch.randn(10, 84)}, [10])
>>> store_td = td.to_store(host="localhost") 
>>> store_td["obs"].shape 
torch.Size([10, 84])
>>>
>>> # Using Dragonfly instead of Redis
>>> store_td = td.to_store(backend="dragonfly", host="dragonfly-host")
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).