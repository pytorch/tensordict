# TensorDictBase.to_store

TensorDictBase.to_store(***, *backend: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'redis'*, *host: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'localhost'*, *port: [int](https://docs.python.org/3/builtins/functions.html#int) = 6379*, *db: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *unix_socket_path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'tensordict'*, *device=None*, ***kwargs*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Uploads the tensordict to a key-value server and returns a tensordict backed by it.

The entries are written to a Redis or Dragonfly server, under the key
namespace `prefix`. The result reads and writes the server. A lazy
stack is uploaded in chunks, without building the dense stack in
memory, and gives a [`LazyStackedTensorDictStore`](tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore).
Requires the `redis` Python package and a running server. See
[Large datasets and storage backends](../../storage.html#storage).

Keyword Arguments:

- **backend** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the server type: `"redis"` or
`"dragonfly"`. Defaults to `"redis"`.
- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the server hostname. Defaults to
`"localhost"`.
- **port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the server port. Defaults to `6379`.
- **db** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the database number. Defaults to `0`.
- **unix_socket_path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the path of a Unix domain socket
to connect through instead of `host` and `port`. Defaults
to `None`.
- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the key namespace of the data on the
server. Defaults to `"tensordict"`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
tensors that the result reads. Defaults to `None`, which
uses the device of the tensordict.
- ****kwargs** - passed to the connection of the client.

Returns:

A [`TensorDictStore`](tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore), or a
[`LazyStackedTensorDictStore`](tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore) for a lazy
stack.

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