# TensorDictServer

*class*tensordict._ucxx.TensorDictServer(*port: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A one-to-many listener that accepts connections and yields TensorDictPipe instances.

Warning

`TensorDictServer` is experimental: its API can change in a minor
release without a deprecation cycle.

Wraps `ucxx.create_listener` to provide a server that can handle
multiple clients, each getting their own [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe).
It needs the optional `ucxx` package. Import it from
`tensordict._ucxx`.

Examples

Each client connects with [`TensorDictPipe.connect()`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe.connect). This
example is not tested: it needs `ucxx` and client processes.

```
import asyncio
from tensordict._ucxx import TensorDictServer

async def handler(pipe):
 async for td in pipe:
 print(td["obs"].shape)

async def main():
 server = TensorDictServer(port=13337)
 await server.serve(handler) # runs until server.close()

asyncio.run(main())
```

*async*aclose() → [None](https://docs.python.org/3/builtins/constants.html#None)

Close the server (async).

close() → [None](https://docs.python.org/3/builtins/constants.html#None)

Close the server and stop accepting connections.

*async*serve(*on_connect: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[TensorDictPipe](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)], [Awaitable](https://docs.python.org/3/library/typing.html#typing.Awaitable)[[None](https://docs.python.org/3/builtins/constants.html#None)]]*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Accept connections and call `on_connect` for each new client.

This runs indefinitely until `close()` is called.

Parameters:

**on_connect** - an async callable that receives a [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)
for each new client connection.