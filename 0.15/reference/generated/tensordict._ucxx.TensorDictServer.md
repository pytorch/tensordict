# TensorDictServer

*class*tensordict._ucxx.TensorDictServer(*port: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A listener that accepts several clients, and gives a [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe) for each.

Warning

`TensorDictServer` is experimental: its API can change in a minor
release without a deprecation cycle.

It needs the optional `ucxx` package. Import it from
`tensordict._ucxx`: there is no public import path. Get the pipes with
`serve()`, or with `async for pipe in server`. It starts
listening when you do either. The protocol does not authenticate the
clients: use it only on a trusted network. A server is an async context
manager, which closes it at the end of the block.

Parameters:

**port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the port to listen on.

Keyword Arguments:

**consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `consolidated` argument of the
pipes (see [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)). The clients must pass the
same value to [`TensorDictPipe.connect()`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe.connect). Defaults to
`True`.

Raises:

[**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) - if `ucxx` is not installed.

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

Closes the server, as `close()` does, in a coroutine.

close() → [None](https://docs.python.org/3/builtins/constants.html#None)

Closes the server: it stops accepting connections, and `serve()` returns.

The pipes of the clients stay open.

*async*serve(*on_connect: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[TensorDictPipe](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)], [Awaitable](https://docs.python.org/3/library/typing.html#typing.Awaitable)[[None](https://docs.python.org/3/builtins/constants.html#None)]]*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Accepts connections, and runs `on_connect` on the pipe of each new client.

Runs until `close()` is called. Each `on_connect` call runs as
its own task, so that the clients are served at the same time. `async
for pipe in server` is the other way to get the pipes.

Parameters:

**on_connect** (*Callable*) - a coroutine function that takes the
[`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe) of a client.