# TensorDictPipe

*class*tensordict._ucxx.TensorDictPipe(*endpoint*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *listener: [Any](https://docs.python.org/3/library/typing.html#typing.Any) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A channel between two processes that streams tensordicts over UCXX.

Warning

`TensorDictPipe` is experimental: its API can change in a minor
release without a deprecation cycle.

It needs the optional `ucxx` package. Import it from
`tensordict._ucxx`: there is no public import path. Create a pipe with
`connect()` on one side and `listen()` on the other, rather than
with the constructor. The protocol does not authenticate the peer, which
controls the size of the allocations: use it only on a trusted network.
See [Distributed tensordicts](../../distributed.html#distributed).

With `consolidated=True`, a pipe sends a tensordict in two phases:

- The first message, and each message with a new structure (keys,
shapes or dtypes), carries the metadata and then the consolidated
storage (see [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)). The
receiver allocates one buffer, and builds a tensordict whose leaves
are views into it.
- The next messages carry the storage only. The receiver writes it into
the buffer of the first message, without allocating memory or parsing
metadata.

With `consolidated=False`, each message carries a JSON header, with the
batch size and the non-tensor entries, and then the tensor leaves of the
tensordict, nested ones included, one by one. The receiver builds a new
tensordict each time, with the batch size of the sent one; its nested
tensordicts get the same batch size. A non-tensor entry must be a
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) or [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack)
whose values are `str`, `int`, `float`, `bool`, `None`, or lists
or dicts (with `str` keys) of them: a send of any other non-tensor entry
raises a `ValueError`. A stack comes back with the same batch size and
stack dimension.

A pipe is a context manager and an async context manager, which close
it at the end of the block, and an async iterator: `async for td in
pipe` receives until the pipe is closed.

Warning

With `consolidated=True`, every receive with the same structure
writes into the same tensordict, and returns it. The next receive
overwrites the values of the previous one. Call `clone()` on a
received tensordict to keep it.

Warning

`async for td in pipe` stops at the first error, without raising
it. To see the error, call `arecv()` or `recv()` in a loop.

Parameters:

**endpoint** (*ucxx.Endpoint*) - the UCXX endpoint of the connection.

Keyword Arguments:

- **consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sends and receives
consolidated tensordicts, in the two phases above. Both ends must
pass the same value. Defaults to `True`.
- **listener** (*ucxx.Listener**,**optional*) - a listener that the pipe closes
when it closes, as `listen()` passes. Defaults to `None`.

Examples

The sender and the receiver run in two processes, on the same or
on two machines. This example is not tested: it needs `ucxx` and
two processes.

```
# receiver, on host "node-b"
import asyncio
from tensordict._ucxx import TensorDictPipe

async def main():
 pipe = await TensorDictPipe.listen(13337)
 td = await pipe.arecv() # first receive: allocates the buffer
 async for td in pipe: # same schema: overwrites td in place
 print(td["obs"].sum())

asyncio.run(main())
```

```
# sender
import asyncio
import torch
from tensordict import TensorDict
from tensordict._ucxx import TensorDictPipe

async def main():
 pipe = await TensorDictPipe.connect("node-b", 13337)
 for _ in range(10):
 await pipe.asend(TensorDict(obs=torch.randn(64, 84)))
 await pipe.aclose()

asyncio.run(main())
```

*async*aclose() → [None](https://docs.python.org/3/builtins/constants.html#None)

Closes the pipe, its endpoint and its listener (if any), asynchronously.

Closing a closed pipe does nothing.

*async*arecv(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Receives a tensordict from the pipe, asynchronously.

With `consolidated=True`, the first message, and each message with a new
structure, carries the metadata: the pipe receives into `td` if its
consolidated layout matches, and builds a new tensordict otherwise. The
next messages carry the storage only, and the pipe writes it into the
same tensordict, without allocating memory. So each receive can overwrite
the values of the previous one: clone a tensordict to keep it.

With `consolidated=False`, each call returns a new tensordict with the
batch size of the sent one (an empty batch size from an older sender).

Parameters:

**td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a consolidated tensordict to receive
into, if its layout matches the data. Otherwise the pipe returns a
new tensordict, or the tensordict of its previous receive, so use the
return value. Not used with `consolidated=False`. Defaults to
`None`, which receives into the tensordict of the previous
receive.

Keyword Arguments:

**device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the storage
that the pipe allocates for a new structure. Defaults to `None`,
which uses the device of the sender with `consolidated=True`, and
the CPU otherwise.

Returns:

The received tensordict.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the pipe is closed, or if the sender and the receiver
 are out of sync.

*async*asend(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Sends a tensordict through the pipe, asynchronously.

With `consolidated=True`, the first call, and each call where the
structure of the tensordict changes, sends the metadata and then the
storage. The other calls send the storage only. A tensordict that is not
consolidated is consolidated on each call, which copies every leaf:
consolidate it once and refresh it with
[`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_) to avoid the copy.

Parameters:

**td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to send.

Raises:

- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the pipe is closed.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - with `consolidated=False`, if `td` holds a non-tensor
 entry that the pipe cannot send (see the class description).

close() → [None](https://docs.python.org/3/builtins/constants.html#None)

Closes the pipe, and waits until it is closed.

Runs `aclose()` in a new event loop. In a coroutine, `await
pipe.aclose()` instead. A pipe also closes at the end of a `with` or
`async with` block, and when it is garbage collected.

*property*closed*: [bool](https://docs.python.org/3/builtins/functions.html#bool)*

Whether the pipe is closed.

*async classmethod*connect(*host: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *port: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → TensorDictPipe

Connects to a pipe that listens on `host` and `port`, and returns the pipe.

The other process calls `listen()` with the same port.

Parameters:

- **host** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - the host name or IP address of the listener.
- **port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the port of the listener.

Keyword Arguments:

**consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sends and receives
consolidated tensordicts (see the class description). Both ends must
pass the same value. Defaults to `True`.

Returns:

A connected `TensorDictPipe`.

Raises:

[**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) - if `ucxx` is not installed.

*async classmethod*listen(*port: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → TensorDictPipe

Listens on `port`, and returns a pipe to the first client that connects.

Waits until a client calls `connect()`. The listener stops when the
pipe is closed. To accept several clients, use [`TensorDictServer`](tensordict._ucxx.TensorDictServer.html#tensordict._ucxx.TensorDictServer).

Parameters:

**port** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the port to listen on.

Keyword Arguments:

**consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sends and receives
consolidated tensordicts (see the class description). Both ends must
pass the same value. Defaults to `True`.

Returns:

A connected `TensorDictPipe`.

Raises:

[**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) - if `ucxx` is not installed.

recv(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Receives a tensordict from the pipe, and waits until it is received.

Runs `arecv()` in a new event loop, with the same arguments. In a
coroutine, `await pipe.arecv()` instead: in a thread where an event loop
runs, `recv` raises a `RuntimeError`.

Parameters:

**td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a consolidated tensordict to receive
into, if its layout matches the data. See `arecv()`. Defaults to
`None`.

Keyword Arguments:

**device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the storage
that the pipe allocates for a new structure. See `arecv()`.
Defaults to `None`.

Returns:

The received tensordict.

send(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Sends a tensordict through the pipe, and waits until it is sent.

Runs `asend()` in a new event loop. In a coroutine, `await
pipe.asend(td)` instead: in a thread where an event loop runs, `send`
raises a `RuntimeError`.

Parameters:

**td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to send.