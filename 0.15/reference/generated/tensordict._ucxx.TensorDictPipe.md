# TensorDictPipe

*class*tensordict._ucxx.TensorDictPipe(*endpoint*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *listener: [Any](https://docs.python.org/3/library/typing.html#typing.Any) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A persistent point-to-point channel for streaming TensorDicts over UCXX.

Warning

`TensorDictPipe` is experimental: its API can change in a minor
release without a deprecation cycle.

It needs the optional `ucxx` package. Import it from
`tensordict._ucxx`: there is no public import path.

Implements a two-phase protocol:

- Handshake (first send): transmits consolidation metadata and data.
The receiver allocates a buffer and builds a TensorDict whose leaves
are views into the consolidated storage.
- Steady state (subsequent sends with same schema): transmits only
the raw storage buffer. The receiver overwrites its pre-allocated
buffer in place, without a new allocation and without metadata parsing.

Warning

With `consolidated=True` (the default), every receive with the same
schema writes into the same TensorDict, and returns it. The next
receive overwrites the values of the previous one. Call `clone()`
on a received TensorDict to keep it.

Warning

`async for td in pipe` stops at the first error, without raising
it. To see the error, call `arecv()` or `recv()` in a loop.

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

Close this pipe (async).

*async*arecv(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Receive a TensorDict from this pipe (async).

On the first call, receives into `td` if its consolidated layout
matches, otherwise builds a new TensorDict from the received metadata.
On subsequent calls with the same schema, overwrites the existing
storage in-place (zero allocation).

Parameters:

**td** - if provided and consolidated, receives into its storage
when its layout matches the received data. Otherwise the
pipe returns a new TensorDict, or on a message with the same
schema its own TensorDict from the previous receive, so use
the return value. If `None`, uses the internally cached
TensorDict from the previous receive (if any). Not used when
the pipe has `consolidated=False`.

Keyword Arguments:

**device** - device for storage allocation on first receive.

Returns:

The received TensorDict.

*async*asend(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Send a TensorDict through this pipe (async).

On the first call (or when the schema changes), sends metadata + data.
On subsequent calls with the same schema, sends only the raw buffer.

Parameters:

**td** - the TensorDict to send.

close() → [None](https://docs.python.org/3/builtins/constants.html#None)

Close this pipe (sync).

*async classmethod*connect(*host: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*, *port: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → TensorDictPipe

Connect to a remote TensorDictPipe listener.

Parameters:

- **host** - hostname or IP address of the remote listener.
- **port** - port number.

Keyword Arguments:

**consolidated** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - whether to use consolidated transfers.
Defaults to `True`.

Returns:

A connected TensorDictPipe.

*async classmethod*listen(*port: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → TensorDictPipe

Listen for an incoming connection and return a pipe to the first client.

Parameters:

**port** - port number to listen on.

Keyword Arguments:

**consolidated** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - whether to use consolidated transfers.
Defaults to `True`.

Returns:

A connected TensorDictPipe for the first accepted client.

recv(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Receive a TensorDict from this pipe (sync).

Blocking wrapper around `arecv()`.

Parameters:

**td** - if provided, receives into this TensorDict in-place when
it is consolidated and its layout matches the received data.
Otherwise the pipe returns a new TensorDict or its own
TensorDict from the previous receive, so use the return
value.

Keyword Arguments:

**device** - device for storage allocation on first receive.

Returns:

The received TensorDict.

send(*td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Send a TensorDict through this pipe (sync).

Blocking wrapper around `asend()`.

Parameters:

**td** - the TensorDict to send.