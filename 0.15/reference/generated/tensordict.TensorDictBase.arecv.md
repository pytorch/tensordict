# TensorDictBase.arecv

*async*TensorDictBase.arecv(*src: [TensorDictPipe](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Receives a tensordict from a UCXX pipe, into the tensordict if possible, asynchronously.

Awaits `src.arecv(self, device=device)`. If the tensordict is
consolidated with the layout of the received data, the data is written
into its storage. Otherwise, the pipe returns another tensordict: use the
return value. See [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe), which needs
the optional `ucxx` package.

Parameters:

**src** ([*TensorDictPipe*](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)) - the pipe to receive from.

Keyword Arguments:

**device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the storage
that the pipe allocates for a new structure. Defaults to `None`,
which uses the device of the sender with a consolidated pipe, and
the CPU otherwise.

Returns:

The received tensordict, `self` or another tensordict.

See also

[`recv()`](tensordict.TensorDictBase.recv.html#tensordict.TensorDictBase.recv), which also takes a pipe, and works without
`await`.

Examples

This example needs `ucxx` and a sender on the other end of the pipe:

```
async def consume(pipe):
 td = await TensorDict().arecv(pipe) # first receive: allocates
 for _ in range(9):
 td = await td.arecv(pipe) # same structure: writes into td
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).