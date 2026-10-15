# TensorDictBase.arecv

*async*TensorDictBase.arecv(*src: [TensorDictPipe](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Receives content into this tensordict from a UCXX pipe (async).

Parameters:

**src** ([*TensorDictPipe*](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)) - the pipe to receive from.

Keyword Arguments:

**device** - device for storage allocation on first receive.

Returns:

The received TensorDict (may be `self` for in-place updates).

See also

[`recv()`](tensordict.TensorDictBase.recv.html#tensordict.TensorDictBase.recv) for the synchronous variant and
torch.distributed-based transport.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).