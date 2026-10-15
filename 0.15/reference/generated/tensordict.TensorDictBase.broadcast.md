# TensorDictBase.broadcast

TensorDictBase.broadcast(*src: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_tensorclass_type: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Broadcasts a tensordict from `src` to all ranks.

Uses consolidated transport: the source rank consolidates the tensordict
and broadcasts metadata (via `broadcast_object_list`) followed by the
contiguous storage tensor (via `dist.broadcast`). Receiving ranks
allocate a buffer and reconstruct the full tensordict.

Parameters:

**src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The rank of the source process.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - The process group
to use. Defaults to `None` (default group).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - The device on which to
allocate the receive buffer. If `None`, the device is
inferred from the source's consolidated storage (transmitted
via metadata). Useful when the backend is `nccl` and
buffers must live on CUDA. Defaults to `None`.

Returns:

the broadcast tensordict on all ranks (consolidated).

Return type:

[*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).