# TensorDictBase.from_remote_init

*classmethod*TensorDictBase.from_remote_init(*src: [int](https://docs.python.org/3/builtins/functions.html#int)*, *group: 'ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *use_broadcast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Creates a new tensordict instance initialized from remotely sent metadata.

This class method receives consolidated metadata and a single storage buffer
sent by [`init_remote()`](tensordict.TensorDictBase.init_remote.html#tensordict.TensorDictBase.init_remote), then reconstructs the full tensordict.

Two transport modes are available (must match the sender's choice):

- **Point-to-point** (default): uses `recv_object_list` +
`dist.recv`. Only sender and receiver participate.
- **Broadcast** (`use_broadcast=True`): delegates to
[`broadcast()`](tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast). All ranks must participate.

Parameters:

- **src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The rank of the source process that sent the metadata.
- **group** (*"ProcessGroup"**,**optional*) - The process group to use for communication. Defaults to None.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device to use for tensor operations. Defaults to None.
- **use_broadcast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - If `True`, use [`broadcast()`](tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast) instead
of point-to-point recv. Must match the sender's setting.
Defaults to `False`.

Returns:

A new tensordict instance initialized with the received metadata and content.

Return type:

[*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)

See also

The sending process should have called ~.init_remote to send the metadata and content.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).