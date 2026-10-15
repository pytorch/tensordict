# TensorDictBase.scatter

TensorDictBase.scatter(*src: [int](https://docs.python.org/3/builtins/functions.html#int)*, *tensordicts: [list](https://docs.python.org/3/builtins/stdtypes.html#list) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Scatters a list of tensordicts from `src` to all ranks.

On the source rank, `tensordicts` must be a list of tensordicts with
one element per rank. Each is consolidated, and its metadata + storage
are scattered. Non-source ranks receive and reconstruct their tensordict.

Parameters:

- **src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - The rank of the source process.
- **tensordicts** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*[*[*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*]**|**None*) - On the source rank, a list
of tensordicts to scatter (one per rank). Ignored on other ranks.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - The process group
to use. Defaults to `None` (default group).
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - The device on which to
allocate the receive buffer. If `None`, the device is
inferred from the source's consolidated storage (transmitted
via metadata). Defaults to `None`.

Returns:

The tensordict assigned to this rank.

Return type:

[*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).