# TensorDictBase.scatter

TensorDictBase.scatter(*src: [int](https://docs.python.org/3/builtins/functions.html#int)*, *tensordicts: [list](https://docs.python.org/3/builtins/stdtypes.html#list) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Scatters a list of tensordicts from `src` to all ranks.

Every rank of the process group calls it, in an initialized
[`torch.distributed`](https://docs.pytorch.org/docs/stable/distributed.html#module-torch.distributed) process group. On the source rank,
`tensordicts` holds one tensordict per rank; each is consolidated
(see [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)) and sent as one buffer. The other ranks
rebuild their tensordict from the buffer they receive. The tensordict
the method is called on is not used. See [Distributed tensordicts](../../distributed.html#distributed).

Parameters:

- **src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the rank of the source process.
- **tensordicts** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - on the source rank,
the tensordicts to scatter, one per rank. Ignored on the other
ranks. Defaults to `None`.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, which uses the default group.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - on the receiving ranks, the
device of the receive buffer. Defaults to `None`, which uses
the device of the consolidated storage of the source.

Returns:

The tensordict that this rank receives.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).