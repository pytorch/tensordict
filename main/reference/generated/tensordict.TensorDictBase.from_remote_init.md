# TensorDictBase.from_remote_init

*classmethod*TensorDictBase.from_remote_init(*src: [int](https://docs.python.org/3/builtins/functions.html#int)*, *group: 'ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *use_broadcast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Receives a tensordict that [`init_remote()`](tensordict.TensorDictBase.init_remote.html#tensordict.TensorDictBase.init_remote) sends, and returns it.

The result is consolidated (see [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)): receive the next
batches into it with `recv(consolidated=True)`, which allocates no new
memory. A tensorclass arrives as the same tensorclass, if this process can
import its class. The metadata is unpickled: use it only with ranks you
trust. See [`init_remote()`](tensordict.TensorDictBase.init_remote.html#tensordict.TensorDictBase.init_remote) for an example, and [Distributed tensordicts](../../distributed.html#distributed).

Parameters:

- **src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the rank of the sender.
- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the result.
Point-to-point, it is also the device that the metadata goes
through, as in [`torch.distributed.recv_object_list()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.recv_object_list).
Defaults to `None`, which is the CPU point-to-point, and the
device of the sender's storage with `use_broadcast=True`.
- **use_broadcast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, receives with
[`broadcast()`](tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast). It must match the sender's value. Defaults to
`False`.

Returns:

A new consolidated tensordict, of the class that was sent.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).