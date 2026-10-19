# TensorDictBase.init_remote

TensorDictBase.init_remote(*dst: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group: 'ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *use_broadcast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *_tensorclass_type: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Sends the structure and the content of the tensordict to a rank that does not know them.

The receiver calls [`from_remote_init()`](tensordict.TensorDictBase.from_remote_init.html#tensordict.TensorDictBase.from_remote_init), which builds a new
consolidated tensordict. The tensordict is consolidated (see
[`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)), then its metadata is sent with
[`torch.distributed.send_object_list()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.send_object_list), which pickles it, and its
storage as one tensor. Send the next batches with
`send(consolidated=True)`, into the tensordict that
[`from_remote_init()`](tensordict.TensorDictBase.from_remote_init.html#tensordict.TensorDictBase.from_remote_init) returned. See [Distributed tensordicts](../../distributed.html#distributed).

With `use_broadcast=True`, it calls [`broadcast()`](tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast) instead, and
every rank of the group must take part: the others call
`from_remote_init(src=..., use_broadcast=True)`.

Parameters:

- **dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the rank of the receiver. Required unless
`use_broadcast=True`, which ignores it. Defaults to `None`.
- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - point-to-point, the device
that the metadata goes through, as in
[`torch.distributed.send_object_list()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.send_object_list) (with `nccl`, a CUDA
device). With `use_broadcast=True`, the device of the
consolidated storage, as in [`broadcast()`](tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast). Defaults to
`None`.
- **use_broadcast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, broadcasts the tensordict
to every rank of the group. Defaults to `False`.

Examples

This example runs in two processes: add it to the
[runner script of the distributed guide](../../distributed.html#distributed) and
pass it to `run`.

```
def cold_start(rank, world_size):
 if rank == 0:
 td = TensorDict(obs=torch.ones(2, 84), reward=torch.ones(2, 1), batch_size=[2])
 td.init_remote(dst=1)
 td["obs"].add_(1)
 td.send(dst=1, consolidated=True)
 elif rank == 1:
 td = TensorDict.from_remote_init(src=0)
 print(td.batch_size, td.is_consolidated()) # torch.Size([2]) True
 td.recv(src=0, consolidated=True)
 print(td["obs"][0, 0]) # tensor(2.)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).