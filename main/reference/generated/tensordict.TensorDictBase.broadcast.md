# TensorDictBase.broadcast

TensorDictBase.broadcast(*src: [int](https://docs.python.org/3/builtins/functions.html#int)*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *_tensorclass_type: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Sends the tensordict of rank `src` to every rank of the group, and returns it on each rank.

Every rank of the group must call it. The other ranks do not need to know
the structure: they can call it on an empty tensordict. Rank `src`
consolidates its tensordict (see [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)), and broadcasts the
metadata with [`torch.distributed.broadcast_object_list()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.broadcast_object_list), which
pickles it, then the storage as one tensor. The other ranks build a new
tensordict from them. Use it only with ranks you trust. See
[Collectives](../../distributed.html#distributed-collectives).

Parameters:

**src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the rank that sends its tensordict.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - on rank `src`, the
device of the consolidated storage, if the tensordict is not
consolidated yet. On the other ranks, the device of the result.
With `nccl`, pass a CUDA device. Defaults to `None`, which keeps
the device of the tensordict on rank `src`, and uses the same
device on the other ranks.

Returns:

The consolidated tensordict of rank `src`. On rank `src`, it is
the tensordict itself if it is consolidated already, and a
consolidated copy otherwise. On the other ranks, it is a new
tensordict.

Examples

This example runs in two processes: add it to the
[runner script of the distributed guide](../../distributed.html#distributed) and
pass it to `run`.

```
def broadcast(rank, world_size):
 td = TensorDict(a=torch.arange(3.0), batch_size=[3]) if rank == 0 else TensorDict()
 td = td.broadcast(src=0)
 print(rank, td["a"]) # tensor([0., 1., 2.]) on every rank
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).