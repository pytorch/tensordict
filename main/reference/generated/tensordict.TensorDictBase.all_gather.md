# TensorDictBase.all_gather

TensorDictBase.all_gather(***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [list](https://docs.python.org/3/builtins/stdtypes.html#list)

Returns a list with the tensordict of each rank of the group, on every rank.

Every rank of the group must call it. The tensordicts can have different
structures. Each rank consolidates its tensordict (see
[`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)), the metadata is gathered with
[`torch.distributed.all_gather_object()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather_object), which pickles it, and the
storages with [`torch.distributed.all_gather()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_gather). Use it only with ranks
you trust. See [Collectives](../../distributed.html#distributed-collectives).

Keyword Arguments:

**group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.

Returns:

A list of new consolidated tensordicts, one per rank, in the order of
the ranks.

Examples

This example runs in two processes: add it to the
[runner script of the distributed guide](../../distributed.html#distributed) and
pass it to `run`.

```
def all_gather(rank, world_size):
 local = TensorDict(a=torch.full((3,), float(rank)), batch_size=[3])
 tds = local.all_gather()
 print(rank, [td["a"][0].item() for td in tds]) # [0.0, 1.0] on every rank
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).