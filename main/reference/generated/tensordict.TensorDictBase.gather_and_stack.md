# TensorDictBase.gather_and_stack

TensorDictBase.gather_and_stack(*dst: [int](https://docs.python.org/3/builtins/functions.html#int)*, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [None](https://docs.python.org/3/builtins/constants.html#None)

Stacks the tensordicts of the other ranks into the tensordict of rank `dst`, in place.

Every rank of the group must call it. On rank `dst`, the tensordict must
have one row per other rank: its first batch dimension is
`world_size - 1`, and its own data is not in the result. The
tensordicts are sent with [`torch.distributed.gather_object()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.gather_object), which
pickles them: use it only with ranks you trust. See
[Collectives](../../distributed.html#distributed-collectives).

Parameters:

- **dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the rank that receives the stack.
- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.

Returns:

`self` on rank `dst`, `None` on the other ranks.

Examples

This example runs in two processes: add it to the
[runner script of the distributed guide](../../distributed.html#distributed) and
pass it to `run`.

```
def gather_and_stack(rank, world_size):
 if rank == 0:
 out = TensorDict(a=torch.zeros(world_size - 1, 3), batch_size=[world_size - 1])
 out.gather_and_stack(dst=0)
 print(out["a"][:, 0]) # with 2 ranks: tensor([1.])
 else:
 td = TensorDict(a=torch.full((3,), float(rank)), batch_size=[3])
 td.gather_and_stack(dst=0)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).