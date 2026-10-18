# TensorDictBase.all_reduce

TensorDictBase.all_reduce(*op=None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *async_op: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Reduces each leaf over all the ranks of the group, in place, so that every rank holds the result.

Every rank of the group must call it, with a tensordict of the same
structure. Each leaf is reduced with [`torch.distributed.all_reduce()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.all_reduce),
for example to sum or average gradients. See
[Collectives](../../distributed.html#distributed-collectives).

Parameters:

**op** ([*ReduceOp*](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)*,**optional*) - the reduction. Defaults to
`None`, which is `torch.distributed.ReduceOp.SUM`.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.
- **async_op** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns the list of `Work`
objects of the reductions without waiting for them: call `wait()`
on each one before you use the tensordict. Defaults to `False`.

Returns:

With `async_op=True`, a list of `Work` objects. Otherwise, `None`.

Examples

This example runs in two processes: add it to the
[runner script of the distributed guide](../../distributed.html#distributed) and
pass it to `run`.

```
def all_reduce(rank, world_size):
 td = TensorDict(a=torch.ones(3) * rank, batch_size=[3])
 td.all_reduce(op=dist.ReduceOp.SUM)
 print(rank, td["a"]) # with 2 ranks: tensor([1., 1., 1.]) on every rank
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).