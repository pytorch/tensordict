# TensorDictBase.reduce

TensorDictBase.reduce(*dst*, *op=None*, *async_op=False*, *return_premature=False*, *group=None*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Reduces each leaf over all the ranks of the group, into the tensordict of rank `dst`.

Every rank of the group must call it, with a tensordict of the same
structure. Each leaf is reduced in place with
[`torch.distributed.reduce()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.reduce). After the call, rank `dst` holds the
result. On the other ranks, the values of the tensordict are not defined
(with `gloo`, they can hold partial results): clone the tensordict
before the call if you need them. A lazy stack reduces each stacked
leaf, like a dense tensordict of the same keys and shapes, and rank
`dst` writes the result back into its members: after the last
reduction, or with `async_op=True`, after each reduction completes.
That write raises if a member leaf cannot be written in place, such as
an expanded tensor, and the call waits for every reduction before it
raises. See [Collectives](../../distributed.html#distributed-collectives).

Parameters:

- **dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the rank that receives the result.
- **op** ([*ReduceOp*](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.ReduceOp)*,**optional*) - the reduction. Defaults to
`None`, which is `torch.distributed.ReduceOp.SUM`.
- **async_op** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the reductions of all the
leaves run at the same time, and the call waits for them unless
`return_premature=True`. Defaults to `False`.
- **return_premature** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - with `async_op=True`, if
`True`, returns the list of `Work` objects of the reductions
without waiting for them: call `wait()` on each one before you use
the tensordict, even if an earlier one raises. On rank `dst`,
the leaves of a lazy stack give a
[`torch.futures.Future`](https://docs.pytorch.org/docs/stable/futures.html#torch.futures.Future) instead, which also writes the
result back. Defaults to `False`.
- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.

Returns:

With `async_op=True` and `return_premature=True`, a list of
`Work` objects. Otherwise, `None`.

Examples

This example runs in two processes: add it to the
[runner script of the distributed guide](../../distributed.html#distributed) and
pass it to `run`.

```
def reduce(rank, world_size):
 td = TensorDict(a=torch.ones(3) * (rank + 1), batch_size=[3])
 td.reduce(dst=0, op=dist.ReduceOp.SUM)
 if rank == 0:
 print(td["a"]) # with 2 ranks: tensor([3., 3., 3.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).