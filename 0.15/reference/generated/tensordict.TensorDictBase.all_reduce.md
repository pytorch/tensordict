# TensorDictBase.all_reduce

TensorDictBase.all_reduce(*op=None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *async_op: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *bucket_bytes: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [None](https://docs.python.org/3/builtins/constants.html#None)

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
- **bucket_bytes** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if given, packs the leaves of each
device and dtype into buckets of at most `bucket_bytes` bytes
(a larger leaf is a bucket of its own), and runs one
reduction per bucket instead of one per leaf, which is faster
with many small leaves. The results are copied back into the
leaves once each reduction completes, and with
`async_op=True`, the list holds one entry per bucket: a
`Work`, or a [`torch.futures.Future`](https://docs.pytorch.org/docs/stable/futures.html#torch.futures.Future) that also copies
the results back. Packing copies the leaves, so a
`bucket_bytes` below the size of the large leaves keeps
them out of the buckets: with `gloo` on CPU, packing leaves
of 4 MB made the call several times slower. Non-contiguous
and sparse leaves are never packed: each is reduced on its
own. Every rank must hold the same keys, shapes and dtypes,
as the buckets are built from the local leaves; if the
layouts differ, the call can hang. Defaults to `None`, one
reduction per leaf.

Returns:

With `async_op=True`, a list of `Work` objects (with
`bucket_bytes`, `Work` or `Future` objects, one per bucket).
Otherwise, `None`.

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