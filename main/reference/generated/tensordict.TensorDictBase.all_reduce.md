# TensorDictBase.all_reduce

TensorDictBase.all_reduce(*op=None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *async_op: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [None](https://docs.python.org/3/builtins/constants.html#None)

All-reduces the tensordict across all ranks in-place.

Each leaf tensor is reduced individually via `dist.all_reduce`.
After the call, every rank holds the reduced values.

Parameters:

**op** (*dist.ReduceOp**,**optional*) - The reduce operation (e.g.
`ReduceOp.SUM`). Defaults to `ReduceOp.SUM`.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - The process group
to use. Defaults to `None` (default group).
- **async_op** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if `True`, returns a list of futures.
Defaults to `False`.

Returns:

None, or a list of futures if `async_op=True`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).