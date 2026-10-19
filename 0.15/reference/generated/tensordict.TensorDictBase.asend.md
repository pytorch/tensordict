# TensorDictBase.asend

*async*TensorDictBase.asend(*dst: [TensorDictPipe](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Sends the tensordict through a UCXX pipe, asynchronously.

Awaits `dst.asend(self)`. See [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe),
which needs the optional `ucxx` package.

Parameters:

**dst** ([*TensorDictPipe*](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)) - the pipe to send through.

See also

[`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send), which also takes a pipe, and works without
`await`.

Examples

This example needs `ucxx` and a receiver on the other end of the
pipe:

```
async def produce(pipe):
 for _ in range(10):
 await TensorDict(obs=torch.randn(64, 84)).asend(pipe)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).