# TensorDictBase.asend

*async*TensorDictBase.asend(*dst: [TensorDictPipe](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Sends the content of a tensordict through a UCXX pipe (async).

Parameters:

**dst** ([*TensorDictPipe*](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)) - the pipe to send through.

See also

[`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send) for the synchronous variant and
torch.distributed-based transport.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).