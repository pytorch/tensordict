# TensorDictBase.reduce

TensorDictBase.reduce(*dst*, *op=None*, *async_op=False*, *return_premature=False*, *group=None*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Reduces the tensordict across all machines.

Only the process with `rank` dst is going to receive the final result.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).