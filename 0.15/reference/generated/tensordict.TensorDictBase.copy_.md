# TensorDictBase.copy

TensorDictBase.copy_(*tensordict: T*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

See [`TensorDictBase.update_`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_).

The non-blocking argument will be ignored and is just present for
compatibility with `torch.Tensor.copy_()`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).