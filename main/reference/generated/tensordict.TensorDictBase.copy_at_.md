# TensorDictBase.copy_at

TensorDictBase.copy_at_(*tensordict: T*, *idx: [None](https://docs.python.org/3/builtins/constants.html#None) | [int](https://docs.python.org/3/builtins/functions.html#int) | [slice](https://docs.python.org/3/builtins/functions.html#slice) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), ...]*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *fast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Copies values from `tensordict` into `self` at the specified index.

`copy_at_` is an explicit copy-oriented variant of
[`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_). Unlike `update_at_`,
it may use optimized tensor-only copy paths and is intended for hot
paths where the source and destination structures are known to match.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - input data to be copied in `self`.
- **idx** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**iterable**,*[*slice*](https://docs.python.org/3/builtins/functions.html#slice)) - index of the tensordict
where the copy should occur.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and this copy is between
different devices, the copy may occur asynchronously with respect
to the host.

Keyword Arguments:

**fast** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - 

controls whether `copy_at_` may
fall back to [`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_).
If `True`, only the optimized tensor-only path is used and a
`RuntimeError` is raised when the fast path is not available.
If `False`, this method delegates directly to `update_at_`.
Defaults to `True`.

Changed in version 0.15: `fast=None`, deprecated in 0.14, is no longer accepted.

Returns:

self

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).