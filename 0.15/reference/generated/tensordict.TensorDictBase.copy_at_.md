# TensorDictBase.copy_at

TensorDictBase.copy_at_(*tensordict: T*, *idx: [None](https://docs.python.org/3/builtins/constants.html#None) | [int](https://docs.python.org/3/builtins/functions.html#int) | [slice](https://docs.python.org/3/builtins/functions.html#slice) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), ...]*, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *fast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Copies the tensors of `tensordict` into the existing entries at `idx`, and returns `self`.

`copy_at_` is a faster [`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_) for the hot paths where
every key of `tensordict` is a tensor entry of the tensordict. With
`fast=True`, a `tensordict` with only non-tensor entries raises a
`RuntimeError`. If it also holds tensors, its non-tensor entries are
skipped, not copied, with a `FutureWarning`: starting with
TensorDict 0.17, the default will change to raising the same
`RuntimeError`. Use `fast=False` or [`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_) to write
them.

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the entries to copy. Their leading
dimensions are those of `td[idx]`.
- **idx** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*slice*](https://docs.python.org/3/builtins/functions.html#slice)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) - the index to write at,
along the batch dimensions.
- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.

Keyword Arguments:

**fast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

if `True`, uses only the tensor-only copy,
and raises a `RuntimeError` if the source, the destination or
the index does not allow it (for example a dictionary source, a
tensordict that is not a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), a
tensor index, a value that needs broadcasting, or a key missing
from the tensordict). If `False`, calls
[`update_at_()`](tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_). Defaults to `True`.

Changed in version 0.15: `fast=None`, deprecated in 0.14, is no longer accepted.

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.zeros(3, 2), batch_size=[3])
>>> _ = td.copy_at_(TensorDict(x=torch.ones(2), batch_size=[]), 1)
>>> td["x"][:, 0]
tensor([0., 1., 0.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).