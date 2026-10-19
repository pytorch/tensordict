# TensorDictBase.auto_batch_size

TensorDictBase.auto_batch_size_(*batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *keep_compliant_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Sets the batch size to the largest leading shape that all the entries share.

The nested tensordicts get their own largest batch size first. Nested
lazy stacks keep their batch size, and a lazy stack itself raises a
`RuntimeError` if its batch size would change. See
[Batch size, shapes, and indexing](../../concepts/batch_size.html#batch-size).

Parameters:

- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the maximum number of batch dimensions,
for the tensordict and for each nested tensordict. Defaults to
`None`, which sets no maximum.
- **keep_compliant_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and `batch_dims`
is passed, a nested tensordict that already has `batch_dims`
batch dimensions or more keeps its batch size. Defaults to
`False`.

Returns:

`self`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3, 4, 5), "b": {"c": torch.zeros(3, 4, 6)}}, batch_size=[])
>>> _ = td.auto_batch_size_()
>>> td.batch_size, td["b"].batch_size
(torch.Size([3, 4]), torch.Size([3, 4, 6]))
>>> _ = td.auto_batch_size_(batch_dims=1)
>>> td.batch_size, td["b"].batch_size
(torch.Size([3]), torch.Size([3]))
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).