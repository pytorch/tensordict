# TensorDictBase.auto_batch_size

TensorDictBase.auto_batch_size_(*batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *keep_compliant_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Sets the maximum batch-size for the tensordict, up to an optional batch_dims.

Parameters:

- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if provided, the batch-size will be at
most `batch_dims` long.
- **keep_compliant_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if True, a sub-tensordict with a compliant
size will not see its shape be changed in case batch_dims is passed.
If False, all contained tensordicts will have a batch_dims that matches
batch_dims.
Defaults to False.

Returns:

self

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> td = TensorDict({"a": torch.randn(3, 4, 5), "b": {"c": torch.randn(3, 4, 6)}}, batch_size=[])
>>> _ = td.auto_batch_size_()
>>> print(td.batch_size)
torch.Size([3, 4])
>>> _ = td.auto_batch_size_(batch_dims=1)
>>> print(td.batch_size)
torch.Size([3])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).