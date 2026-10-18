# TensorDictBase.param_count

TensorDictBase.param_count(***, *count_duplicates: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [int](https://docs.python.org/3/builtins/functions.html#int)

Returns the total number of elements of the tensor leaves.

It sums [`numel()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.numel.html#torch.Tensor.numel) over the tensor leaves, nested
tensordicts included. Non-tensor data is not counted.

Keyword Arguments:

**count_duplicates** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a tensor stored
under several keys is counted once per key. If `False`, it
is counted once. Two different views of one tensor are counted
twice in both cases. Defaults to `True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> x = torch.zeros(6)
>>> td = TensorDict(a=x, b=x, c=torch.zeros(2, 3), s="text")
>>> td.param_count()
18
>>> td.param_count(count_duplicates=False)
12
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).