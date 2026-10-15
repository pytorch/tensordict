# TensorDictBase.empty

TensorDictBase.empty(*recurse=False*, ***, *batch_size=None*, *device=NO_DEFAULT*, *names=None*) → Self

Returns a new, empty tensordict with the same device and batch size.

The result is not locked, even if `self` is.

Parameters:

**recurse** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the entire structure of the
`TensorDict` will be reproduced without content: the result
has the sub-tensordicts of `self`, without their leaves.
Otherwise, only the root will be duplicated.
Defaults to `False`.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - a new batch-size for the tensordict.
With `recurse=True`, a sub-tensordict whose batch size is the
start of the new one gets the new one, a sub-tensordict whose
batch size starts with the new one keeps it, and any other
sub-tensordict raises a `RuntimeError`. A
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) whose members took
the batch size before 0.15 (one stacked along dim 0, for
instance) keeps that result, the empty stack of its members,
with a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning), when a sub-tensordict does not
fit the new batch size or `names` has another length.
Starting with TensorDict 0.17, the default will change to
raising a `RuntimeError` or a `ValueError`, as
`TensorDict.empty` and the names setter do.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - a new device.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - dimension names, one per dim of the
new batch size. Where `empty()` keeps names that the names
setter would reject or drop (a list of another length, or a
name used twice), it emits a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning). Starting
with TensorDict 0.17, the default will change to applying them
as the names setter does, with or without `recurse`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3), "b": {"c": torch.ones(3)}}, [3])
>>> empty = td.empty()
>>> list(empty.keys()), empty.batch_size
([], torch.Size([3]))
>>> list(td.empty(recurse=True).keys(include_nested=True))
['b']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).