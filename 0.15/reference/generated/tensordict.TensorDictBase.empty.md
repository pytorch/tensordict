# TensorDictBase.empty

TensorDictBase.empty(*recurse=False*, ***, *batch_size=None*, *device=NO_DEFAULT*, *names=None*) → Self

Returns a new tensordict of the same type, with no tensor leaves.

The result has the batch size, device and dimension names of the
tensordict, unless the keyword arguments change them. It is not locked,
even if the tensordict is.

Parameters:

**recurse** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the result has the nested
tensordicts and the non-tensor data of the tensordict, without
its tensor leaves. If `False`, the result has no entries.
Defaults to `False`.

Keyword Arguments:

- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result. Defaults to `None`, which keeps the batch size of
the tensordict.
With `recurse=True`, a nested tensordict whose batch size is the
start of the new one gets the new one, a nested tensordict whose
batch size starts with the new one keeps it, and any other
nested tensordict raises a `RuntimeError`. A
[`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) whose members took
the batch size before 0.15 (one stacked along dim 0, for
instance) keeps that result, the empty stack of its members,
with a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning), when a nested tensordict does not
fit the new batch size or `names` has another length.
Starting with TensorDict 0.17, the default will change to
raising a `RuntimeError` or a `ValueError`, as
`TensorDict.empty` and the names setter do.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. `None` gives a result with no device. By default, the
result has the device of the tensordict.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions
of the result, one per dimension. Defaults to `None`, which
keeps the names of the tensordict for the dimensions that
remain. Where `empty()` keeps names that the names
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
>>> td.empty(batch_size=[]).batch_size
torch.Size([])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).