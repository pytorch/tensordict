# TensorDictBase.data_ptr

TensorDictBase.data_ptr(***, *storage: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns the data pointers of the leaves, as a tensordict of integers.

The result has the keys of the tensordict, an empty batch size and the
CPU device. Each leaf is replaced by its
[`data_ptr()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html#torch.Tensor.data_ptr), as a 0-d `int64` tensor. Compare two
results to check whether two tensordicts share memory. In a lazy stack,
each member is a nested entry named by its index (`"0"`, `"1"`,
...), which gives the data pointers of the leaves it really holds.

Keyword Arguments:

**storage** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns the data pointer of
the storage of each leaf
(`tensor.untyped_storage().data_ptr()`) instead, which a view
shares with its base. Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(2), b=torch.zeros(2), batch_size=[2])
>>> td0 = td.unsqueeze(0) # the leaves of td0 are views on the leaves of td
>>> assert (td0.data_ptr() == td.data_ptr()).all()
>>> td1 = td[1:] # views with an offset
>>> (td1.data_ptr() == td.data_ptr()).all()
False
>>> (td1.data_ptr(storage=True) == td.data_ptr(storage=True)).all()
True
>>> stack = TensorDict.lazy_stack([td, td])
>>> list(stack.data_ptr().keys(include_nested=True, leaves_only=True))
[('0', 'a'), ('0', 'b'), ('1', 'a'), ('1', 'b')]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).