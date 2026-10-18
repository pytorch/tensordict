# TensorDictBase.data

*property*TensorDictBase.data*: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

the same storage, without autograd.

The leaves of the result share storage with the leaves of the
tensordict, and do not require gradients. An in-place change to the
result changes the tensordict.

Assigning `td.data = other` rebinds the `.data` of each leaf of
the tensordict to the matching leaf of `other`, as `torch.Tensor.data`
does. The leaves then share storage with `other`: memory-mapped or
shared-memory leaves stop writing to their file or shared memory.
`other` and each of its nested tensordicts must have the batch size
of the matching tensordict, and each key of `other` must be in the
tensordict.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(x=torch.zeros(3, requires_grad=True), batch_size=[3])
>>> data = td.data
>>> data["x"].requires_grad, data["x"].data_ptr() == td["x"].data_ptr()
(False, True)
```

Type:

A tensordict of the `.data` of every leaf

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).