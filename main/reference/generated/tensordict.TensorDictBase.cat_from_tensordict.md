# TensorDictBase.cat_from_tensordict

TensorDictBase.cat_from_tensordict(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *sorted: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Concatenates all the leaves of the tensordict into one tensor, with [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat).

Parameters:

**dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension to concatenate along, counted
in the dimensions of the leaves. Defaults to `0`.

Keyword Arguments:

- **sorted** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the order of the
leaves. `True` sorts the keys alphabetically, and `False`
or `None` keeps the order of insertion. A list of keys gives
the order, and must hold every leaf key, or a
`RuntimeError` is raised. Defaults to `None`.
- **out** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**optional*) - a tensor to write the result into.
Defaults to `None`.

Returns:

The concatenated tensor.

See also

[`pack()`](tensordict.TensorDictBase.pack.html#tensordict.TensorDictBase.pack), which flattens leaves of any shape into one
tensor and returns the layout to rebuild the tensordict.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(2, 1), b=torch.ones(2, 2), batch_size=[2])
>>> td.cat_from_tensordict(dim=1)
tensor([[0., 1., 1.],
 [0., 1., 1.]])
>>> td.cat_from_tensordict(dim=1, sorted=["b", "a"])
tensor([[1., 1., 0.],
 [1., 1., 0.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).