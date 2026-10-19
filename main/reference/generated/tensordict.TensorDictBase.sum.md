# TensorDictBase.sum

TensorDictBase.sum(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['feature'] = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = NO_DEFAULT*, ***, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *key_transform: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)], [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the sum of every leaf, over all its elements or along batch dimensions.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the batch dimensions to
reduce. A negative `dim` counts from the end of the batch
dimensions. `"feature"` reduces the feature dimensions of each
leaf instead, and the result keeps the batch size. Defaults to
`None`, which reduces every element of each leaf, and the result
has an empty batch size.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the reduced batch dimensions are
kept with size 1. Not allowed with `dim="feature"`. Defaults to
`False`.

Keyword Arguments:

- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - if given, each leaf is cast to this dtype
before the reduction, which helps to prevent overflows. Defaults to
`None`.
- **reduce** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, reduces all the leaves together
and returns a single tensor. Without `dim`, the result is a
scalar. With an int `dim`, the leaves are concatenated along
`dim` first, so they must have the same number of dimensions. With
`dim="feature"`, the result has the batch size as its shape. With
feature dimensions, a negative `dim` counts from the end of each
leaf. Starting with TensorDict 0.17, the default will change to
counting it from the end of the batch dimensions, as with
`reduce=False`. Defaults to `False`.
- **key_transform** (*Callable**,**optional*) - a function applied to the name of
every entry of the result, at every level of nesting, such as
`lambda key: f"sum_{key}"`. Ignored when `reduce=True`. Defaults
to `None`.

Returns:

A new tensordict, or a tensor if `reduce=True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> _ = torch.manual_seed(0)
>>> td = TensorDict(
... a=torch.randn(3, 4, 5),
... b=TensorDict(c=torch.randn(3, 4, 5, 6), batch_size=(3, 4, 5)),
... batch_size=(3, 4),
... )
>>> td.sum(dim=0).batch_size
torch.Size([4])
>>> td.sum(dim=0)["b", "c"].shape
torch.Size([4, 5, 6])
>>> td.sum()["a"].shape
torch.Size([])
>>> td.sum(dim="feature")["a"].shape
torch.Size([3, 4])
>>> td.sum(reduce=True)
tensor(6.5324)
>>> list(td.sum(key_transform=lambda key: f"sum_{key}").keys(True, True))
['sum_a', ('sum_b', 'sum_c')]
```

With `reduce=True` and an int `dim`, the leaves are concatenated
along `dim` before the reduction:

```
>>> td = TensorDict(a=torch.ones(3, 4), b=torch.ones(3, 4), batch_size=(3, 4))
>>> td.sum(dim=0, reduce=True)
tensor([6., 6., 6., 6.])
>>> td.sum(dim="feature", reduce=True)
tensor([[2., 2., 2., 2.],
 [2., 2., 2., 2.],
 [2., 2., 2., 2.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).