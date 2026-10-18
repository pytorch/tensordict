# TensorDictBase.var

TensorDictBase.var(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['feature'] = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = NO_DEFAULT*, ***, *correction: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *key_transform: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)], [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the variance of every leaf, over all its elements or along batch dimensions.

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

- **correction** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the difference between the sample size and
the degrees of freedom. Defaults to `1`, Bessel's correction.
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
`lambda key: f"var_{key}"`. Ignored when `reduce=True`. Defaults
to `None`.

Returns:

A new tensordict, or a tensor if `reduce=True`.

Note

With `dim="feature"` and the default `correction`, a leaf
without feature dimensions gives NaN, and torch warns that the degrees
of freedom are `<= 0`.

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
>>> td.var(dim=0).batch_size
torch.Size([4])
>>> td.var(dim=0)["b", "c"].shape
torch.Size([4, 5, 6])
>>> td.var()["a"].shape
torch.Size([])
>>> td.var(dim="feature")["a"].shape
torch.Size([3, 4])
>>> td.var(reduce=True)
tensor(1.0535)
>>> list(td.var(key_transform=lambda key: f"var_{key}").keys(True, True))
['var_a', ('var_b', 'var_c')]
```

With `reduce=True` and an int `dim`, the leaves are concatenated
along `dim` before the reduction:

```
>>> td = TensorDict(a=torch.ones(3, 4), b=torch.ones(3, 4), batch_size=(3, 4))
>>> td.var(dim=0, reduce=True)
tensor([0., 0., 0., 0.])
>>> td.var(dim="feature", reduce=True)
tensor([[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).