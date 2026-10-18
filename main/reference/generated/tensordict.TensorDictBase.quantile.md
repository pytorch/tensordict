# TensorDictBase.quantile

TensorDictBase.quantile(*q: [float](https://docs.python.org/3/builtins/functions.html#float) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['feature'] = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = NO_DEFAULT*, ***, *interpolation: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'linear'*, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *key_transform: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)], [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the `q`-th quantile of every leaf, over all its elements or along a batch dimension.

Parameters:

- **q** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - the quantile, between 0 and 1. With a
1-dimensional tensor of `n` quantiles, every leaf of the result
gets a leading dimension of size `n`, and so does the batch size
when `dim` is given.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the batch dimension to reduce. A negative
`dim` counts from the end of the batch dimensions. `"feature"`
reduces the feature dimensions of each leaf instead, and the result
keeps the batch size. A tuple reduces only its first dimension;
passing one is deprecated and will be removed in TensorDict 0.17.
Defaults to `None`, which reduces every element of each leaf, and
the result has an empty batch size.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the reduced batch dimensions are
kept with size 1. Not allowed with `dim="feature"`. Defaults to
`False`.

Keyword Arguments:

- **interpolation** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the method used when the quantile lies
between two data points: `"linear"`, `"lower"`, `"higher"`,
`"midpoint"` or `"nearest"`. Defaults to `"linear"`.
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
`lambda key: f"median_{key}"`. Ignored when `reduce=True`.
Defaults to `None`.

Returns:

A new tensordict, or a tensor if `reduce=True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.arange(5.0), b=torch.arange(5.0) * 2, batch_size=[5])
>>> td.quantile(0.5)["b"]
tensor(4.)
>>> td.quantile(torch.tensor([0.25, 0.75]), dim=0)["a"]
tensor([1., 3.])
>>> td.quantile(0.5, reduce=True) # the median of the ten elements
tensor(2.5000)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).