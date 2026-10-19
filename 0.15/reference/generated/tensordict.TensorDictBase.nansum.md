# TensorDictBase.nansum

TensorDictBase.nansum(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['feature'] = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = NO_DEFAULT*, ***, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the sum of the non-NaN elements of every leaf.

Works as [`sum()`](tensordict.TensorDictBase.sum.html#tensordict.TensorDictBase.sum), but counts NaN values as zero.

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

Returns:

A new tensordict, or a tensor if `reduce=True`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.tensor([[1.0, float("nan")], [3.0, 4.0]]), batch_size=[2])
>>> td.nansum()["a"]
tensor(8.)
>>> td.sum()["a"]
tensor(nan)
>>> td.nansum(dim=0)["a"]
tensor([4., 4.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).