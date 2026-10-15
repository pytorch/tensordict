# TensorDictBase.quantile

TensorDictBase.quantile(*q: [float](https://docs.python.org/3/builtins/functions.html#float) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['feature'] = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = NO_DEFAULT*, ***, *interpolation: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'linear'*, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *key_transform: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)], [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the q-th quantile of all elements in the input tensordict.

Parameters:

- **q** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - quantile to compute, must be between 0 and 1.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - if `None`, returns a dimensionless
tensordict containing the quantile value of all leaves (if this can be computed).
If integer, quantile is called upon the dimension specified if
and only if this dimension is compatible with the tensordict
shape.
A tuple reduces only its first dim; passing one is deprecated
and will be removed in TensorDict 0.17.
Only the "feature" string is currently permitted. Using dim="feature" will
achieve the reduction over all feature dimensions. If reduce=True, a tensor of the
shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
with the same structure as `self` with reduced feature dimensions will be returned.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - whether the output tensor has dim retained or not.

Keyword Arguments:

- **interpolation** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - interpolation method to use when the desired quantile lies
between two data points. Options are 'linear', 'lower', 'higher', 'midpoint', and 'nearest'.
Defaults to 'linear'.
- **reduce** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the reduction will occur across all TensorDict values
and a single reduced tensor will be returned.
Defaults to `False`. With feature dims, a negative `dim` counts from the end of
each leaf. Starting with TensorDict 0.17, the default will change to counting it from
the end of the batch dims, as with `reduce=False`.
- **key_transform** (*Callable**[**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**,*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**,**optional*) - A function to transform key names.
If provided, all keys in the result will be transformed using this function.
For string keys, the function receives a string. For tuple keys, it receives a tuple.
Only applied when `reduce=False`. Default: `None`.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> _ = torch.manual_seed(0)
>>> td = TensorDict(
... a=torch.randn(3, 4, 5),
... b=TensorDict(
... c=torch.randn(3, 4, 5, 6),
... d=torch.randn(3, 4, 5),
... batch_size=(3, 4, 5),
... ),
... batch_size=(3, 4)
... )
>>> td.quantile(0.5, dim=0) # median along dim 0
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([4, 5]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([4]),
 device=None,
 is_shared=False)
>>> td.quantile(0.5) # median of all elements
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> td.quantile(0.5, reduce=True) # single median value
tensor(0.0603)
>>> td.quantile(0.5, dim="feature") # median along feature dimensions
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 4, 5]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False)
>>> # Multiple quantiles
>>> td.quantile(torch.tensor([0.25, 0.5, 0.75]), dim=0)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([3, 4, 5, 6]), device=cpu, dtype=torch.float32, is_shared=False),
 d: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 4, 5]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).