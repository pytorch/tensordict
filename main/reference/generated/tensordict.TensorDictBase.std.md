# TensorDictBase.std

TensorDictBase.std(*dim: [int](https://docs.python.org/3/builtins/functions.html#int) | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['feature'] = NO_DEFAULT*, *keepdim: [bool](https://docs.python.org/3/builtins/functions.html#bool) = NO_DEFAULT*, ***, *correction: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*, *reduce: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *key_transform: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)], [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Returns the standard deviation value of all elements in the input tensordict.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if `None`, returns a dimensionless
tensordict containing the sum value of all leaves (if this can be computed).
If integer or tuple of integers, std is called upon the dimension specified if
and only if this dimension is compatible with the tensordict
shape.
Only the "feature" string is currently permitted. Using dim="feature" will
achieve the reduction over all feature dimensions. If reduce=True, a tensor of the
shape of the TensorDict's batch-size will be returned. Otherwise, a new tensordict
with the same structure as `self` with reduced feature dimensions will be returned.
With `dim="feature"`, a leaf that has no feature dimension (as `b.d` in the
example) gives NaN, and torch warns that the degrees of freedom are `<= 0`.
- **keepdim** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - whether the output tensor has dim retained or not.

Keyword Arguments:

- **correction** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - difference between the sample size and sample degrees of freedom.
Defaults to Bessel's correction, correction=1.
- **reduce** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the reduction will occur across all TensorDict values
and a single reduced tensor will be returned.
Defaults to `False`.
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
>>> td.std(dim=0)
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
>>> td.std()
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
>>> td.std(reduce=True)
tensor(1.0353)
>>> td.std(dim="feature") # b.d has no feature dimension: NaN and a UserWarning
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
>>> td = TensorDict(
... a=torch.ones(3, 4, 5),
... b=TensorDict(
... c=torch.ones(3, 4, 5),
... d=torch.ones(3, 4, 5),
... batch_size=(3, 4, 5),
... ),
... batch_size=(3, 4)
... )
>>> td.std(reduce=True, dim="feature")
tensor([[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]])
>>> td.std(reduce=True, dim=0)
tensor([[0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).