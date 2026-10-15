# TensorDictBase.from_dict

*abstract classmethod*TensorDictBase.from_dict(*input_dict*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [List](https://docs.python.org/3/library/typing.html#typing.List)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Returns a TensorDict created from a dictionary or another [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

If `batch_size` is not specified and `auto_batch_size=True`, the maximum batch size possible is used.

This function works on nested dictionaries too. For [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict), a tensor
collection passed as `input_dict` is currently returned as is, and the keyword arguments are
not applied to it: use [`auto_batch_size_()`](tensordict.TensorDictBase.auto_batch_size_.html#tensordict.TensorDictBase.auto_batch_size_) to compute the batch size of an existing
tensordict.

Parameters:

**input_dict** (*dictionary**,**optional*) - a dictionary to use as a data source
(nested keys compatible).

Keyword Arguments:

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the batch size will be computed automatically.
Defaults to `None`, which behaves like `False`, unless
`batch_dims` is passed: then it behaves like `True`.
- **batch_size** (*iterable**of*[*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - a batch size for the tensordict.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible type**,**optional*) - a device for the TensorDict.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the maximum number of leading dimensions
of the computed batch size. A smaller number is used if the
tensors do not share that many leading dimensions. Exclusive
with `batch_size`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the dimension names of the tensordict.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
>>> print(TensorDict.from_dict(input_dict, auto_batch_size=True))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> # nested dict: the nested TensorDict can have a different batch-size
>>> # as long as its leading dims match.
>>> input_dict = {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}
>>> print(TensorDict.from_dict(input_dict, auto_batch_size=True))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> # to work out the batch size of an existing tensordict, use auto_batch_size_
>>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
>>> print(input_td.auto_batch_size_())
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).