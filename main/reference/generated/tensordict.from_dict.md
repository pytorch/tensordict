# from_dict

tensordict.from_dict(*d*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Converts a dictionary to a TensorDict.

See also

[`TensorDictBase.from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict) for more information.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, from_dict
>>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
>>> print(from_dict(input_dict, auto_batch_size=True))
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
>>> print(from_dict(input_dict, auto_batch_size=True))
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
>>> # Without auto_batch_size=True or batch_dims, the batch size is empty.
>>> from_dict(input_dict).batch_size
torch.Size([])
>>> # To work out the batch size of an existing tensordict, use auto_batch_size_
>>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
>>> input_td.auto_batch_size_().batch_size
torch.Size([3])
```