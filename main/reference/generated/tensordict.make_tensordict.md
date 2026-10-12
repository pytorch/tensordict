# make_tensordict

tensordict.make_tensordict(*input_dict: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | TensorCollection*) → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Returns a TensorDict created from the keyword arguments or an input dictionary.

If `batch_size` is not specified and `auto_batch_size=True`, the maximum batch size possible is used.

This function works on nested dictionaries too, or can be used to determine the
batch-size of a nested tensordict.

Parameters:

- **input_dict** (*dictionary**,**optional*) - a dictionary to use as a data source
(nested keys compatible).
- ****kwargs** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*or*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - keyword arguments as data source
(incompatible with nested keys).
- **batch_size** (*iterable**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - a batch size for the tensordict.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible type**,**optional*) - a device for the TensorDict.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the batch size will be computed automatically.
Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, make_tensordict
>>> input_dict = {"a": torch.randn(3, 4), "b": torch.randn(3)}
>>> print(make_tensordict(input_dict, auto_batch_size=True))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> # alternatively
>>> td = make_tensordict(**input_dict)
>>> # nested dict: the nested TensorDict can have a different batch-size
>>> # as long as its leading dims match.
>>> input_dict = {"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}
>>> print(make_tensordict(input_dict, auto_batch_size=True))
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
>>> # we can also use this to work out the batch sie of a tensordict
>>> input_td = TensorDict({"a": torch.randn(3), "b": {"c": torch.randn(3, 4)}}, [])
>>> print(make_tensordict(input_td, auto_batch_size=True))
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