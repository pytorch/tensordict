# make_tensordict

tensordict.make_tensordict(*input_dict: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)*) → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Returns a tensordict built from a dict or from keyword arguments.

Same as [`tensordict.TensorDict.from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict). Nested dicts become
nested tensordicts. With `auto_batch_size=True`, each nested tensordict
gets the largest batch size that its own entries share, which starts with
the batch size of its parent.

Parameters:

- **input_dict** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - the entries. Its
entries are added to those of `**kwargs` and replace them on a
shared key. Defaults to `None`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size
of the result: the leading dimensions that all its entries share.
Defaults to `None`, which gives `[]` unless
`auto_batch_size` is `True`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. Defaults to
`None`, which leaves each leaf where it is.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and `batch_size` is
not given, sets the batch size to the largest leading shape that
all the entries share. Defaults to `None`, which does not.
- ****kwargs** - the entries, as keyword arguments.

Examples

```
>>> import torch
>>> from tensordict import make_tensordict
>>> td = make_tensordict(a=torch.zeros(3, 4), b=torch.zeros(3))
>>> td.batch_size
torch.Size([])
>>> td = make_tensordict(
... {"a": torch.zeros(3), "b": {"c": torch.zeros(3, 4)}}, auto_batch_size=True
... )
>>> td.batch_size, td["b"].batch_size
(torch.Size([3]), torch.Size([3, 4]))
```