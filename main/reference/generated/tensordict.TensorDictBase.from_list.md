# TensorDictBase.from_list

*classmethod*TensorDictBase.from_list(*input*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lazy: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Builds a tensordict by stacking a list of dicts or tensor collections.

Each dict is converted with [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict) and the results are
stacked along a new leading dimension. A tensordict or tensorclass in
the list is stacked as it is: the keyword arguments do not apply to
it.

Parameters:

**input** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)) - the dicts or tensor collections to stack.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
of each element to the largest leading shape that all its
entries share. Defaults to `None`, which behaves like
`False` unless `batch_dims` is given.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
the result, with the length of the list first. Each element
gets `batch_size[1:]`. Defaults to `None`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the maximum number of batch dimensions
of the result, the list dimension included. Turns on
`auto_batch_size` when that is `None`. Exclusive with
`batch_size`. Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the batch dimensions
of the result, one per dimension, with the name of the list
dimension first. Defaults to `None`.
- **lazy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the result is a lazy stack;
if `False`, a dense stack. Defaults to `None`, which builds
a dense stack when the elements have the same shape and a lazy
stack otherwise. See [Stacking and variable-size data](../../concepts/stacking.html#stacking).

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `batch_size[0]` is not the length of the list.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.from_list(
... [{"a": torch.zeros(3)}, {"a": torch.zeros(3)}],
... batch_size=[2, 3],
... names=["x", "y"],
... )
>>> td.batch_size
torch.Size([2, 3])
>>> td.names
['x', 'y']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).