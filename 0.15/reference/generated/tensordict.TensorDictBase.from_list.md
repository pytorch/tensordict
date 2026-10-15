# TensorDictBase.from_list

*classmethod*TensorDictBase.from_list(*input*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lazy: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Builds a tensordict by stacking a list of dicts or tensor collections.

Each dict is converted with [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict) and the results are
stacked along a new leading dimension.

Parameters:

**input** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)) - the dicts or tensor collections to stack.

Keyword Arguments:

- **auto_batch_size** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the batch size of
each element is computed automatically. See [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict).
- **batch_size** (*iterable**of*[*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the batch size of the
result, with the length of the list first. Each element gets
`batch_size[1:]`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible type**,**optional*) - a device for
the result.
- **batch_dims** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the maximum number of batch dimensions
of the result, the list dimension included. Exclusive with
`batch_size`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the dimension names of the result,
one per batch dimension, with the name of the list dimension
first.
- **lazy** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the result is a lazy stack;
if `False`, a dense stack. Defaults to `None`, which builds
a dense stack when the elements have the same shape and a lazy
stack otherwise.

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