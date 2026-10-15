# TensorDictBase.to_dict

TensorDictBase.to_dict(***, *retain_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *convert_tensors: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['numpy'] = False*, *tolist_first: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Returns a dictionary with key-value pairs matching those of the tensordict.

Parameters:

- **retain_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, the `None` values from tensorclass instances
will be written in the dictionary.
Otherwise, they will be discarded. Default: `True`.
- **convert_tensors** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**"numpy"*) - if `True`, tensors will be converted to lists when creating the dictionary.
If "numpy", tensors will be converted to numpy arrays.
Otherwise, they will remain as tensors. Default: `False`.
- **tolist_first** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, the tensordict will be converted to a list first when
it has batch dimensions. Default: `False`.

Returns:

A dictionary representation of the tensordict.

See also

[`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist)

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>>
>>> td = TensorDict(
... a=torch.arange(6).view(2, 3),
... b=TensorDict(c=torch.arange(4).reshape(2, 2), batch_size=(2, 2)),
... batch_size=(2,)
... )
>>> print(td.to_dict())
{'a': tensor([[0, 1, 2],
 [3, 4, 5]]), 'b': {'c': tensor([[0, 1],
 [2, 3]])}}
>>> print(td.to_dict(convert_tensors=True))
{'a': [[0, 1, 2], [3, 4, 5]], 'b': {'c': [[0, 1], [2, 3]]}}
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).