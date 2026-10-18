# TensorDictBase.to_dict

TensorDictBase.to_dict(***, *retain_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *convert_tensors: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['numpy'] = False*, *tolist_first: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Returns a nested dictionary with the entries of the tensordict.

Nested tensordicts and tensorclasses become nested dicts. The tensors
are not copied unless `convert_tensors` is set. Non-tensor data is
returned as its Python value.

Keyword Arguments:

- **retain_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the `None` fields of
tensorclasses are written in the dict as `None`. If
`False`, they are dropped. Defaults to `True`.
- **convert_tensors** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - if `True`, tensors are
converted to (nested) Python lists with
[`torch.Tensor.tolist()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.tolist.html#torch.Tensor.tolist). If `"numpy"`, they are converted
to NumPy arrays. If `False`, they are kept as tensors.
Defaults to `False`.
- **tolist_first** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a nested tensordict is
converted with [`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist), so that a nested tensordict
with batch dimensions becomes a list of dicts, one per element.
Defaults to `False`.

Returns:

A dict whose keys are those of the tensordict.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - on a lazy stack whose tensordicts have different
 keys (see [Variable-size data](../../concepts/stacking.html#stacking-variable-size)).

See also

[`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist), [`numpy()`](tensordict.TensorDictBase.numpy.html#tensordict.TensorDictBase.numpy) and
[`from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict), which converts back.

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