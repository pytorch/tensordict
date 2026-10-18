# TensorDictBase.tolist

TensorDictBase.tolist(***, *convert_nodes: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *convert_tensors: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['numpy'] = False*, *tolist_first: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *as_linked_list: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Returns the elements of the tensordict as nested lists, one level per batch dimension.

The innermost elements are the tensordict indexed at every batch
index, converted with [`to_dict()`](tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict): for batch size `[2, 3]`,
the result is a list of 2 lists of 3 dicts. A tensordict without
batch dimensions returns one dict.

Keyword Arguments:

- **convert_nodes** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the elements are
converted to dicts. If `False`, they are kept as
tensordicts. Defaults to `True`.
- **convert_tensors** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - as in [`to_dict()`](tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict):
if `True`, tensors become Python lists, and if `"numpy"`,
NumPy arrays. Requires `convert_nodes=True`. Defaults to
`False`.
- **tolist_first** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - as in [`to_dict()`](tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict): if
`True`, a nested tensordict with batch dimensions becomes a
list of dicts. Defaults to `False`.
- **as_linked_list** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns a
[`LinkedList`](tensordict.utils.LinkedList.html#tensordict.utils.LinkedList): assigning an element of
the outer list (`result[0] = {...}`) writes it into the
tensordict. Defaults to `False`.

Returns:

A list (nested once per batch dimension), or a dict (or `self`
if `convert_nodes=False`) when the tensordict has no batch
dimensions.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `convert_tensors` is set and `convert_nodes=False`.

See also

[`to_dict()`](tensordict.TensorDictBase.to_dict.html#tensordict.TensorDictBase.to_dict).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=torch.arange(4).view(2, 2),
... b=TensorDict(c=torch.arange(8).view(2, 2, 2), batch_size=(2, 2, 2)),
... batch_size=(2,),
... )
>>> td.tolist()
[{'a': tensor([0, 1]), 'b': {'c': tensor([[0, 1],
 [2, 3]])}}, {'a': tensor([2, 3]), 'b': {'c': tensor([[4, 5],
 [6, 7]])}}]
>>> td.tolist(convert_tensors=True)
[{'a': [0, 1], 'b': {'c': [[0, 1], [2, 3]]}}, {'a': [2, 3], 'b': {'c': [[4, 5], [6, 7]]}}]
>>> td.tolist(tolist_first=True, convert_tensors=True)[0]
{'a': [0, 1], 'b': [[{'c': 0}, {'c': 1}], [{'c': 2}, {'c': 3}]]}
>>> type(td.tolist(convert_nodes=False)[0]).__name__
'TensorDict'
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).