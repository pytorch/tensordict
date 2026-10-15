# TensorDictBase.tolist

TensorDictBase.tolist(***, *convert_nodes: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *convert_tensors: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['numpy'] = False*, *tolist_first: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *as_linked_list: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)]

Returns a nested list representation of the tensordict.

If the tensordict has no batch dimensions, this method returns a single list or dictionary.
Otherwise, it returns a nested list where each inner list represents a batch dimension.

Parameters:

- **convert_nodes** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, leaf nodes will be converted to dictionaries.
Otherwise, they will be returned as tensordicts. Default: `True`.
- **convert_tensors** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**"numpy"*) - if `True`, tensors will be converted to lists when creating the dictionary.
If "numpy", tensors will be converted to numpy arrays.
Otherwise, they will remain as tensors. Default: `False`.
- **tolist_first** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, the tensordict will be converted to a list first when
it has batch dimensions. Default: `False`.
- **as_linked_list** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, the list will be converted to a [`tensordict.utils.LinkedList`](tensordict.utils.LinkedList.html#tensordict.utils.LinkedList)
which will automatically update the tensordict when the list is modified. Default: `False`.

Returns:

A nested list representation of the tensordict.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>>
>>> td = TensorDict(
... a=torch.arange(24).view(2, 3, 4),
... b=TensorDict(c=torch.arange(12).reshape(2, 3, 2), batch_size=(2, 3, 2)),
... batch_size=(2, 3)
... )
>>> print(td.tolist(tolist_first=True))
[[{'a': tensor([0, 1, 2, 3]), 'b': [{'c': tensor(0)}, {'c': tensor(1)}]}, {'a': tensor([4, 5, 6, 7]), 'b': [{'c': tensor(2)}, {'c': tensor(3)}]}, {'a': tensor([ 8, 9, 10, 11]), 'b': [{'c': tensor(4)}, {'c': tensor(5)}]}], [{'a': tensor([12, 13, 14, 15]), 'b': [{'c': tensor(6)}, {'c': tensor(7)}]}, {'a': tensor([16, 17, 18, 19]), 'b': [{'c': tensor(8)}, {'c': tensor(9)}]}, {'a': tensor([20, 21, 22, 23]), 'b': [{'c': tensor(10)}, {'c': tensor(11)}]}]]
>>> print(td.tolist(tolist_first=False))
[[{'a': tensor([0, 1, 2, 3]), 'b': {'c': tensor([0, 1])}}, {'a': tensor([4, 5, 6, 7]), 'b': {'c': tensor([2, 3])}}, {'a': tensor([ 8, 9, 10, 11]), 'b': {'c': tensor([4, 5])}}], [{'a': tensor([12, 13, 14, 15]), 'b': {'c': tensor([6, 7])}}, {'a': tensor([16, 17, 18, 19]), 'b': {'c': tensor([8, 9])}}, {'a': tensor([20, 21, 22, 23]), 'b': {'c': tensor([10, 11])}}]]
>>> print(td.tolist(convert_tensors=True))
[[{'a': [0, 1, 2, 3], 'b': {'c': [0, 1]}}, {'a': [4, 5, 6, 7], 'b': {'c': [2, 3]}}, {'a': [8, 9, 10, 11], 'b': {'c': [4, 5]}}], [{'a': [12, 13, 14, 15], 'b': {'c': [6, 7]}}, {'a': [16, 17, 18, 19], 'b': {'c': [8, 9]}}, {'a': [20, 21, 22, 23], 'b': {'c': [10, 11]}}]]
>>> print(td.tolist(convert_nodes=False)[0][0])
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([4]), device=cpu, dtype=torch.int64, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).