# NonTensorStack

`NonTensorStack` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `NonTensorStack` adds, and the overrides that have their own
documentation.

*class*tensordict.NonTensorStack(**args*, ***kwargs*)

Bases: [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)

A lazy stack of [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData), one value per batch element.

You get a `NonTensorStack` when you stack
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) (or `NonTensorStack`) instances
with [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack), or when you assign a list to a key:
`td["names"] = ["a", "b", "c"]`. You can also build one with
`NonTensorStack("a", "b")` or `NonTensorStack.from_list(["a", "b"])`.

To read the values, call `tolist()`. `td["names"]` returns a
[`LinkedList`](tensordict.utils.LinkedList.html#tensordict.utils.LinkedList) of the values, and
`td.get("names")` returns the `NonTensorStack` itself. If every
element holds the same value, `data` returns it. To keep one shared
value for the whole batch, use [`MetaData`](tensordict.MetaData.html#tensordict.MetaData). See
[Non-tensor data and unbatched entries](../../concepts/non_tensor.html#non-tensor-data).

Note

`torch.utils._pytree.tree_unflatten` and `tree_map` rebuild a
`NonTensorStack` entry of a tensordict whose values are all equal
from the input, as a [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData), and drop the value
that the function returned for it; they warn when that changes the result.
Starting with TensorDict 0.17, the default will change to keeping the
returned value, as they do when the values differ.

Examples

```
>>> import torch
>>> from tensordict import NonTensorData, NonTensorStack, TensorDict
>>> NonTensorStack("a", "b").tolist()
['a', 'b']
>>> td = TensorDict(batch_size=[3])
>>> td["names"] = ["a", "b", "c"]
>>> td["names"]
LinkedList(['a', 'b', 'c'])
>>> td[1]["names"]
'b'
>>> type(td.get("names")).__name__
'NonTensorStack'
>>> data = torch.stack([
... torch.stack([NonTensorData(data=(i, j)) for i in range(2)])
... for j in range(3)])
>>> data.batch_size
torch.Size([3, 2])
>>> data[2, 1]
NonTensorData(data=(1, 2), batch_size=torch.Size([]), device=None)
```

*property*data*: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*

The value that every element of the stack holds.

Raises:

[**AttributeError**](https://docs.python.org/3/builtins/exceptions.html#AttributeError) - if the elements hold different values. Use
 `tolist()` to read them.

Examples

```
>>> import torch
>>> from tensordict import NonTensorData
>>> torch.stack([NonTensorData("x"), NonTensorData("x")]).data
'x'
```

*classmethod*from_nontensordata(*non_tensor: [NonTensorData](tensordict.NonTensorData.html#tensordict.NonTensorData)*)

Expands a [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) into a stack of copies of it.

Deprecated since version 0.15: Use `NonTensorData.maybe_to_stack()` instead. It differs in
two ways: its elements share the data object of `non_tensor`,
whereas this method copies it into every element; and when
`non_tensor` has an empty batch size, it returns `non_tensor`
itself, whereas this method returns a copy.

maybe_to_stack()

Returns a new stack in which each element with a batch size is a stack too.

It calls [`NonTensorDataBase.maybe_to_stack`](tensordict.NonTensorDataBase.html#tensordict.NonTensorDataBase.maybe_to_stack) on each element, so that
`maybe_to_stack` can be called on any non-tensor entry.

tolist(***, *convert_nodes: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *convert_tensors: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['numpy'] = False*, *tolist_first: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *as_linked_list: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns the values of the stack in a nested list, with one item per batch element.

Keyword Arguments:

- **convert_nodes** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - accepted for compatibility with
[`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist). It has no effect:
the items are the stored values, not tensordicts. Defaults to
`True`.
- **convert_tensors** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or**"numpy"**,**optional*) - accepted for
compatibility with [`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist).
It has no effect: a value that is a tensor stays a tensor.
Defaults to `False`.
- **tolist_first** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - accepted for compatibility with
[`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist). It has no effect.
Defaults to `False`.
- **as_linked_list** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns a
`tensordict.utils.LinkedList` bound to the stack: assigning
one of its items writes into the stack. Defaults to `False`.

Examples

```
>>> from tensordict import NonTensorData
>>> import torch
>>> data = torch.stack([
... torch.stack([NonTensorData(data=(i, j)) for i in range(2)])
... for j in range(3)])
>>> data.tolist()
[[(0, 0), (1, 0)], [(0, 1), (1, 1)], [(0, 2), (1, 2)]]
```