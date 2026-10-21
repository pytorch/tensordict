# LinkedList

*class*tensordict.utils.LinkedList(**args*, *td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*)

Bases: [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

A list of the values of a non-tensor entry that writes item assignments back to the tensordict.

`td[key]` returns a `LinkedList` when the entry is a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack), and so does
[`tolist()`](tensordict.TensorDictBase.tolist.html#tensordict.TensorDictBase.tolist) with `as_linked_list=True`.
Setting an item (`values[0] = "z"`) writes the value to the tensordict.
The methods that change the length of the list (`append`, `extend`,
`insert`, `pop`, `remove`, `clear` and `del`) change the list
only, and unlink it from the tensordict: later item assignments no longer
write back. A `LinkedList` compares equal to a list with the same
items. If the write to the tensordict raises, the list is unchanged.
`tolist()` returns a plain [`list`](https://docs.python.org/3/builtins/stdtypes.html#list) copy that is not linked to
the tensordict, as [`tolist()`](tensordict.NonTensorStack.html#tensordict.NonTensorStack.tolist) does.
`LinkedList(iterable)`, without `td`, builds a list that is not
linked to any tensordict.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(batch_size=[3])
>>> td["names"] = ["a", "b", "c"]
>>> values = td["names"]
>>> values[0] = "z"
>>> td.get("names").tolist()
['z', 'b', 'c']
>>> values.tolist()
['z', 'b', 'c']
```

append(*object: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Appends `object` to the list and unlinks the list from the tensordict.

clear() → [None](https://docs.python.org/3/builtins/constants.html#None)

Removes every item and unlinks the list from the tensordict.

extend(*object: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Extends the list with the items of `object` and unlinks it from the tensordict.

insert(*index: [int](https://docs.python.org/3/builtins/functions.html#int)*, *object: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Inserts `object` before `index` and unlinks the list from the tensordict.

pop(*index: [int](https://docs.python.org/3/builtins/functions.html#int) = -1*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Removes and returns the item at `index` and unlinks the list from the tensordict.

remove(*value: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Removes the first item equal to `value` and unlinks the list from the tensordict.

tolist() → [list](https://docs.python.org/3/builtins/stdtypes.html#list)

Returns a plain [`list`](https://docs.python.org/3/builtins/stdtypes.html#list) copy of the items, not linked to the tensordict.

Nested `LinkedList` items, as `td[key]` returns for a
non-tensor entry with more than one batch dimension, become nested
lists, as with [`tolist()`](tensordict.NonTensorStack.html#tensordict.NonTensorStack.tolist). The other
items are not copied.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(batch_size=[2, 2])
>>> td["names"] = [["a", "b"], ["c", "d"]]
>>> td["names"].tolist()
[['a', 'b'], ['c', 'd']]
```