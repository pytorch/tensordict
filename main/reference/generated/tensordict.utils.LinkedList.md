# LinkedList

*class*tensordict.utils.LinkedList(**args*, *td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***kwargs*)

Bases: [`list`](https://docs.python.org/3/builtins/stdtypes.html#list)

A list of the values of a non-tensor entry that writes item assignments back to the tensordict.

`td[key]` returns a `LinkedList` when the entry is a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack). Setting an item
(`values[0] = "z"`) writes the value to the tensordict. `append` and
`extend` change the list only, and unlink it from the tensordict.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(batch_size=[3])
>>> td["names"] = ["a", "b", "c"]
>>> values = td["names"]
>>> values[0] = "z"
>>> td.get("names").tolist()
['z', 'b', 'c']
```

append(*object: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Append object to the end of the list.

clear() → [None](https://docs.python.org/3/builtins/constants.html#None)

Remove all items from list.

extend(*object: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Extend list by appending elements from the iterable.

insert(*index: [int](https://docs.python.org/3/builtins/functions.html#int)*, *object: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Insert object before index.

pop(*index: [int](https://docs.python.org/3/builtins/functions.html#int) = -1*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Remove and return item at index (default last).

Raises IndexError if list is empty or index is out of range.

remove(*value: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Remove first occurrence of value.

Raises ValueError if the value is not present.