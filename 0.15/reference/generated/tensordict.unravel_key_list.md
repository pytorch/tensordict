# unravel_key_list

tensordict.unravel_key_list(*keys*)

Unravels every key of a list, as [`unravel_key()`](tensordict.unravel_key.html#tensordict.unravel_key) does.

Parameters:

**keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys to unravel.

Returns:

A list of the unraveled keys (strings, or flat tuples of strings).

Raises:

- [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `keys` is not a list or a tuple.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a key is not a valid nested key.

Examples

```
>>> from tensordict import unravel_key_list
>>> unravel_key_list(["a", ("b",), ("c", ("d",))])
['a', 'b', ('c', 'd')]
```