# TensorDictBase.split_keys

TensorDictBase.split_keys(**key_sets*, *inplace=False*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *reproduce_struct: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[T, ...]

Splits the tensordict in subsets given one or more set of keys.

The method will return `N+1` tensordicts, where `N` is the number of
the arguments provided. The last tensordict holds the keys that are in
no set.

Parameters:

- **key_sets** (*sequence**of**Dict**[**in_key**,**out_key**] or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of**keys*) - the various splits.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the keys are removed from `self`
in-place. Defaults to `False`.
- **default** (*Any**,**optional*) - the value to be returned when a key is missing.
If not specified and `strict=True`, an exception is raised.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, an exception is raised when a key
is missing. Defaults to `True`.
- **reproduce_struct** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, all tensordict returned have
the same tree structure as `self`, even if some sub-tensordicts
contain no leaves.

Note

`None` non-tensor values will be ignored and not returned.

Note

The method does not check for duplicates in the provided lists.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict(
... a=0,
... b=0,
... c=0,
... d=0,
... )
>>> td_a, td_bc, td_d = td.split_keys(["a"], ["b", "c"])
>>> sorted(td_a.keys()), sorted(td_bc.keys()), sorted(td_d.keys())
(['a'], ['b', 'c'], ['d'])
>>> # with inplace=True, the last tensordict is ``td`` itself
>>> td_a, rest = td.split_keys(["a"], inplace=True)
>>> rest is td, sorted(td.keys())
(True, ['b', 'c', 'd'])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).