# TensorDictBase.split_keys

TensorDictBase.split_keys(**key_sets*, *inplace=False*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *reproduce_struct: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[T, ...]

Splits the tensordict into one tensordict per set of keys, plus one for the remaining entries.

Returns `N + 1` tensordicts for `N` key sets: the last one holds
the entries that no set took. The values are not copied. Each set
takes its entries from what the earlier sets left. Empty nested
tensordicts are removed from the last tensordict, unless
`reproduce_struct=True`.

Parameters:

***key_sets** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*or**Callable*) - the sets of keys,
applied in order. A list selects keys, and a dict selects its
keys and renames each to its value. A callable
`fn(key, value) -> bool` selects every tensor leaf that no
earlier set took and for which `fn` returns `True`. `key`
is the nested key of the leaf as a tuple of str, for example
`("observation",)` or `("next", "done")`, and `value` is
the leaf as `get` returns it. `fn` is not called on nested
tensordicts or non-tensor data: select a whole nested
tensordict with its key in a list, and non-tensor entries stay
in the last tensordict unless a list names them. `fn` is
called once per leaf, outside any graph, so it should test
metadata (key, dtype, shape, device) rather than tensor values.

Keyword Arguments:

- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the selected entries are
removed from the tensordict, which is returned as the last
tensordict. Defaults to `False`.
- **default** (*Any**,**optional*) - with `strict=False`, the value put in
the output for a missing key. By default, a missing key is
skipped. A `default` of `None` also skips the key. Applies to the keys in lists and dicts only: a
callable never selects a missing key.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, a missing key raises a
`KeyError`, and `default` is ignored. Applies to the keys in
lists and dicts only. Defaults to `True`.
- **reproduce_struct** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, every returned
tensordict has the tree of nested tensordicts of the
tensordict, even where a nested tensordict holds no leaves.
Non-tensor entries are not copied: each one goes to the output
whose key set names it, or to the last one, as with
`reproduce_struct=False`. Defaults to `False`.

Returns:

A tuple of `N + 1` tensordicts.

Note

Duplicate keys in the sets are not checked.

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
>>> # with inplace=True, the last tensordict is td itself
>>> td_a, rest = td.split_keys(["a"], inplace=True)
>>> rest is td, sorted(td.keys())
(True, ['b', 'c', 'd'])
>>> # a callable selects leaves by key and value; the first match wins
>>> import torch
>>> td = TensorDict(
... s=torch.zeros(10),
... v=torch.zeros(10, 3),
... m=torch.zeros(10, 4, 4),
... nested=TensorDict(t=torch.zeros(10, 3), batch_size=[10]),
... batch_size=[10],
... )
>>> nb = td.batch_dims
>>> scalars, vectors, rest = td.split_keys(
... lambda k, v: v.ndim == nb, lambda k, v: v.ndim == nb + 1
... )
>>> sorted(scalars.keys(True, True), key=str)
['s']
>>> sorted(vectors.keys(True, True), key=str)
[('nested', 't'), 'v']
>>> sorted(rest.keys(True, True), key=str)
['m']
>>> # lists and callables mix: the nested key goes first
>>> nested, vectors, rest = td.split_keys(
... ["nested"], lambda k, v: v.ndim == nb + 1
... )
>>> sorted(vectors.keys())
['v']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).