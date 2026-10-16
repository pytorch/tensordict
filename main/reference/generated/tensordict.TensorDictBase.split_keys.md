# TensorDictBase.split_keys

TensorDictBase.split_keys(**key_sets*, *inplace=False*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *reproduce_struct: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[T, ...]

Splits the tensordict in subsets given one or more set of keys.

The method will return `N+1` tensordicts, where `N` is the number of
the arguments provided. The last tensordict holds the keys that are in
no set.

Parameters:

- **key_sets** (*sequence**of**keys**,**Dict**[**in_key**,**out_key**] or**callable*) - the
splits, applied in order. A list or dict selects keys (a dict
also renames them). A callable `fn(key, value) -> bool`
selects every tensor leaf that no earlier set took and for
which `fn` returns `True`. `key` is the leaf's nested key
as a tuple of str, for example `("observation",)` or
`("next", "done")`, and `value` is the leaf as `get`
returns it. `fn` is not called on sub-tensordicts or
non-tensor data: select a whole sub-tensordict with its key in
a list, and non-tensor entries stay in the last tensordict
unless a list names them. `fn` is called once per leaf,
outside any graph, so it should test metadata (key, dtype,
shape, device) rather than tensor values.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the keys are removed from `self`
in-place. The leaves a callable selects are removed too.
Defaults to `False`.
- **default** (*Any**,**optional*) - the value to be returned when a key is missing.
If not specified and `strict=True`, an exception is raised.
Applies to keys in lists and dicts only: a callable never
selects a missing key.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, an exception is raised when a key
is missing. Applies to keys in lists and dicts only.
Defaults to `True`.
- **reproduce_struct** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, all tensordict returned have
the same tree structure as `self`, even if some sub-tensordicts
contain no leaves. Non-tensor entries are not copied: each
one goes to the output whose key set names it, or to the last
one, as with `reproduce_struct=False`.

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