# skip_existing

tensordict.nn.skip_existing() → [bool](https://docs.python.org/3/builtins/functions.html#bool) | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]

Returns the mode set by [`set_skip_existing`](tensordict.nn.set_skip_existing.html#tensordict.nn.set_skip_existing): `False` (the default), `True`, or a list of keys.

Examples

```
>>> from tensordict.nn import set_skip_existing, skip_existing
>>> skip_existing()
False
>>> with set_skip_existing(["a"]):
... skip_existing()
['a']
```