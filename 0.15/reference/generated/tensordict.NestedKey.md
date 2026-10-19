# NestedKey

*class*tensordict.NestedKey

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

The type of the keys of a tensordict: a string, or a tuple of strings for a nested entry.

A nested key is either a string or a non-empty tuple of nested keys.
`td["a", "b"]` and `td[("a", "b")]` read the entry `"b"` of the
nested tensordict `"a"`. Nested tuples are flattened: `("a", ("b",))`
is the same key as `("a", "b")`. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).

Use `NestedKey` in type hints and in `isinstance` checks.

Examples

```
>>> from tensordict import NestedKey, TensorDict
>>> isinstance("a", NestedKey), isinstance(("a", ("b",)), NestedKey)
(True, True)
>>> isinstance(("a", 1), NestedKey), isinstance((), NestedKey)
(False, False)
>>> td = TensorDict({"a": {"b": 0}})
>>> td["a", "b"] is td[("a", ("b",))]
True
```

## Tutorials that use `NestedKey`

![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams