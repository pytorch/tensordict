# TensorDictBase.keys

*abstract*TensorDictBase.keys(*include_nested: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *leaves_only: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *is_leaf: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Type](https://docs.python.org/3/library/typing.html#typing.Type)], [bool](https://docs.python.org/3/builtins/functions.html#bool)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *sort: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns a view of the keys of the tensordict.

By default, only the top-level keys are returned. A nested key is a
tuple of strings, for example `("nested", "b")`. See [Keys, nested keys and locking](../../concepts/keys.html#keys).

Warning

The view is lazy. If you modify the tensordict after you call
`keys()` and before you iterate over the result, the iteration
returns the keys after the change.

Parameters:

- **include_nested** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the entries of the
sub-tensordicts are also returned, with nested keys (tuples of
strings). Defaults to `False`.
- **leaves_only** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, only leaves are returned,
and sub-tensordicts are skipped. Defaults to `False`.
- **is_leaf** (*callable**,**optional*) - a function that takes a class and
returns `True` if values of that class count as leaves. It
only filters the result when `leaves_only=True`. It does not
stop the recursion: with `include_nested=True`, a
sub-tensordict is still searched even if `is_leaf` returns
`True` for it. By default, tensordicts and tensorclasses are
not leaves. See [`is_leaf_nontensor()`](tensordict.is_leaf_nontensor.html#tensordict.is_leaf_nontensor) and
[`default_is_leaf()`](tensordict.default_is_leaf.html#tensordict.default_is_leaf).

Keyword Arguments:

**sort** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, the keys are sorted. Nested keys
are sorted by their joined name: `("a", "key")` sorts as
`"a.key"`. Sorting takes extra time on large tensordicts.
Defaults to `False`.

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({"0": 0, "1": {"2": 2}}, batch_size=[])
>>> list(data.keys())
['0', '1']
>>> list(data.keys(leaves_only=True))
['0']
>>> list(data.keys(include_nested=True, leaves_only=True))
['0', ('1', '2')]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `keys`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting and nesting keys