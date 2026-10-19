# unravel_key

tensordict.unravel_key(*key*)

Unravels a nested key into a str or a flat tuple of str.

Nested tuples are flattened, and a tuple with a single str gives that
str, so equal keys compare equal: `unravel_key(("a",)) == "a"`. A tuple
with a part that is neither a str nor a tuple of str unravels to `()`.
See [Keys, nested keys, and locking](../../concepts/keys.html#keys).

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to unravel.

Returns:

A str, a flat tuple of at least two str, or `()`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `key` is neither a str nor a tuple.

Examples

```
>>> from tensordict import unravel_key
>>> unravel_key("a")
'a'
>>> unravel_key(("a",))
'a'
>>> unravel_key((("a", ("b",))))
('a', 'b')
>>> unravel_key(("a", 1))
()
```