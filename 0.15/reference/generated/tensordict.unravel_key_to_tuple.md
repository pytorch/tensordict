# unravel_key_to_tuple

tensordict.unravel_key_to_tuple(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), ...]

Unravels a nested key into a flat tuple of str.

Unlike [`unravel_key()`](tensordict.unravel_key.html#tensordict.unravel_key), a str key gives a one-element
tuple, and an invalid key gives `()` instead of raising.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to unravel.

Returns:

A flat tuple of str, or `()` if `key` is not a valid nested key.

Examples

```
>>> from tensordict import unravel_key_to_tuple
>>> unravel_key_to_tuple("a")
('a',)
>>> unravel_key_to_tuple(("a", ("b", ("c",))))
('a', 'b', 'c')
>>> unravel_key_to_tuple(("a", 1))
()
>>> unravel_key_to_tuple(1)
()
```