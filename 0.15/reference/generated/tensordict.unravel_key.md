# unravel_key

tensordict.unravel_key(*key*)

Unravel a nested key.

A tuple with a part that is neither a str nor a tuple of str unravels to
`()`. A key that is neither a str nor a tuple raises a `RuntimeError`.

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