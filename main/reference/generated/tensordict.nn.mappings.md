# mappings

tensordict.nn.mappings(*key: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*) → [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)

Returns the function named by `key`.

All the functions except `"none"` (the identity) map real numbers to
non-negative numbers. `"relu"` can return 0, the others return
positive numbers.

Parameters:

**key** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - one of "softplus", "exp", "relu", "expln",
"biased_softplus" or "none" (no mapping).

Note

If the key begins with `"biased_softplus"`, then it needs to take the following form:
`"biased_softplus_{bias}"` where `bias` can be converted to a floating point number.
Alternatively, the `"biased_softplus_{bias}_{min_val}"` syntax can be used.
The function is then [`biased_softplus`](tensordict.nn.biased_softplus.html#tensordict.nn.biased_softplus) with these arguments:
`softplus(x + inv_softplus(bias - min_val)) + min_val`, which returns
`bias` for a zero input and is always greater than `min_val`.

Note

Custom mappings can be added through `tensordict.nn.add_custom_mapping`.

Returns:

a Callable