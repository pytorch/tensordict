# is_integer

tensordict.is_integer(**args*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the last argument is an integer tensor (signed or unsigned, not bool).

The predicate tests its last argument only, so it can be passed as
`fn(value)` and as `fn(key, value)`: see [`is_inexact()`](tensordict.is_inexact.html#tensordict.is_inexact).

Parameters:

***args** - the value to test is the last one. A key may come before it.

Returns:

`True` if the dtype of the value is neither floating-point, complex
nor `torch.bool`.

Examples

```
>>> import torch
>>> from tensordict import is_integer
>>> is_integer(torch.zeros(3, dtype=torch.uint8)), is_integer(torch.zeros(3, dtype=torch.bool))
(True, False)
```