# is_floating

tensordict.is_floating(**args*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the last argument is a floating-point tensor.

The predicate tests its last argument only, so it can be passed as
`fn(value)` and as `fn(key, value)`: see [`is_inexact()`](tensordict.is_inexact.html#tensordict.is_inexact).

Parameters:

***args** - the value to test is the last one. A key may come before it.

Returns:

`True` if the dtype of the value is a floating-point dtype.

Examples

```
>>> import torch
>>> from tensordict import is_floating
>>> is_floating(torch.zeros(3)), is_floating(torch.zeros(3, dtype=torch.complex64))
(True, False)
```