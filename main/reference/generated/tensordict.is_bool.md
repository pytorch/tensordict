# is_bool

tensordict.is_bool(**args*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the last argument is a boolean tensor.

The predicate tests its last argument only, so it can be passed as
`fn(value)` and as `fn(key, value)`: see [`is_inexact()`](tensordict.is_inexact.html#tensordict.is_inexact).

Parameters:

***args** - the value to test is the last one. A key may come before it.

Returns:

`True` if the dtype of the value is `torch.bool`.

Examples

```
>>> import torch
>>> from tensordict import is_bool
>>> is_bool(torch.zeros(3, dtype=torch.bool)), is_bool("mask", torch.zeros(3))
(True, False)
```