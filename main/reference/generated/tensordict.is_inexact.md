# is_inexact

tensordict.is_inexact(**args*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Whether the last argument is a floating-point or complex tensor.

The predicate tests its last argument only, so it works both as
`fn(value)`, for example inside a function given to
[`apply()`](tensordict.TensorDictBase.apply.html#tensordict.TensorDictBase.apply), and as `fn(key, value)`, for
example as the `cast_filter` of [`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to).

Parameters:

***args** - the value to test is the last one. A key may come before it.

Returns:

`True` if the dtype of the value is a floating-point or complex dtype.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, is_inexact
>>> is_inexact(torch.zeros(3)), is_inexact("a", torch.zeros(3, dtype=torch.long))
(True, False)
>>> td = TensorDict(x=torch.zeros(3), i=torch.zeros(3, dtype=torch.long))
>>> td_bf16 = td.bfloat16(cast_filter=is_inexact)
>>> td_bf16["x"].dtype, td_bf16["i"].dtype
(torch.bfloat16, torch.int64)
```