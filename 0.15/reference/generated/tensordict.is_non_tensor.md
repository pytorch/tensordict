# is_non_tensor

tensordict.is_non_tensor(*data*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if `data` is non-tensor data or a stack of non-tensor data.

Parameters:

**data** ([*object*](https://docs.python.org/3/builtins/functions.html#object)) - the object to check, for example a value read from a
tensordict with [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get).

Examples

```
>>> import torch
>>> from tensordict import NonTensorData, TensorDict, is_non_tensor
>>> td = TensorDict(x=torch.zeros(2), name=NonTensorData("a"))
>>> is_non_tensor(td.get("name")), is_non_tensor(td.get("x"))
(True, False)
>>> td["name"]
'a'
```