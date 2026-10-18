# is_tensorclass

tensordict.is_tensorclass(*obj: [type](https://docs.python.org/3/builtins/functions.html#type) | [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if `obj` is a tensorclass type or an instance of one.

Parameters:

**obj** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*or*[*object*](https://docs.python.org/3/builtins/functions.html#object)) - the class or the object to check.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, is_tensorclass, tensorclass
>>> @tensorclass
... class Data:
... x: torch.Tensor
>>> is_tensorclass(Data), is_tensorclass(Data(x=torch.zeros(2)))
(True, True)
>>> is_tensorclass(TensorDict())
False
```