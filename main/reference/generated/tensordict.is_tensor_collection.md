# is_tensor_collection

tensordict.is_tensor_collection(*datatype: [type](https://docs.python.org/3/builtins/functions.html#type) | [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Returns `True` if an object or a type is a tensordict or a tensorclass.

A tensor, a dict and any other object return `False`.

Parameters:

**datatype** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*or*[*object*](https://docs.python.org/3/builtins/functions.html#object)) - the type to check, or an object whose type
is checked.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, TensorDictBase, is_tensor_collection, tensorclass
>>> is_tensor_collection(TensorDictBase)
True
>>> is_tensor_collection(TensorDict())
True
>>> @tensorclass
... class MyClass:
... pass
...
>>> is_tensor_collection(MyClass)
True
>>> is_tensor_collection(MyClass(batch_size=[]))
True
>>> is_tensor_collection(torch.zeros(3)), is_tensor_collection({"a": 0})
(False, False)
```