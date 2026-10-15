# is_tensor_collection

tensordict.is_tensor_collection(*datatype: [type](https://docs.python.org/3/builtins/functions.html#type) | [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Checks if a data object or a type is a tensor container from the tensordict lib.

Returns:

`True` if the input is a TensorDictBase subclass, a tensorclass or an istance of these.
`False` otherwise.

Examples

```
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
```