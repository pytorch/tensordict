# as_tensordict_module

*class*tensordict.nn.as_tensordict_module(***, *in_keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *out_keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A decorator that converts a function into a TensorDictModule.

Keyword Arguments:

- **in_keys** (*List**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**|*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - The input keys of the resulting TensorDictModule.
- **out_keys** (*List**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*]**|*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - The output keys of the resulting TensorDictModule.

Returns:

A decorator that can be applied to a function to convert it into a TensorDictModule.

Return type:

*Callable*

Examples

```
>>> from tensordict import TensorDict
>>> from tensordict.nn import as_tensordict_module
>>> class MyClass:
... @as_tensordict_module(in_keys="c", out_keys="d")
... def my_method(self, c):
... return c + 1
>>> obj = MyClass()
>>> result = obj.my_method(TensorDict(c=0))
>>> print(result["d"])
tensor(1)
>>> @as_tensordict_module(in_keys="c", out_keys="d")
... def my_function(c):
... return c + 1
>>> result = my_function(TensorDict(c=0))
>>> print(result["d"])
tensor(1)
```