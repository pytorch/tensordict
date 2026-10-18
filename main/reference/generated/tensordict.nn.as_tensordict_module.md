# as_tensordict_module

*class*tensordict.nn.as_tensordict_module(***, *in_keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *out_keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Turns a function or a method into a tensordict module that reads `in_keys` and writes `out_keys`.

The decorated function is wrapped in a
[`TensorDictModule`](tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule), built at the first call.
Called with a tensordict, it returns the tensordict with the outputs
written at `out_keys`. Called with tensors (see
[`dispatch()`](tensordict.nn.dispatch.html#tensordict.nn.dispatch)), it returns the output tensors. On a
method, one module serves all the instances of the class, and the
instance is passed as `self` at each call.

Keyword Arguments:

- **in_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys of the entries
passed to the function, in the order of its arguments.
- **out_keys** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys where the outputs
of the function are written, one key per output.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import as_tensordict_module
>>> @as_tensordict_module(in_keys="c", out_keys="d")
... def my_function(c):
... return c + 1
>>> my_function(TensorDict(c=0))["d"]
tensor(1)
>>> my_function(c=torch.zeros(2))
tensor([1., 1.])
```

It works on methods too:

```
>>> class MyClass:
... @as_tensordict_module(in_keys="c", out_keys="d")
... def my_method(self, c):
... return c + 1
>>> MyClass().my_method(TensorDict(c=0))["d"]
tensor(1)
```