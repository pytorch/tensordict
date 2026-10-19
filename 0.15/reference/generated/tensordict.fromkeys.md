# fromkeys

tensordict.fromkeys(*keys: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*, *value: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = 0*)

Creates a tensordict with the same value for every key, like [`dict.fromkeys()`](https://docs.python.org/3/builtins/stdtypes.html#dict.fromkeys).

Same as [`tensordict.TensorDict.fromkeys()`](tensordict.TensorDictBase.fromkeys.html#tensordict.TensorDictBase.fromkeys), and as [`dict.fromkeys()`](https://docs.python.org/3/builtins/stdtypes.html#dict.fromkeys)
for a dict. The batch size of the result is `[]`.

Parameters:

- **keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the keys of the new tensordict. A tuple of
strings creates a nested entry.
- **value** ([*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**number**or**Any**,**optional*) - the value of every
entry. Defaults to `0`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict).

Examples

```
>>> import tensordict
>>> td = tensordict.fromkeys(["a", ("b", "c")], 1)
>>> td["b", "c"]
tensor(1)
```