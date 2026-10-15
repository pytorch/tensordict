# TensorDictBase.fromkeys

*classmethod*TensorDictBase.fromkeys(*keys: [List](https://docs.python.org/3/library/typing.html#typing.List)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)]*, *value: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = 0*)

Creates a tensordict from a list of keys and a single value.

Parameters:

- **keys** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - An iterable specifying the keys of the new dictionary.
- **value** (*compatible type**,**optional*) - The value for all keys. Defaults to `0`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).