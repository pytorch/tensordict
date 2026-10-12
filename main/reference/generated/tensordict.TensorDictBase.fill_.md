# TensorDictBase.fill

TensorDictBase.fill_(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) | [bool](https://docs.python.org/3/builtins/functions.html#bool)*) → Self

Fills a tensor pointed by the key with a given scalar value.

Parameters:

- **key** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**nested key*) - entry to be filled.
- **value** (*Number**or*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - value to use for the filling.

Returns:

self

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).