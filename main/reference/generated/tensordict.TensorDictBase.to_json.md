# TensorDictBase.to_json

TensorDictBase.to_json(*path*, ***, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lines: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***kwargs*)

Writes this TensorDict to a JSON file.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - Path to the output JSON file.

Keyword Arguments:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - If provided, nested keys are joined
with this separator. Defaults to `None`.
- **lines** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - If `True`, writes in JSON Lines format.
Defaults to `False`.
- ****kwargs** - Additional keyword arguments forwarded to the JSON
writer.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).