# TensorDictBase.to_csv

TensorDictBase.to_csv(*path*, ***, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*)

Writes this TensorDict to a CSV file.

Requires pandas to be installed.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - Path to the output CSV file.

Keyword Arguments:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - If provided, nested keys are joined
with this separator. Defaults to `None`.
- ****kwargs** - Additional keyword arguments forwarded to
`pandas.DataFrame.to_csv`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).