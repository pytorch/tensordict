# TensorDictBase.to_parquet

TensorDictBase.to_parquet(*path*, ***, *separator: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***kwargs*)

Writes this TensorDict to a Parquet file.

Requires either pyarrow or pandas to be installed.

Parameters:

**path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - Path to the output Parquet file.

Keyword Arguments:

- **separator** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - If provided, nested keys are joined
with this separator. Defaults to `None`.
- ****kwargs** - Additional keyword arguments forwarded to the Parquet
writer.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).