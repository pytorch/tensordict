# get_defaults_to_none

tensordict.get_defaults_to_none(*set_to_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Returns whether [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns `None` for a missing key.

`False` means that `get` raises a `KeyError` when no default is
passed. See [`set_get_defaults_to_none()`](tensordict.set_get_defaults_to_none.html#tensordict.set_get_defaults_to_none).

Deprecated since version 0.15: Since v0.7, [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns `None` for a
missing key by default. This function is deprecated and will be removed in
TensorDict 0.17. Use `td[key]` to raise a `KeyError` for a missing key.

Parameters:

**set_to_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - ignored. Defaults to `True`.