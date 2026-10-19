# set_get_defaults_to_none

tensordict.set_get_defaults_to_none(*set_to_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Sets whether [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns `None` or raises for a missing key.

The setting applies to every call of `get` and `get_at` that passes no
default. The environment variable `TD_GET_DEFAULTS_TO_NONE` sets its
initial value.

Deprecated since version 0.15: Since v0.7, [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns `None` for a
missing key by default. This function, and setting `TD_GET_DEFAULTS_TO_NONE`
to a false value, are deprecated and will be removed in TensorDict 0.17.
Use `td[key]` to raise a `KeyError` for a missing key.

Parameters:

**set_to_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, `get` returns `None` for
a missing key when no default is passed. If `False`, it raises a
`KeyError`. Defaults to `True`.