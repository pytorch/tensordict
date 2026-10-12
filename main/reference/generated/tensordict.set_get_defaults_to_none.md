# set_get_defaults_to_none

tensordict.set_get_defaults_to_none(*set_to_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*)

Sets the default of get to None and silences deprecation warnings during calls to get that result in a KeyError.

This can also be controlled via the environment variable `TD_GET_DEFAULTS_TO_NONE`.

Deprecated since version 0.15: Since v0.7, [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) returns `None` for a
missing key by default. This function, and setting `TD_GET_DEFAULTS_TO_NONE`
to a false value, are deprecated and will be removed in TensorDict 0.17.
Use `td[key]` to raise a `KeyError` for a missing key.

Parameters:

**set_to_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - whether the default of get should be None, or should get raise a KeyError if
no default is passed and the key is absent from the TensorDict.
Defaults to True.