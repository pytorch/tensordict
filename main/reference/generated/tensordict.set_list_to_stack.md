# set_list_to_stack

*class*tensordict.set_list_to_stack(*mode: [bool](https://docs.python.org/3/builtins/functions.html#bool)*)

Context manager and decorator to control list handling in TensorDict.

Deprecated since version 0.15: Turning list-to-stack conversion off (`set_list_to_stack(False)` or
`LIST_TO_STACK=0`) is deprecated and will be removed in TensorDict 0.17.
Use [`td.set_non_tensor(key, value)`](tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor)
to store a list as one value. `set_list_to_stack(True)`, the default,
does not warn.