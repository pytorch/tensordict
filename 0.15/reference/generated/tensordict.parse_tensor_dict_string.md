# parse_tensor_dict_string

tensordict.parse_tensor_dict_string(*s: [str](https://docs.python.org/3/builtins/stdtypes.html#str)*)

Parse a TensorDict repr to a TensorDict.

Note

This functions is intended to be used for debugging, to reproduce a tensordict
given its printed version, and should not be used in real applications.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if a field is not a `Tensor` or a `TensorDict` printed
 with the default print options, or if `s` is the repr of a lazy
 stack with exclusive keys.