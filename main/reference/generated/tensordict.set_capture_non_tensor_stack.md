# set_capture_non_tensor_stack

*class*tensordict.set_capture_non_tensor_stack(*mode: [bool](https://docs.python.org/3/builtins/functions.html#bool)*)

Controls whether identical non-tensor data should be captured when stacked.

Deprecated since version 0.15: Capturing identical non-tensor data (`set_capture_non_tensor_stack(True)`
or `CAPTURE_NONTENSOR_STACK=1`) is deprecated and will be removed in
TensorDict 0.17. Use [`NonTensorStack.data`](tensordict.NonTensorStack.html#tensordict.NonTensorStack.data)
to get the single value of a stack of identical values.
`set_capture_non_tensor_stack(False)`, the default, does not warn.