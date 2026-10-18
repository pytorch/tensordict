# set_capture_non_tensor_stack

*class*tensordict.set_capture_non_tensor_stack(*mode: [bool](https://docs.python.org/3/builtins/functions.html#bool)*)

Sets whether stacking identical non-tensor data gives one value instead of a stack.

By default, [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) of [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData)
returns a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack), even when all the values
are equal. With `mode=True`, a stack of equal values gives a single
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData). The `CAPTURE_NONTENSOR_STACK`
environment variable sets the mode at import. Use it as a context manager
or a decorator, or call `set_capture_non_tensor_stack(mode).set()` to
change the mode for the rest of the program.

Deprecated since version 0.15: Capturing identical non-tensor data (`set_capture_non_tensor_stack(True)`
or `CAPTURE_NONTENSOR_STACK=1`) is deprecated and will be removed in
TensorDict 0.17. Use [`NonTensorStack.data`](tensordict.NonTensorStack.html#tensordict.NonTensorStack.data)
to get the single value of a stack of identical values.
`set_capture_non_tensor_stack(False)`, the default, does not warn,
and will keep working, doing nothing, after the removal.

Parameters:

**mode** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, stacking identical non-tensor data gives one
value, and the call emits a `DeprecationWarning`.

Examples

```
>>> import torch
>>> from tensordict import NonTensorData, set_capture_non_tensor_stack
>>> with set_capture_non_tensor_stack(False):
... stack = torch.stack([NonTensorData("a"), NonTensorData("a")])
>>> type(stack).__name__, stack.data
('NonTensorStack', 'a')
```