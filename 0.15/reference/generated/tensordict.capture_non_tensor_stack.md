# capture_non_tensor_stack

tensordict.capture_non_tensor_stack(*allow_none=False*)

Returns whether stacking identical non-tensor data gives one value instead of a stack.

See [`set_capture_non_tensor_stack`](tensordict.set_capture_non_tensor_stack.html#tensordict.set_capture_non_tensor_stack).

Parameters:

**allow_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and the option was never
set, returns `None` instead of the default (`False`).
Defaults to `False`.

Examples

```
>>> from tensordict import capture_non_tensor_stack
>>> capture_non_tensor_stack()
False
```