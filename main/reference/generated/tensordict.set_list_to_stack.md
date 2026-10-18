# set_list_to_stack

*class*tensordict.set_list_to_stack(*mode: [bool](https://docs.python.org/3/builtins/functions.html#bool)*)

Sets whether assigning a list to a key stores a stack of non-tensor data.

By default (`mode=True`), `td[key] = ["a", "b"]` stores a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) with one value per element along the
first batch dimension. With `mode=False`, the list is stored as one
[`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) value. The `LIST_TO_STACK`
environment variable sets the mode at import. Use it as a context manager
or a decorator, or call `set_list_to_stack(mode).set()` to change the
mode for the rest of the program.

Deprecated since version 0.15: Turning list-to-stack conversion off (`set_list_to_stack(False)` or
`LIST_TO_STACK=0`) is deprecated and will be removed in TensorDict 0.17.
Use [`td.set_non_tensor(key, value)`](tensordict.TensorDictBase.set_non_tensor.html#tensordict.TensorDictBase.set_non_tensor)
to store a list as one value. `set_list_to_stack(True)`, the default,
does not warn, and will keep working, doing nothing, after the removal.

Parameters:

**mode** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, a list is stored as a stack. `False`
stores it as one value, and emits a `DeprecationWarning`.

Examples

```
>>> from tensordict import TensorDict, set_list_to_stack
>>> td = TensorDict(batch_size=[2])
>>> with set_list_to_stack(True):
... td["x"] = ["a", "b"]
>>> type(td.get("x")).__name__, td["x"]
('NonTensorStack', LinkedList(['a', 'b']))
```