# set_lazy_legacy

*class*tensordict.set_lazy_legacy(*mode: [bool](https://docs.python.org/3/builtins/functions.html#bool)*)

Turns the legacy lazy mode on or off.

In the legacy lazy mode, [`unsqueeze()`](tensordict.TensorDictBase.unsqueeze.html#tensordict.TensorDictBase.unsqueeze),
[`squeeze()`](tensordict.TensorDictBase.squeeze.html#tensordict.TensorDictBase.squeeze),
[`view()`](tensordict.TensorDictBase.view.html#tensordict.TensorDictBase.view),
[`transpose()`](tensordict.TensorDictBase.transpose.html#tensordict.TensorDictBase.transpose) and
[`permute()`](tensordict.TensorDictBase.permute.html#tensordict.TensorDictBase.permute) return lazy views of the
tensordict instead of tensordicts, and [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) returns a lazy
stack. The mode is off by default, and the `LAZY_LEGACY_OP`
environment variable sets it at import. Use it as a context manager or a
decorator, or call `set_lazy_legacy(mode).set()` to change the mode for
the rest of the program.

Deprecated since version 0.15: The legacy lazy mode (`set_lazy_legacy(True)`) is deprecated and will
be removed in TensorDict 0.17. `set_lazy_legacy(False)` will then do
nothing. Use [`lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) to build a lazy stack.

Parameters:

**mode** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - if `True`, turns the legacy lazy mode on, and emits a
`DeprecationWarning`.