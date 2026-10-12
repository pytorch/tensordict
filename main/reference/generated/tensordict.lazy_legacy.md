# lazy_legacy

tensordict.lazy_legacy(*allow_none=False*)

Returns True if lazy representations will be used for selected methods.

Parameters:

**allow_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and the option was never
set, `None` is returned instead of the default (`False`).
Defaults to `False`.

Deprecated since version 0.15: The legacy lazy mode is deprecated and will be removed in TensorDict 0.17,
see [`set_lazy_legacy`](tensordict.set_lazy_legacy.html#tensordict.set_lazy_legacy).