# lazy_legacy

tensordict.lazy_legacy(*allow_none=False*)

Returns whether the legacy lazy mode is on.

See [`set_lazy_legacy`](tensordict.set_lazy_legacy.html#tensordict.set_lazy_legacy).

Deprecated since version 0.15: The legacy lazy mode is deprecated and will be removed in TensorDict 0.17,
see [`set_lazy_legacy`](tensordict.set_lazy_legacy.html#tensordict.set_lazy_legacy).

Parameters:

**allow_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True` and the option was never
set, returns `None` instead of the default (`False`).
Defaults to `False`.

Examples

```
>>> from tensordict import lazy_legacy
>>> lazy_legacy()
False
```