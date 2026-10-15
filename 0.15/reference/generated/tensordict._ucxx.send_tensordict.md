# send_tensordict

*async*tensordict._ucxx.send_tensordict(*endpoint*, *td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Send a TensorDict over a UCXX endpoint (stateless, always sends metadata).

Warning

`send_tensordict` is experimental: its API can change in a minor
release without a deprecation cycle.

Parameters:

- **endpoint** - a UCXX endpoint object.
- **td** - the TensorDict to send.

Keyword Arguments:

**consolidated** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if `True`, consolidates into a single buffer
before sending. Defaults to `True`.