# send_tensordict

*async*tensordict._ucxx.send_tensordict(*endpoint*, *td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Sends a tensordict over a UCXX endpoint, with its metadata, in one call.

Warning

`send_tensordict` is experimental: its API can change in a minor
release without a deprecation cycle.

Unlike a [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe), it keeps no state: each call sends the
metadata (as JSON) and then the consolidated storage. The receiver calls
[`recv_tensordict()`](tensordict._ucxx.recv_tensordict.html#tensordict._ucxx.recv_tensordict). It needs the optional `ucxx` package. The
protocol does not authenticate the peer: use it only on a trusted network.

Parameters:

- **endpoint** (*ucxx.Endpoint*) - the endpoint to send through, for example
from `ucxx.create_endpoint`.
- **td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to send.

Keyword Arguments:

**consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, consolidates the tensordict
(see [`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)) unless it is
already, and sends its storage as one buffer. If `False`, sends the
tensor leaves of the tensordict, nested ones included, one by one, in
the format of a pipe with `consolidated=False`: the receiver calls
[`recv_tensordict()`](tensordict._ucxx.recv_tensordict.html#tensordict._ucxx.recv_tensordict) with `consolidated=False`. Defaults to
`True`.

Raises:

- [**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) - if `ucxx` is not installed.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - with `consolidated=False`, if the tensordict holds a
 non-tensor entry other than a [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) or
 [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) whose values are `str`,
 `int`, `float`, `bool`, `None`, or lists or dicts (with
 `str` keys) of them. Send it with `consolidated=True`.