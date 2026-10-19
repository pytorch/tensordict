# recv_tensordict

*async*tensordict._ucxx.recv_tensordict(*endpoint*, *td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Receives a tensordict that [`send_tensordict()`](tensordict._ucxx.send_tensordict.html#tensordict._ucxx.send_tensordict) sends over a UCXX endpoint.

Warning

`recv_tensordict` is experimental: its API can change in a minor
release without a deprecation cycle.

It keeps no state: each call reads the metadata, then the storage. It
needs the optional `ucxx` package. The sender controls the size of the
allocation: use it only on a trusted network.

Parameters:

- **endpoint** (*ucxx.Endpoint*) - the endpoint to receive from.
- **td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a consolidated tensordict to receive
into. If its layout, storage size and device match the data, the
data is written into its storage, and `td` is returned. Otherwise
`td` is not changed, and a new tensordict is returned. Defaults to
`None`.

Keyword Arguments:

- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device to receive on.
Defaults to `None`, which uses the device of the storage of `td`
if it is consolidated, and the CPU otherwise. With
`consolidated=False`, the default is the CPU.
- **consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `consolidated` argument of the
sender's [`send_tensordict()`](tensordict._ucxx.send_tensordict.html#tensordict._ucxx.send_tensordict). With `False`, reads the
leaves one by one, as a pipe with `consolidated=False` does, and
returns a new tensordict: `td` is not used. Defaults to `True`.

Returns:

`td` or a new consolidated tensordict, or
with `consolidated=False`, a new tensordict that is not consolidated.

Return type:

The received tensordict

Raises:

[**ImportError**](https://docs.python.org/3/builtins/exceptions.html#ImportError) - if `ucxx` is not installed.