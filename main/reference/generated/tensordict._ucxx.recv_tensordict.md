# recv_tensordict

*async*tensordict._ucxx.recv_tensordict(*endpoint*, *td: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Receive a TensorDict from a UCXX endpoint (stateless, always expects metadata).

Warning

`recv_tensordict` is experimental: its API can change in a minor
release without a deprecation cycle.

Parameters:

- **endpoint** - a UCXX endpoint object.
- **td** - if provided and consolidated with matching layout, storage size,
and device, receives directly into its storage buffer (in-place
update). Otherwise returns a new TensorDict without modifying `td`.

Keyword Arguments:

**device** - device on which to receive. If omitted, uses the storage
device of a consolidated `td`, even when its layout does not
match. Otherwise defaults to CPU.

Returns:

The received TensorDict.