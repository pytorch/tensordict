# TensorDictBase.__or__

TensorDictBase.__or__(*other: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → Self | [bool](https://docs.python.org/3/builtins/functions.html#bool)

Computes `self | other` leaf by leaf and returns a new tensordict.

The operation is a logical OR for boolean leaves and a bitwise OR for
integer leaves. `other` is paired as in [`__eq__()`](tensordict.TensorDictBase.__eq__.html#tensordict.TensorDictBase.__eq__): a
tensordict or `dict` must have the same keys, or a `KeyError` is
raised.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the other
operand.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).