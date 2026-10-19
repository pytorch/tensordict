# TensorDictBase.__xor__

TensorDictBase.__xor__(*other: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → Self | [bool](https://docs.python.org/3/builtins/functions.html#bool)

Computes `self ^ other` leaf by leaf and returns a new tensordict.

The operation is a logical XOR for boolean leaves and a bitwise XOR for
integer leaves. `other` is paired as in [`__eq__()`](tensordict.TensorDictBase.__eq__.html#tensordict.TensorDictBase.__eq__): a
tensordict or `dict` must have the same keys, or a `KeyError` is
raised. An `other` of another type, such as `None` or a string,
gives `True` and a `FutureWarning`. Starting with TensorDict 0.17,
the default will change to returning `NotImplemented`, so that
Python raises a `TypeError`, as it does for tensors and for `&`.

The result is a new tensordict that is not locked and, except with
a tensor `other` of one or more dims, has no dim names. `&`, `~`
and the arithmetic operators keep the dim names and the lock.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the other
operand.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).