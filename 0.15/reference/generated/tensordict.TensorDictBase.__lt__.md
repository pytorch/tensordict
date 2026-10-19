# TensorDictBase.__lt__

TensorDictBase.__lt__(*other: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → Self | [bool](https://docs.python.org/3/builtins/functions.html#bool)

Compares the tensordict with `other` leaf by leaf with `<`.

Works as [`__eq__()`](tensordict.TensorDictBase.__eq__.html#tensordict.TensorDictBase.__eq__), except that an `other` of another type,
such as `None`, gives `False` and a `FutureWarning`. Starting
with TensorDict 0.17, the default will change to returning
`NotImplemented`, so that Python raises a `TypeError`, as it does
for tensors and for `&`.

Parameters:

**other** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or**Number*) - the value to
compare with.

Returns:

A new tensordict of boolean leaves, or `False`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).