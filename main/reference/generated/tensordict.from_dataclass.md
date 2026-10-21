# from_dataclass

tensordict.from_dataclass(*obj: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***, *dest_cls: [Type](https://docs.python.org/3/library/typing.html#typing.Type) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *frozen: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *autocast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *nocast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *shadow: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *tensor_only: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Converts a dataclass instance to a tensorclass instance, or a dataclass type to a tensorclass type.

Use it to move code from `@dataclass` to a tensorclass. For a type,
the result is a new tensorclass named `<name>_tc` with the same fields,
or the type itself, made a tensorclass, if `inplace=True`. A type that
is already a tensorclass is returned as it is. For an instance, the
result is an instance of a new `<name>_tc` tensorclass, or of
`dest_cls`, built from the fields of `obj`.

[`TensorDict.from_dataclass`](tensordict.TensorDictBase.from_dataclass.html#tensordict.TensorDictBase.from_dataclass)
converts a dataclass instance to a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict)
instead.

Parameters:

**obj** (*Any*) - the dataclass instance or type to convert.

Keyword Arguments:

- **dest_cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*,**optional*) - for an instance, the tensorclass to build.
When it is given, the class flags below have no effect. Not used
for a type. Defaults to `None`, which creates a new tensorclass.
- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - for an instance, if `True`, sets
the batch size to the largest leading shape that all the entries
share. Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without `auto_batch_size`,
with a `FutureWarning` when it would change the result.
Starting with TensorDict 0.17, the default will change to
turning on `auto_batch_size` when `batch_dims` is given, as
`from_dict` does. Defaults to `None`, which allows as many as
the entries share.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - for an instance, the
batch size of the result: the leading dimensions that all its
entries share. Defaults to `None`, which gives an empty batch
size.
- **frozen** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `frozen` class flag: if `True`,
assigning a field raises `dataclasses.FrozenInstanceError`.
Defaults to `False`.
- **autocast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `autocast` class flag: if `True`, a
value set to a field is converted to the annotated type. Defaults
to `False`.
- **nocast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `nocast` class flag: if `True`,
tensor-compatible values (`int`, `float`, `np.ndarray`) are
stored as they are, not cast to tensors. Defaults to `False`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - for a type, if `True`, turns `obj`
itself into a tensorclass instead of creating a new class. Not
used for an instance. Defaults to `False`.
- **shadow** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `shadow` class flag: if `True`,
fields can have the name of a tensorclass member (`batch_size`,
`sum`, ...) and hide it. Defaults to `False`.
- **tensor_only** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `tensor_only` class flag: if
`True`, every field annotation must be a tensor type or a tensor
collection, and reading a field is faster. Defaults to `False`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - for an instance, the
device of the result. Defaults to `None`, which leaves each leaf
where it is.

The class flags are described in [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass). Pass
at most one of `autocast`, `nocast` and `tensor_only`, as for
[`tensorclass()`](tensordict.tensorclass.html#tensordict.tensorclass). When the flags are applied (not with
`dest_cls`, and not for a type that is already a tensorclass),
`autocast` with `nocast`, or, for a type, `tensor_only` with either
of them, emits a `FutureWarning` today. Starting with TensorDict 0.17,
the default will change to raising the `TypeError` that
[`tensorclass()`](tensordict.tensorclass.html#tensordict.tensorclass) raises.

Returns:

A tensorclass type if `obj` is a type, a tensorclass instance
otherwise.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `obj` is not a dataclass instance or type, or if, for
 an instance, `tensor_only` is combined with `autocast` or
 `nocast`, or `auto_batch_size=True` with `batch_size`.

Examples

```
>>> from dataclasses import dataclass
>>> import torch
>>> from tensordict.tensorclass import from_dataclass
>>>
>>> @dataclass
... class X:
... a: int
... b: torch.Tensor
...
>>> x = X(0, 0)
>>> x2 = from_dataclass(x)
>>> print(x2)
X_tc(batch_size=[], fields={a: i64[], b: i64[]})
>>> X2 = from_dataclass(X, autocast=True)
>>> print(X2(a=0, b=0))
X_tc(batch_size=[], fields={a: int, b: i64[]})
```