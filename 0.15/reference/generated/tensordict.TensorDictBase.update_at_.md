# TensorDictBase.update_at

TensorDictBase.update_at_(*input_dict_or_td: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)] | T*, *idx: [None](https://docs.python.org/3/builtins/constants.html#None) | [int](https://docs.python.org/3/builtins/functions.html#int) | [slice](https://docs.python.org/3/builtins/functions.html#slice) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), ...]*, *clone: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *keys_to_update: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Copies the entries of a dictionary or tensordict into the existing entries at `idx`.

`td.update_at_(other, idx)` writes each entry as
`td.set_at_(key, other[key], idx)`: the existing tensors keep their
storage, so `update_at_` works on a locked tensordict. Every key of
the input must exist in the tensordict. Tensor and non-tensor leaves
are supported. For a faster copy of tensors only, use
[`copy_at_()`](tensordict.TensorDictBase.copy_at_.html#tensordict.TensorDictBase.copy_at_).

Parameters:

- **input_dict_or_td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the entries to copy.
Their leading dimensions are those of `td[idx]`.
- **idx** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*slice*](https://docs.python.org/3/builtins/functions.html#slice)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)) - the index to write at,
along the batch dimensions.
- **clone** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, clones the tensors of the
input before copying them. Defaults to `False`.

Keyword Arguments:

- **non_blocking** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.
- **keys_to_update** (*sequence**of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - if passed, only
these keys are copied. Defaults to `None`, which copies every
entry.

Returns:

`self`.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if a key of the input is missing from the tensordict.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if a value does not fit the shape of its entry.

Deprecated since version 0.15: The keyword names `index` (for `idx`) and `input_dict` (for
`input_dict_or_td`) warn, and will be removed in TensorDict 0.17.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3, 4, 5), b=torch.zeros(3, 4, 10), batch_size=[3, 4])
>>> other = TensorDict(a=torch.ones(1, 4, 5), b=torch.ones(1, 4, 10), batch_size=[1, 4])
>>> _ = td.update_at_(other, slice(1, 2))
>>> td["a"].sum(dim=(1, 2))
tensor([ 0., 20., 0.])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).