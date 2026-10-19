# pad

tensordict.pad(*tensordict: T*, *pad_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)]*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.0*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *safe: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → T

Pads all tensors in a tensordict along the batch dimensions with a constant value.

Use it to grow the batch dimensions of one tensordict, for example to
bring samples to a common size before you stack them. To pad and stack a
list of tensordicts in one call, use [`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence)
(see [Stacking and variable-size data](../../concepts/stacking.html#stacking)).

Parameters:

- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instance*) - the tensordict to
pad.
- **pad_size** (*sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the padding before and after each batch
dimension, starting from the first one (unlike
[`torch.nn.functional.pad()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.pad.html#torch.nn.functional.pad), which starts from the last):
`(dim0_before, dim0_after, dim1_before, dim1_after, ...)`. Its
length must be even and at most twice the number of batch
dimensions, or a `RuntimeError` is raised.
- **value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the value of the padded elements. Non-tensor
entries hold `None` there instead. Defaults to `0.0`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one with padded
tensors, and the call returns it. A lazy stack pads each of its
tensordicts in place and grows along the stack dimension with new
tensordicts filled with `value`. See
[In-place operations](../../concepts/operations.html#inplace-conventions). Defaults to `False`.
- **safe** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, checks that every leaf can be
padded before anything is changed, so that a bad `pad_size`
raises before an in-place pad changes some of the leaves. Set it
to `False` to skip the check when the inputs are known to be
valid. Defaults to `True`.

Returns:

The padded tensordict. When `inplace=True` this is the same object as
the input; otherwise a new tensordict, or a new tensorclass of the
same type for a tensorclass input.

Note

With `inplace=False`, a lazy stack gives a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict). If the stacked tensordicts have no
names, it has no names either, even when the stack dimension has a
name (from `stack_dim_name` or the `names` setter). To keep that
name, name the dimension of the result again with
[`refine_names()`](tensordict.TensorDictBase.refine_names.html#tensordict.TensorDictBase.refine_names).

Examples

```
>>> from tensordict import TensorDict, pad
>>> import torch
>>> td = TensorDict({'a': torch.ones(3, 4, 1),
... 'b': torch.ones(3, 4, 1, 1)}, batch_size=[3, 4])
>>> dim0_left, dim0_right, dim1_left, dim1_right = [0, 1, 0, 2]
>>> padded_td = pad(td, [dim0_left, dim0_right, dim1_left, dim1_right], value=0.0)
>>> print(padded_td.batch_size)
torch.Size([4, 6])
>>> print(padded_td.get("a").shape)
torch.Size([4, 6, 1])
>>> print(padded_td.get("b").shape)
torch.Size([4, 6, 1, 1])
```