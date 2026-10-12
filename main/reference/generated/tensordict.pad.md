# pad

tensordict.pad(*tensordict: T*, *pad_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)]*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.0*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *safe: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → T

Pads all tensors in a tensordict along the batch dimensions with a constant value.

Use it to grow the batch dimensions of one tensordict, for example to
bring samples to a common size before you stack them. To pad and stack a
list of tensordicts in one call, use [`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence)
(see [Stacking and variable-size data](../../concepts/stacking.html#stacking)).

Parameters:

- **tensordict** ([*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)) - The tensordict to pad
- **pad_size** (*Sequence**[*[*int*](https://docs.python.org/3/builtins/functions.html#int)*]*) - The padding size by which to pad some batch
dimensions of the tensordict, starting from the first dimension and
moving forward. [len(pad_size) / 2] dimensions of the batch size will
be padded. For example to pad only the first dimension, pad has the form
(padding_left, padding_right). To pad two dimensions,
(padding_left, padding_right, padding_top, padding_bottom) and so on.
pad_size must be even and less than or equal to twice the number of batch dimensions.
- **value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - The fill value to pad by, default 0.0.
Non-tensor entries ([`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData) /
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack)) keep their values at the
valid positions and hold `None` in the pad slots instead of
`value`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

If `True`, the input tensordict's identity
and key set are preserved, and each leaf's storage is replaced by
its padded counterpart one at a time. This keeps peak memory close
to the size of the tensordict itself rather than 2x (the case when
a fresh tensordict is allocated alongside the original). The leaf
tensors themselves are still freshly allocated (`pad` necessarily
grows shapes), so this is not a same-storage operation. Defaults to
`False`.

On [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict), `inplace=True` pads
each constituent tensordict along the non-stack dimensions in place
and grows the stack along the stack dimension by appending or
prepending zero-filled copies of the edge constituents; the lazy
stack's identity is preserved.

Warning

If `inplace=True` and the operation fails partway through
(for example an out-of-memory error during a leaf's padding),
the tensordict is left in an inconsistent state: some leaves
will have the new shape and others the old, and the
`batch_size` will not have been updated. Restoring the
original state would require keeping every old leaf alive
until the whole pass succeeded, which would defeat the 1x
memory contract. Use `safe=True` (the default) to catch
the realistic user-error class of failures before any
mutation happens.
- **safe** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - If `True`, validate that the operation
would succeed for every leaf before any mutation occurs. This
catches errors such as negative pad widths that exceed a leaf's
dimension size, leaves that are not paddable, etc., raising
before any in-place rebind. Set to `False` to skip the
pre-flight walk for a small speedup when the inputs are known
to be valid. Defaults to `True`.

Returns:

The padded tensordict. When `inplace=True` this is the same object as
the input; otherwise a new tensordict.

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