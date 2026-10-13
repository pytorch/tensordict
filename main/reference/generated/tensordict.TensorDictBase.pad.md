# TensorDictBase.pad

TensorDictBase.pad(*pad_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)]*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.0*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *safe: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Pads this tensordict along the batch dimensions with a constant value.

Parameters:

- **pad_size** (*Sequence**[*[*int*](https://docs.python.org/3/builtins/functions.html#int)*]*) - The padding size, applied to the batch
dimensions starting from the first. For a tensordict of
`ndim` batch dimensions, `pad_size` is a sequence of even
length up to `2 * ndim` formatted as
`(dim0_left, dim0_right, dim1_left, dim1_right, ...)`.
- **value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - The fill value. Defaults to `0.0`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

If `True`, this tensordict's object
identity and key set are preserved; each leaf storage is
replaced by its padded counterpart one leaf at a time, keeping
peak memory close to `1x` instead of `2x`. The leaf
tensors themselves are still freshly allocated because
padding necessarily grows shapes. Defaults to `False`.

Warning

If `inplace=True` and a later leaf's pad fails (e.g.
OOM), the tensordict is left in an inconsistent state.
`safe=True` (the default) catches the user-error class
of failures before any mutation occurs.
- **safe** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - If `True`, validate that the operation
would succeed for every leaf before any mutation occurs. Set
to `False` to skip the pre-flight walk when the inputs are
known to be valid. Defaults to `True`.

Returns:

The padded tensordict. When `inplace=True` this is `self`.

Note

On a lazy stack whose tensordicts have no names, the result of
`inplace=False` has no names, even when the stack dimension has
a name. See [`pad()`](tensordict.pad.html#tensordict.pad).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({"a": torch.zeros(3, 4)}, batch_size=[3, 4])
>>> td.pad([0, 0, 0, 1]).batch_size
torch.Size([3, 5])
>>> td.pad([0, 0, 0, 1], inplace=True) is td
True
>>> td.batch_size
torch.Size([3, 5])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).