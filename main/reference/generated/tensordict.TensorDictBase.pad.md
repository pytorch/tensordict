# TensorDictBase.pad

TensorDictBase.pad(*pad_size: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[int](https://docs.python.org/3/builtins/functions.html#int)]*, *value: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.0*, *inplace: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *safe: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Returns the tensordict padded along its batch dimensions with a constant value.

Parameters:

- **pad_size** (*sequence**of*[*int*](https://docs.python.org/3/builtins/functions.html#int)) - the padding, applied to the batch
dimensions starting from the first. For a tensordict with
`ndim` batch dimensions, it is a sequence of even length up
to `2 * ndim`:
`(dim0_left, dim0_right, dim1_left, dim1_right, ...)`.
- **value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the fill value. Defaults to `0.0`.
- **inplace** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - 

if `True`, the tensordict keeps its
identity and keys, its leaves are replaced one by one by
their padded copies, and the call returns `self`. Peak memory
stays close to `1x` instead of `2x`. See
[In-place operations](../../concepts/operations.html#inplace-conventions). Defaults to `False`.

Warning

If `inplace=True` and the padding of a later leaf fails
(for example, out of memory), the tensordict is left in an
inconsistent state. `safe=True` catches the user errors
before any leaf is changed.
- **safe** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, checks that the padding works
for every leaf before any leaf is changed. Pass `False` to
skip this check when the inputs are known to be valid.
Defaults to `True`.

Returns:

The padded tensordict, or `self` with `inplace=True`.

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