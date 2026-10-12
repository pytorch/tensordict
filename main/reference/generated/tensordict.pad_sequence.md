# pad_sequence

tensordict.pad_sequence(*list_of_tensordicts: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[T]*, *pad_dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *padding_value: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.0*, *out: T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_mask: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) = False*) → T

Pads a list of tensordicts in order for them to be stacked together in a contiguous format.

Use it to batch samples whose entries differ in size along one
dimension, such as sequences of different lengths. Every entry is padded
along `pad_dim` to the largest size among the inputs, then the inputs
are stacked along a new first dimension. The inputs must have the same
keys, and the entry shapes must match in every dimension other than
`pad_dim`. To keep the inputs unpadded, use
[`lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) instead (see [Stacking and variable-size data](../../concepts/stacking.html#stacking)).

Parameters:

- **list_of_tensordicts** (*List**[*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*]*) - the list of instances to pad and stack.
- **pad_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension of the entries to pad, counted
from the first dimension of each entry (batch dimensions
included). If it is a batch dimension, the batch size of the
result is padded too. Defaults to `0`.
- **padding_value** (*number**,**optional*) - the padding value. Defaults to `0.0`.
Non-tensor entries are not filled with `padding_value`: when the padded
dim is a batch dimension of the output, they keep their values at the
valid positions and hold `None` in the pad slots, as with
[`pad()`](tensordict.pad.html#tensordict.pad).
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - if provided, the destination where the data will be
written.
- **return_mask** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - if `True`, the result has
a `"masks"` entry. If `return_mask` is a nested key (a string
or a tuple of strings), the masks are stored under that key. The
masks entry is a tensordict with the structure of the stacked
tensordict. Each of its entries is the mask of the valid values,
of size `torch.Size([stack_len, *new_shape])`, where
`new_shape[pad_dim]` is the largest size and the rest of
`new_shape` matches the shape of the input entries. Not
available for tensorclass inputs.

Returns:

a [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) (or the tensorclass of the inputs)
with batch size `[len(list_of_tensordicts), *batch_size]`.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `list_of_tensordicts` is empty, or if the entry
 shapes differ in a dimension other than `pad_dim`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, pad_sequence
>>> list_td = [
... TensorDict({"a": torch.zeros((3, 8)), "b": torch.zeros((6, 8))}, batch_size=[]),
... TensorDict({"a": torch.zeros((5, 8)), "b": torch.zeros((6, 8))}, batch_size=[]),
... ]
>>> padded_td = pad_sequence(list_td, return_mask=True)
>>> print(padded_td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([2, 5, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([2, 6, 8]), device=cpu, dtype=torch.float32, is_shared=False),
 masks: TensorDict(
 fields={
 a: Tensor(shape=torch.Size([2, 5]), device=cpu, dtype=torch.bool, is_shared=False),
 b: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
```