# pad_sequence

tensordict.pad_sequence(*list_of_tensordicts: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[T]*, *pad_dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *padding_value: [float](https://docs.python.org/3/builtins/functions.html#float) = 0.0*, *out: T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_mask: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) = False*, *padding_side: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'right'*) → T

Pads tensordicts to a common size along one dimension and stacks them densely.

Use it to batch samples whose entries differ in size along one
dimension, such as sequences of different lengths. Every entry is padded
along `pad_dim` to the largest size among the inputs, then the inputs
are stacked along a new first dimension. The inputs must have the same
keys, and the entry shapes must match in every dimension other than
`pad_dim`. To keep the inputs unpadded, use
[`lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) instead (see [Stacking and variable-size data](../../concepts/stacking.html#stacking)).

Parameters:

- **list_of_tensordicts** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instances*) - the tensordicts to pad and stack.
- **pad_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension of the entries to pad, counted
from the first dimension of each entry (batch dimensions
included). If it is a batch dimension, the batch size of the
result is padded too. A negative `pad_dim` counts from the last
dimension of each entry. If it lands on a batch dimension of some
entries and on another dimension of others, the batch size is not
padded, and, unless `out` is given, no entry may be longer along
it than the batch of the first sample. Defaults to `0`.
- **padding_value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the value of the padded elements.
When `pad_dim` is a batch dimension, non-tensor entries hold
`None` there instead, as with [`pad()`](tensordict.pad.html#tensordict.pad). Defaults
to `0.0`.
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into. Defaults to `None`.
- **return_mask** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*or*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - if `True`, the result has
a `"masks"` entry. If `return_mask` is a nested key (a string
or a tuple of strings), the masks are stored under that key. The
masks entry is a tensordict with the structure of the stacked
tensordict. Each of its entries is the mask of the valid values.
If `pad_dim` is a batch dimension, the mask has the output's
batch size. Otherwise, its shape is the output's batch size
followed by the padded length, regardless of the feature
dimension's position. Tensorclass inputs raise a
`RuntimeError`. Use [`unpad_sequence()`](tensordict.unpad_sequence.html#tensordict.unpad_sequence) to turn
the result back into the variable-length samples. Defaults to
`False`.
- **padding_side** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - `"right"` puts the padding after the
data along `pad_dim`, `"left"` puts it before, as
`get(key, as_padded_tensor=True, padding_side=...)` does on a
lazy stack. The masks mark the data positions on either side.
Defaults to `"right"`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) (or the tensorclass of the inputs)
with batch size `[len(list_of_tensordicts), *batch_size]`.

Raises:

- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if `list_of_tensordicts` is empty, if the entry
 shapes differ in a dimension other than `pad_dim`, or if
 `return_mask` is set for tensorclass inputs.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `padding_side` is neither `"left"` nor `"right"`,
 or if a negative `pad_dim` lands on a batch dimension of some
 entries and on another dimension of others, and, with no `out`,
 an entry is longer along it than the batch of the first sample.
- [**IndexError**](https://docs.python.org/3/builtins/exceptions.html#IndexError) - if `pad_dim` is out of range for an entry.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, pad_sequence
>>> list_td = [
... TensorDict({"a": torch.zeros((3, 8)), "b": torch.zeros((6, 8))}, batch_size=[]),
... TensorDict({"a": torch.zeros((5, 8)), "b": torch.zeros((6, 8))}, batch_size=[]),
... ]
>>> padded_td = pad_sequence(list_td, return_mask=True)
>>> padded_td["a"].shape, padded_td["b"].shape
(torch.Size([2, 5, 8]), torch.Size([2, 6, 8]))
>>> padded_td["masks", "a"].sum(-1)
tensor([3, 5])
```