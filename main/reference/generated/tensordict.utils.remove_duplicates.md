# remove_duplicates

tensordict.utils.remove_duplicates(*input: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *return_indices: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Returns a tensordict without the elements whose `key` value repeats an earlier one.

The entry `key` is cut into slices along `dim`. For each group of
equal slices, the first element is kept, and the same indices are kept in
every other entry. The result is sorted by the value of `key`, not in
the original order, and its entries are new tensors.

Parameters:

- **input** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to deduplicate.
- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the tensor that defines duplicates. It
must hold a tensor, not a tensordict.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch dimension along which to look for
duplicates. Defaults to `0`.

Keyword Arguments:

**return_indices** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, also returns, for each
element of `input` along `dim`, the index of its value in the
result. Defaults to `False`.

Returns:

The deduplicated tensordict, or a tuple of the tensordict and the
indices if `return_indices=True`.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if `key` is missing or does not hold a tensor.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `dim` is not a batch dimension.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.utils import remove_duplicates
>>> td = TensorDict(x=torch.tensor([7, 1, 7, 3]), y=torch.arange(4), batch_size=[4])
>>> out, indices = remove_duplicates(td, key="x", return_indices=True)
>>> out["x"], out["y"]
(tensor([1, 3, 7]), tensor([1, 3, 0]))
>>> indices
tensor([2, 0, 2, 1])
```