# remove_duplicates

tensordict.utils.remove_duplicates(*input: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *return_indices: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Removes indices duplicated in key along the specified dimension.

This method detects duplicate elements in the tensor associated with the specified key along the specified
dim and removes elements in the same indices in all other tensors within the TensorDict. It is expected for
dim to be one of the dimensions within the batch size of the input TensorDict to ensure consistency in all
tensors. Otherwise, an error will be raised.

Parameters:

- **input** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - The TensorDict containing potentially duplicate elements.
- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - The key of the tensor along which duplicate elements should be identified and removed. It
must be one of the leaf keys within the TensorDict, pointing to a tensor and not to another TensorDict.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The dimension along which duplicate elements should be identified and removed. It must be one of
the dimensions within the batch size of the input TensorDict. Defaults to `0`.
- **return_indices** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - If `True`, the indices of the unique elements in the input tensor will be
returned as well. Defaults to `False`.

Returns:

input tensordict with the indices corrsponding to duplicated elements

in tensor key along dimension dim removed.

unique_indices (torch.Tensor, optional): The indices of the first occurrences of the unique elements in the

input tensordict for the specified key along the specified dim. Only provided if return_indices is True.

Return type:

output ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase))

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.utils import remove_duplicates
>>> td = TensorDict(
... {
... "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]),
... "tensor2": torch.tensor([[10, 20], [30, 40], [40, 50], [50, 60]]),
... },
... batch_size=[4],
... )
>>> output_tensordict = remove_duplicates(td, key="tensor1", dim=0)
>>> expected_output = TensorDict(
... {
... "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
... "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
... },
... batch_size=[3],
... )
>>> assert (output_tensordict == expected_output).all()
```