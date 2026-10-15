# isin

tensordict.utils.isin(*input: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *reference: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*) → Tensor

Tests if each element of `key` in input `dim` is also present in the reference.

This function returns a boolean tensor of length `input.batch_size[dim]` that is `True` for elements in
the entry `key` that are also present in the `reference`. This function assumes that both `input` and
`reference` have the same batch size and contain the specified entry, otherwise an error will be raised.

Parameters:

- **input** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - Input TensorDict.
- **reference** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - Target TensorDict against which to test.
- **key** (*Nestedkey*) - The key to test.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The dimension along which to test. Defaults to `0`.

Returns:

A boolean tensor of length `input.batch_size[dim]` that is `True` for elements in

the `input` `key` tensor that are also present in the `reference`.

Return type:

out (Tensor)

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.utils import isin
>>> td = TensorDict(
... {
... "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [1, 2, 3], [7, 8, 9]]),
... "tensor2": torch.tensor([[10, 20], [30, 40], [40, 50], [50, 60]]),
... },
... batch_size=[4],
... )
>>> td_ref = TensorDict(
... {
... "tensor1": torch.tensor([[1, 2, 3], [4, 5, 6], [10, 11, 12]]),
... "tensor2": torch.tensor([[10, 20], [30, 40], [50, 60]]),
... },
... batch_size=[3],
... )
>>> in_reference = isin(td, td_ref, key="tensor1")
>>> expected_in_reference = torch.tensor([True, True, True, False])
>>> torch.testing.assert_close(in_reference, expected_in_reference)
```