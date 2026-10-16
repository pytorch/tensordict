# TensorDictBase.update_at

TensorDictBase.update_at_(*input_dict_or_td: [dict](https://docs.python.org/3/builtins/stdtypes.html#dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)] | T*, *idx: [None](https://docs.python.org/3/builtins/constants.html#None) | [int](https://docs.python.org/3/builtins/functions.html#int) | [slice](https://docs.python.org/3/builtins/functions.html#slice) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Tuple](https://docs.python.org/3/library/typing.html#typing.Tuple)[[Any](https://docs.python.org/3/library/typing.html#typing.Any), ...]*, *clone: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *non_blocking: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *keys_to_update: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Updates the TensorDict in-place at the specified index with values from either a dictionary or another TensorDict.

Unlike TensorDict.update, this function will throw an error if the key is unknown to the TensorDict.
This method keeps the general `set_at_` semantics and supports tensor
and non-tensor leaves. For optimized tensor-only copies in hot paths,
use [`copy_at_()`](tensordict.TensorDictBase.copy_at_.html#tensordict.TensorDictBase.copy_at_) with `fast=True`.

Parameters:

- **input_dict_or_td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - input data to be written
in self.
- **idx** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,*[*Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*,**iterable**,*[*slice*](https://docs.python.org/3/builtins/functions.html#slice)) - index of the tensordict
where the update should occur.
- **clone** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether the tensors in the input (
tensor) dict should be cloned before being set. Default is
False.

Keyword Arguments:

- **keys_to_update** (*sequence**of**NestedKeys**,**optional*) - if provided, only
the list of keys in `key_to_update` will be updated.
- **non_blocking** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True` and this copy is between
different devices, the copy may occur asynchronously with respect
to the host.

Returns:

self

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... 'a': torch.zeros(3, 4, 5),
... 'b': torch.zeros(3, 4, 10)}, batch_size=[3, 4])
>>> td.update_at_(
... TensorDict({
... 'a': torch.ones(1, 4, 5),
... 'b': torch.ones(1, 4, 10)}, batch_size=[1, 4]),
... slice(1, 2))
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
 b: Tensor(shape=torch.Size([3, 4, 10]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=None,
 is_shared=False)
>>> assert (td[1] == 1).all()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).