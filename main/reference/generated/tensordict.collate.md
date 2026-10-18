# collate

tensordict.collate(*batch: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

Collates a list of samples into a batch, for [`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).

Pass it as `collate_fn` when the samples are tensordicts or
tensorclasses, or tuples or dicts that hold them, such as
`(td, label)` pairs. The default `collate_fn` of a `DataLoader`
cannot read a tensordict (see [Load data and share it between processes](../../data_loading.html#data-loading)).

The samples are collated by the type of the first one:

- tensordicts and tensorclasses go through
[`maybe_dense_stack()`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.maybe_dense_stack): the batch
is a dense stack if the samples have the same keys and entry shapes,
and a [`LazyStackedTensorDict`](tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) otherwise. A
tensorclass keeps its type.
- tensors go through [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack).
- mappings give a `dict` and tuples a tuple (a namedtuple keeps its
type), with each entry collated across the samples.
- anything else goes through [`torch.utils.data.default_collate()`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.default_collate).

Parameters:

**batch** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)) - the samples of one batch. Anything other than a
non-empty list is returned as is, for example the tensordict
that a batch sampler reads in one index operation.

Returns:

The collated batch.

Examples

```
>>> import torch
>>> from torch.utils.data import DataLoader
>>> from tensordict import TensorDict, collate
>>> samples = [(TensorDict(x=torch.randn(3)), i) for i in range(10)]
>>> td, label = next(iter(DataLoader(samples, batch_size=4, collate_fn=collate)))
>>> td.batch_size, label
(torch.Size([4]), tensor([0, 1, 2, 3]))
>>> clouds = [TensorDict(points=torch.randn(n, 3)) for n in (5, 2, 7, 3)]
>>> batch = next(iter(DataLoader(clouds, batch_size=4, collate_fn=collate)))
>>> type(batch).__name__, batch.batch_size
('LazyStackedTensorDict', torch.Size([4]))
```