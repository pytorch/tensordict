# collate

tensordict.collate(*batch: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***, *pad_keys: [Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [float](https://docs.python.org/3/builtins/functions.html#float)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *nested_keys: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *mask_key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Any](https://docs.python.org/3/library/typing.html#typing.Any)

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

The keyword arguments batch variable-length entries of the tensordicts
and tensorclasses in the batch. Pass them with
[`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial) to get a `collate_fn`. The entries they do
not name are collated as above.

Parameters:

**batch** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)) - the samples of one batch. Anything other than a
non-empty list is returned as is, for example the tensordict
that a batch sampler reads in one index operation.

Keyword Arguments:

- **pad_keys** (*Mapping**[*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,*[*float*](https://docs.python.org/3/builtins/functions.html#float)*]**,**optional*) - the entries to pad,
each mapped to its padding value. They are padded along their
first dimension to the longest sample and stacked, as
[`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence) does. Defaults to `None`.
- **nested_keys** (*sequence**of*[*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - the entries to stack
into nested tensors with the `torch.jagged` layout, whose first
dimension is ragged. The other entries must then stack densely:
name every entry that varies in length in `pad_keys` or
`nested_keys`. Defaults to `None`.
- **mask_key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - where to store the masks of the
`pad_keys` entries: a tensordict with an entry for each of them
that is `True` at the valid positions, as
[`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence) builds with `return_mask`. It
must not be an entry of the samples. A tensorclass needs a field
of that name. Requires `pad_keys`. Defaults to `None`, for no
masks.

With `pad_keys`, the samples must have an empty batch size. These
options do not support [`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) (h5)
samples: call [`to_tensordict()`](tensordict.TensorDictBase.to_tensordict.html#tensordict.TensorDictBase.to_tensordict) first.

Returns:

The collated batch.

Raises:

- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if a sample has no entry for a key of `pad_keys` or
 `nested_keys`.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `mask_key` is given without `pad_keys` or names
 an entry of a sample, if a key is in both `pad_keys` and
 `nested_keys`, if `pad_keys` is given for samples with batch
 dimensions, or if `nested_keys` is given and the other entries
 do not stack densely.

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

Pad the token ids with the pad id of a tokenizer and the labels with
`-100`, keep a mask of the valid tokens, and stack the point clouds as
jagged nested tensors:

```
>>> from functools import partial
>>> samples = [
... TensorDict(
... ids=torch.ones(n, dtype=torch.long),
... labels=torch.ones(n, dtype=torch.long),
... points=torch.randn(n + 1, 3),
... score=torch.tensor(float(n)),
... )
... for n in (2, 3)
... ]
>>> collate_fn = partial(
... collate,
... pad_keys={"ids": 7, "labels": -100},
... nested_keys=["points"],
... mask_key="mask",
... )
>>> batch = next(iter(DataLoader(samples, batch_size=2, collate_fn=collate_fn)))
>>> batch["ids"]
tensor([[1, 1, 7],
 [1, 1, 1]])
>>> batch["mask", "ids"]
tensor([[ True, True, False],
 [ True, True, True]])
>>> batch["points"].is_nested, batch["score"]
(True, tensor([2., 3.]))
```