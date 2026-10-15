# TensorDictBase.all_gather

TensorDictBase.all_gather(***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [list](https://docs.python.org/3/builtins/stdtypes.html#list)

All-gathers tensordicts from every rank.

Each rank consolidates its tensordict, then metadata is gathered via
`all_gather_object` and storage via `dist.all_gather`. Returns a
list of tensordicts, one per rank.

Keyword Arguments:

**group** (*torch.distributed.ProcessGroup**,**optional*) - The process group
to use. Defaults to `None` (default group).

Returns:

A list of tensordicts from all ranks.

Return type:

[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)[[*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)]

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).