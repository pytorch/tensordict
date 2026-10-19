# TensorDictBase.send

TensorDictBase.send(*dst: [int](https://docs.python.org/3/builtins/functions.html#int) | 'TensorDictPipe' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group_dst: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *init_tag: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *pseudo_rand: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Sends the leaves of the tensordict to another rank, with [`torch.distributed`](https://docs.pytorch.org/docs/stable/distributed.html#module-torch.distributed).

The receiver calls [`recv()`](tensordict.TensorDictBase.recv.html#tensordict.TensorDictBase.recv) on a tensordict with the same keys,
shapes and dtypes, which writes the values into its tensors. By default,
each leaf is one message, marked with a tag. To send to a rank that does
not know the structure, use [`init_remote()`](tensordict.TensorDictBase.init_remote.html#tensordict.TensorDictBase.init_remote). See [Distributed tensordicts](../../distributed.html#distributed).

Parameters:

**dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*TensorDictPipe*](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)*,**optional*) - the global rank of the
receiver, or a [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe) to send
through with UCXX, which ignores the other arguments. Exclusive
with `group_dst`: pass exactly one of them. Defaults to `None`.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.
- **group_dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the rank of the receiver in `group`,
which must be given. Exclusive with `dst`. The messages then go
through the backend of `group` directly, so `group` can be a
standalone `ProcessGroup` built on a store,
never registered by [`init_process_group()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) or
[`new_group()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.new_group). Defaults to `None`.
- **init_tag** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the tag of the first leaf. The next leaves
use `init_tag + 1`, `init_tag + 2`, and so on. The receiver must
pass the same value. Give two transfers that run at the same time
tag ranges that do not overlap. Defaults to `0`.
- **pseudo_rand** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tags are pseudo-random
numbers seeded by `init_tag`, so that transfers from several
ranks do not overlap. Generating them costs about 10
microseconds per leaf. The receiver must pass the same value.
Defaults to `False`.
- **consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sends all the leaves as one
message: the tensordict is consolidated first (see
[`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate)), unless it is already. The receiver must call
`recv(consolidated=True)` on a consolidated tensordict with the
same structure. `init_tag` and `pseudo_rand` are ignored.
Defaults to `False`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both or none of `dst` and `group_dst` are given, or
 if `group_dst` is given without `group`.

Examples

This example runs in two processes: add it to the
[runner script of the distributed guide](../../distributed.html#distributed) and
pass it to `run`.

```
def send_recv(rank, world_size):
 if rank == 0:
 td = TensorDict(a=torch.ones(4), b=torch.ones(4, 3), batch_size=[4])
 td.send(dst=1)
 elif rank == 1:
 td = TensorDict(a=torch.zeros(4), b=torch.zeros(4, 3), batch_size=[4])
 td.recv(src=0)
 print(td["b"].sum()) # tensor(12.)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).