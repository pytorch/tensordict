# TensorDictBase.isend

TensorDictBase.isend(*dst: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group_dst: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *init_tag: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *pseudo_rand: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [int](https://docs.python.org/3/builtins/functions.html#int) | List['Work'] | [None](https://docs.python.org/3/builtins/constants.html#None)

Sends the leaves of the tensordict to another rank, one asynchronous transfer per leaf.

As [`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send), but the transfers of all the leaves run at the same
time. By default, the call waits until they are done. With
`return_early=True`, it returns at once: call `wait()` on each
returned `Work` object before you change the tensordict. The receiver
calls [`irecv()`](tensordict.TensorDictBase.irecv.html#tensordict.TensorDictBase.irecv) or [`recv()`](tensordict.TensorDictBase.recv.html#tensordict.TensorDictBase.recv). See [Distributed tensordicts](../../distributed.html#distributed).

Parameters:

**dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the global rank of the receiver. Exclusive with
`group_dst`: pass exactly one of them. Defaults to `None`.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.
- **group_dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the rank of the receiver in `group`,
which must be given. Exclusive with `dst`. As in [`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send),
`group` can then be a standalone process group. Defaults to
`None`.
- **init_tag** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the tag of the first leaf. The next leaves
use `init_tag + 1`, `init_tag + 2`, and so on. The receiver must
pass the same value. Defaults to `0`.
- **pseudo_rand** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the tags are pseudo-random
numbers seeded by `init_tag`. The receiver must pass the same
value. Defaults to `False`.
- **return_early** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns the list of
`Work` objects of the transfers without waiting for them.
Defaults to `False`.
- **consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sends all the leaves as one
message, as `send(consolidated=True)` does. A tensordict that is
not consolidated is consolidated on each call, which copies every
leaf: keep a consolidated copy and refresh it with
[`update_()`](tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_) to avoid the copy. The receiver must call
`irecv(consolidated=True)` or `recv(consolidated=True)`.
`init_tag` and `pseudo_rand` are ignored. Defaults to `False`.

Returns:

With `return_early=True`, a list of `Work` objects. Otherwise, the
tag of the last leaf sent, or `None` with `consolidated=True`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both or none of `dst` and `group_dst` are given, or
 if `group_dst` is given without `group`.

Examples

This example runs in two processes: add it to the
[runner script of the distributed guide](../../distributed.html#distributed) and
pass it to `run`.

```
def isend_irecv(rank, world_size):
 if rank == 0:
 td = TensorDict(a=torch.ones(4), b=torch.ones(4, 3), batch_size=[4])
 works = td.isend(dst=1, return_early=True)
 # ... other work, which does not change td ...
 for work in works:
 work.wait()
 elif rank == 1:
 td = TensorDict(a=torch.zeros(4), b=torch.zeros(4, 3), batch_size=[4])
 works = td.irecv(src=0, return_premature=True)
 # ... other work, which does not read td ...
 for work in works:
 work.wait()
 print(td["b"].sum()) # tensor(12.)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).