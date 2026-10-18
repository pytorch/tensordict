# TensorDictBase.recv

TensorDictBase.recv(*src: [int](https://docs.python.org/3/builtins/functions.html#int) | 'TensorDictPipe' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group_src: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *init_tag: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *pseudo_rand: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [int](https://docs.python.org/3/builtins/functions.html#int)

Receives the leaves that [`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send) sends, and writes them into the tensordict in place.

The tensordict must have the keys, shapes and dtypes of the tensordict
that the sender sends. See [`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send) for an example, and
[Distributed tensordicts](../../distributed.html#distributed).

Parameters:

**src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*TensorDictPipe*](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)*,**optional*) - the global rank of the sender,
or a [`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe) to receive from
with UCXX, which ignores the other arguments. Exclusive with
`group_src`: pass exactly one of them. Defaults to `None`.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.
- **group_src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the rank of the sender in `group`, which
must be given. Exclusive with `src`. The messages then go through
the backend of `group` directly, so `group` can be a standalone
`ProcessGroup` built on a store, never
registered by [`init_process_group()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) or
[`new_group()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.new_group). Defaults to `None`.
- **init_tag** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the `init_tag` that the sender passed.
Defaults to `0`.
- **pseudo_rand** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `pseudo_rand` that the sender
passed. Defaults to `False`.
- **consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, receives one message into
the consolidated storage of the tensordict, which updates all its
leaves, as sent by `send(consolidated=True)`. The tensordict must
be consolidated already, for example by
`consolidate(metadata=True)` or [`from_remote_init()`](tensordict.TensorDictBase.from_remote_init.html#tensordict.TensorDictBase.from_remote_init);
otherwise an `AttributeError` is raised. `init_tag` and
`pseudo_rand` are ignored. Defaults to `False`.

Returns:

With a pipe, the received tensordict, which is `self` only if the
tensordict is consolidated with the layout of the data, so use the
return value. With `consolidated=True`, `None`. Otherwise, the
tag of the last leaf received.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both or none of `src` and `group_src` are given, or
 if `group_src` is given without `group`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).