# TensorDictBase.irecv

TensorDictBase.irecv(*src: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group_src: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_premature: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *init_tag: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *pseudo_rand: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [list](https://docs.python.org/3/builtins/stdtypes.html#list)[torch.Future]] | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[torch.Future] | [None](https://docs.python.org/3/builtins/constants.html#None)

Receives the leaves that [`isend()`](tensordict.TensorDictBase.isend.html#tensordict.TensorDictBase.isend) or [`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send) sends, one asynchronous transfer per leaf.

The tensordict must have the keys, shapes and dtypes of the tensordict
that the sender sends; the values are written into its tensors in place.
By default, the call waits until all the transfers are done. With
`return_premature=True`, it returns at once: call `wait()` on each
returned `Work` object before you read the tensordict. See
[`isend()`](tensordict.TensorDictBase.isend.html#tensordict.TensorDictBase.isend) for an example, and [Distributed tensordicts](../../distributed.html#distributed).

Parameters:

**src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the global rank of the sender. Exclusive with
`group_src`: pass exactly one of them. Defaults to `None`.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - the process group.
Defaults to `None`, the default process group.
- **group_src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the rank of the sender in `group`, which
must be given. Exclusive with `src`. As in [`recv()`](tensordict.TensorDictBase.recv.html#tensordict.TensorDictBase.recv),
`group` can then be a standalone process group. Defaults to
`None`.
- **return_premature** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, returns the list of
`Work` objects of the transfers without waiting for them.
Defaults to `False`.
- **init_tag** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the `init_tag` that the sender passed.
Defaults to `0`.
- **pseudo_rand** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the `pseudo_rand` that the sender
passed. Defaults to `False`.
- **consolidated** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, receives one message into
the consolidated storage of the tensordict, as
`recv(consolidated=True)` does. The tensordict must be
consolidated already. `init_tag` and `pseudo_rand` are ignored.
Defaults to `False`.

Returns:

With `return_premature=True`, a list of `Work` objects. Otherwise,
`None`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if both or none of `src` and `group_src` are given, or
 if `group_src` is given without `group`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).