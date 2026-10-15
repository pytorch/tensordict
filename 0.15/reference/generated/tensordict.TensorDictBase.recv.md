# TensorDictBase.recv

TensorDictBase.recv(*src: [int](https://docs.python.org/3/builtins/functions.html#int) | 'TensorDictPipe' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group_src: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *init_tag: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *pseudo_rand: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [int](https://docs.python.org/3/builtins/functions.html#int)

Receives the content of a tensordict and updates content with it.

Check the example in the send method for context.

Parameters:

**src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*TensorDictPipe*](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)*,**optional*) - the global rank of the
source worker, or a
[`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe) for UCXX-based
transport. When a pipe is passed, the data is received
in-place into this tensordict's consolidated storage (if
available and its layout matches the received data);
otherwise the pipe returns a tensordict of its own, so use
the returned value. Mutually exclusive with `group_src`;
exactly one of the two must be provided.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - if set, the specified process group
will be used for communication. Otherwise, the default process group
will be used.
Defaults to `None`.
- **group_src** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the rank of the source worker *relative
to* `group`. Requires `group` to be passed and is mutually
exclusive with `src`. When set, the p2p calls are issued
directly on the group's backend, so `group` may be a
standalone `ProcessGroup` built
against a store (never registered through
[`init_process_group()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) or
[`new_group()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.new_group)).
Defaults to `None`.
- **init_tag** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - the `init_tag` used by the source worker.
- **pseudo_rand** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if True, the sequence of tags will be pseudo-
random, allowing to send multiple data from different nodes
without overlap. Notice that the generation of these pseudo-random
numbers is expensive (1e-5 sec/number), meaning that it could
slow down the runtime of your algorithm.
This value must match the one passed to [`send()`](tensordict.TensorDictBase.send.html#tensordict.TensorDictBase.send).
Defaults to `False`.
- **consolidated** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if True, receives a single consolidated storage
tensor directly into `self._consolidated["storage"]`. The
tensordict must already be consolidated (e.g. from a prior
[`from_remote_init()`](tensordict.TensorDictBase.from_remote_init.html#tensordict.TensorDictBase.from_remote_init) call). All leaf tensor views update
in-place automatically.
Defaults to `False`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).