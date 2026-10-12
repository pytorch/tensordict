# TensorDictBase.irecv

TensorDictBase.irecv(*src: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group_src: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *return_premature: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *init_tag: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *pseudo_rand: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [tuple](https://docs.python.org/3/builtins/stdtypes.html#tuple)[[int](https://docs.python.org/3/builtins/functions.html#int), [list](https://docs.python.org/3/builtins/stdtypes.html#list)[torch.Future]] | [list](https://docs.python.org/3/builtins/stdtypes.html#list)[torch.Future] | [None](https://docs.python.org/3/builtins/constants.html#None)

Receives the content of a tensordict and updates content with it asynchronously.

Check the example in the [`isend()`](tensordict.TensorDictBase.isend.html#tensordict.TensorDictBase.isend) method for context.

Parameters:

**src** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the global rank of the source worker. Mutually
exclusive with `group_src`; exactly one of the two must be
provided.

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
- **return_premature** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if `True`, returns a list of futures to wait
upon until the tensordict is updated. Defaults to `False`,
i.e. waits until update is completed withing the call.
- **init_tag** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - the `init_tag` used by the source worker.
- **pseudo_rand** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if True, the sequence of tags will be pseudo-
random, allowing to send multiple data from different nodes
without overlap. Notice that the generation of these pseudo-random
numbers is expensive (1e-5 sec/number), meaning that it could
slow down the runtime of your algorithm.
This value must match the one passed to [`isend()`](tensordict.TensorDictBase.isend.html#tensordict.TensorDictBase.isend).
Defaults to `False`.

Returns:

if `return_premature=True`, a list of futures to wait

upon until the tensordict is updated.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).