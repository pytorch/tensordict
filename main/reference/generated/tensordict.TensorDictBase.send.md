# TensorDictBase.send

TensorDictBase.send(*dst: [int](https://docs.python.org/3/builtins/functions.html#int) | 'TensorDictPipe' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group_dst: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *init_tag: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *pseudo_rand: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *consolidated: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Sends the content of a tensordict to a distant worker.

Parameters:

**dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*TensorDictPipe*](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe)*,**optional*) - the global rank of the
destination worker where the content should be sent, or a
[`TensorDictPipe`](tensordict._ucxx.TensorDictPipe.html#tensordict._ucxx.TensorDictPipe) for UCXX-based
transport. Mutually exclusive with `group_dst`; exactly one
of the two must be provided.

Keyword Arguments:

- **group** (*torch.distributed.ProcessGroup**,**optional*) - if set, the specified process group
will be used for communication. Otherwise, the default process group
will be used.
Defaults to `None`.
- **group_dst** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the rank of the destination worker
*relative to* `group`. Requires `group` to be passed and is
mutually exclusive with `dst`. When set, the p2p calls are
issued directly on the group's backend, so `group` may be a
standalone `ProcessGroup` built
against a store (never registered through
[`init_process_group()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) or
[`new_group()`](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.new_group)).
Defaults to `None`.
- **init_tag** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)) - the initial tag to be used to mark the tensors.
Note that this will be incremented by as much as the number of
tensors contained in the TensorDict.
- **pseudo_rand** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if True, the sequence of tags will be pseudo-
random, allowing to send multiple data from different nodes
without overlap. Notice that the generation of these pseudo-random
numbers is expensive (1e-5 sec/number), meaning that it could
slow down the runtime of your algorithm.
Defaults to `False`.
- **consolidated** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)) - if True, sends the consolidated storage as a
single tensor (1 message). The tensordict is consolidated first
if needed. The receiver must use `recv(consolidated=True)` and
must already hold a consolidated tensordict with matching schema.
Defaults to `False`.

Example

```
>>> from torch import multiprocessing as mp
>>> from tensordict import TensorDict
>>> import torch
>>>
>>>
>>> def client():
... torch.distributed.init_process_group(
... "gloo",
... rank=1,
... world_size=2,
... init_method=f"tcp://localhost:10003",
... )
...
... td = TensorDict(
... {
... ("a", "b"): torch.randn(2),
... "c": torch.randn(2, 3),
... "_": torch.ones(2, 1, 5),
... },
... [2],
... )
... td.send(0)
...
>>>
>>> def server(queue):
... torch.distributed.init_process_group(
... "gloo",
... rank=0,
... world_size=2,
... init_method=f"tcp://localhost:10003",
... )
... td = TensorDict(
... {
... ("a", "b"): torch.zeros(2),
... "c": torch.zeros(2, 3),
... "_": torch.zeros(2, 1, 5),
... },
... [2],
... )
... td.recv(1)
... assert (td != 0).all()
... queue.put("yuppie")
...
>>>
>>> if __name__=="__main__":
... queue = mp.Queue(1)
... main_worker = mp.Process(target=server, args=(queue,))
... secondary_worker = mp.Process(target=client)
...
... main_worker.start()
... secondary_worker.start()
... out = queue.get(timeout=10)
... assert out == "yuppie"
... main_worker.join()
... secondary_worker.join()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).