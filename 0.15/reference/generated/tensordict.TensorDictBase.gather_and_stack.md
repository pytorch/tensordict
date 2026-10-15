# TensorDictBase.gather_and_stack

TensorDictBase.gather_and_stack(*dst: [int](https://docs.python.org/3/builtins/functions.html#int)*, *group: 'torch.distributed.ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self | [None](https://docs.python.org/3/builtins/constants.html#None)

Gathers tensordicts from various workers and stacks them onto self in the destination worker.

Parameters:

- **dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the rank of the destination worker where `gather_and_stack()` will be called.
- **group** (*torch.distributed.ProcessGroup**,**optional*) - if set, the specified process group
will be used for communication. Otherwise, the default process group
will be used.
Defaults to `None`.

Example

```
>>> from torch import multiprocessing as mp
>>> from tensordict import TensorDict
>>> import torch
>>>
>>> def client():
... torch.distributed.init_process_group(
... "gloo",
... rank=1,
... world_size=2,
... init_method=f"tcp://localhost:10003",
... )
... # Create a single tensordict to be sent to server
... td = TensorDict(
... {("a", "b"): torch.randn(2),
... "c": torch.randn(2)}, [2]
... )
... td.gather_and_stack(0)
...
>>> def server():
... torch.distributed.init_process_group(
... "gloo",
... rank=0,
... world_size=2,
... init_method=f"tcp://localhost:10003",
... )
... # Creates the destination tensordict on server.
... # The first dim must be equal to world_size-1
... td = TensorDict(
... {("a", "b"): torch.zeros(2),
... "c": torch.zeros(2)}, [2]
... ).expand(1, 2).contiguous()
... td.gather_and_stack(0)
... assert td["a", "b"] != 0
... print("yuppie")
...
>>> if __name__ == "__main__":
... mp.set_start_method("spawn")
...
... main_worker = mp.Process(target=server)
... secondary_worker = mp.Process(target=client)
...
... main_worker.start()
... secondary_worker.start()
...
... main_worker.join()
... secondary_worker.join()
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).