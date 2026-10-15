# TensorDictBase.init_remote

TensorDictBase.init_remote(*dst: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *group: 'ProcessGroup' | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *use_broadcast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *_tensorclass_type: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Initializes a remote tensordict by sending its metadata and content.

Two transport modes are available:

- **Point-to-point** (default): uses `send_object_list` +
`dist.send` to transfer metadata and storage to a single
destination rank. Only the sender and receiver need to call the
pair `init_remote` / `from_remote_init`.
- **Broadcast** (`use_broadcast=True`): delegates to
[`broadcast()`](tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast). Metadata and storage are broadcast to
*every* rank in the group. All ranks must participate (receivers
call `from_remote_init(src=..., use_broadcast=True)`). `dst`
is ignored in this mode.

Parameters:

- **dst** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - The rank of the destination process. Required
when `use_broadcast=False` (the default). Ignored when
`use_broadcast=True`.
- **group** (*"ProcessGroup"**,**optional*) - The process group to use for communication. Defaults to None.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device to use for tensor operations. Defaults to None.
- **use_broadcast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)) - If `True`, use [`broadcast()`](tensordict.TensorDictBase.broadcast.html#tensordict.TensorDictBase.broadcast) instead
of point-to-point send. Defaults to `False`.

See also

The receiving process should call ~.from_remote_init or an equivalent method to receive and initialize a new tensordict based on the sent metadata.

Examples

```
>>> import os
>>> import torch
>>> import torch.distributed as dist
>>> from tensordict import TensorDict, MemoryMappedTensor
>>> import multiprocessing as mp
>>>
>>> def server(queue):
... # Set environment variables for distributed communication
... os.environ["MASTER_ADDR"] = "localhost"
... os.environ["MASTER_PORT"] = "29505"
...
... # Initialize the distributed backend
... dist.init_process_group("gloo", rank=0, world_size=2)
...
... # Create a sample tensordict
... td = (
... TensorDict(
... {
... ("a", "b"): torch.ones(2),
... "c": torch.ones(2),
... ("d", "e", "f"): MemoryMappedTensor.from_tensor(torch.ones(2, 2)),
... },
... [2],
... )
... .expand(1, 2)
... .contiguous()
... )
...
... # Send the tensordict metadata and content to the client
... td.init_remote(dst=1)
...
>>> def client(queue):
... # Set environment variables for distributed communication
... os.environ["MASTER_ADDR"] = "localhost"
... os.environ["MASTER_PORT"] = "29505"
...
... # Initialize the distributed backend
... dist.init_process_group("gloo", rank=1, world_size=2)
...
... # Receive the tensordict metadata and content from the server
... received_td = TensorDict.from_remote_init(src=0)
...
... # Verify that the received tensordict matches the expected structure and values
... assert set(received_td.keys()) == {"a", "c", "d"}
... assert (received_td == 1).all()
...
... # Signal that the test has completed successfully
... queue.put("yuppie")
>>>
>>> if __name__ == "__main__":
... queue = mp.Queue(1)
...
... # Create and start the server and client processes
... main_worker = mp.Process(target=server, args=(queue,))
... secondary_worker = mp.Process(target=client, args=(queue,))
...
... main_worker.start()
... secondary_worker.start()
...
... try:
... out = queue.get(timeout=10) # Wait for the signal with a timeout
... print(out) # Should print "yuppie"
... finally:
... queue.close()
... main_worker.join(timeout=10)
... secondary_worker.join(timeout=10)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).