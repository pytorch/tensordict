TensorDict in distributed settings
==================================

TensorDict can be used in distributed settings to pass tensors from one node
to another.
If two nodes have access to a shared physical storage, a memory-mapped tensor can
be used to efficiently pass data from one running process to another.
Here, we provide some details on how this can be achieved in a distributed RPC setting.
For more details on distributed RPC, check the
`official pytorch documentation <https://pytorch.org/docs/stable/rpc.html>`_.

Creating a memory-mapped TensorDict
-----------------------------------

Memory-mapped tensors (and arrays) have the great advantage that they can store
a great amount of data and allow slices of data to be accessed readily without
reading the whole file in memory.
TensorDict offers an interface between memory-mapped
arrays and the :obj:`torch.Tensor` class named :obj:`MemmapTensor`.
:obj:`MemmapTensor` instances can be stored in :obj:`TensorDict` objects, allowing a
tensordict to represent a big dataset, stored on disk, easily accessible in a
batched way across nodes.

A memory-mapped tensordict is simply created via (1) populating a TensorDict with
memory-mapped tensors or (2) by calling :obj:`tensordict.memmap_()` to put it on
physical storage.
One can easily check that a tensordict is put on physical storage by querying
`tensordict.is_memmap()`.

Creating a memory-mapped tensor can itself be done in several ways.
Firstly, one can simply create an empty tensor:

>>> shape = torch.Size([3, 4, 5])
>>> tensor = Memmaptensor(*shape, prefix="/tmp")
>>> tensor[:2] = torch.randn(2, 4, 5)

The :obj:`prefix` attribute indicates where the temporary file has to be stored.
It is crucial that the tensor is stored in a directory that is accessible to every
node!

Another option is to represent an existing tensor on disk:

>>> tensor = torch.randn(3)
>>> tensor = Memmaptensor(tensor, prefix="/tmp")

The former method will be preferred when tensors are big or do not fit in memory:
it is suitable for tensors that are extremely big and serve as common storage
across nodes. For instance, one could create a dataset that would be easily accessed
by a single or different nodes, much faster than it would be if each file had to be
loaded independently in memory:

.. code-block::
   :caption: Creating an empty dataset on disk

   >>> dataset = TensorDict({
   ...      "images": MemmapTensor(50000, 480, 480, 3),
   ...      "masks": MemmapTensor(50000, 480, 480, 3, dtype=torch.bool),
   ...      "labels": MemmapTensor(50000, 1, dtype=torch.uint8),
   ... }, batch_size=[50000], device="cpu")
   >>> idx = [1, 5020, 34572, 11200]
   >>> batch = dataset[idx].clone()
   TensorDict(
       fields={
           images: Tensor(torch.Size([4, 480, 480, 3]), dtype=torch.float32),
           labels: Tensor(torch.Size([4, 1]), dtype=torch.uint8),
           masks: Tensor(torch.Size([4, 480, 480, 3]), dtype=torch.bool)},
       batch_size=torch.Size([4]),
       device=cpu,
       is_shared=False)

Notice that we have indicated the device of the :obj:`MemmapTensor`.
This syntax sugar allows for the tensors that are queried to be directly loaded
on device if needed.

Another consideration to take into account is that currently :obj:`MemmapTensor`
is not compatible with autograd operations.

Operating on Memory-mapped tensors across nodes
-----------------------------------------------

We provide a simple example of a distributed script where one process creates a
memory-mapped tensor, and sends its reference to another worker that is responsible of
updating it. You will find this example in the
`benchmark directory <https://github.com/pytorch-labs/tensordict/tree/main/benchmarks/distributed_benchmark.py>`_.

In short, our goal is to show how to handle read and write operations on big
tensors when nodes have access to a shared physical storage. The steps involve:

  - Creating the empty tensor on disk;

  - Setting the local and remote operations to be executed;

  - Passing commands from worker to worker using RPC to read and write the
    shared data.

This example first writes a function that updates a TensorDict instance
at specific indices with a one-filled tensor:

>>> def fill_tensordict(tensordict, idx):
...     tensordict[idx] = TensorDict(
...         {"memmap": torch.ones(5, 640, 640, 3, dtype=torch.uint8)}, [5]
...     )
...     return tensordict
>>> fill_tensordict_cp = CloudpickleWrapper(fill_tensordict)

The :obj:`CloudpickleWrapper` ensures that the function is serializable.
Next, we create a tensordict of a considerable size, to make the point that
this would be hard to pass from worker to worker if it had to be passed through
a regular tensorpipe:

>>> tensordict = TensorDict(
...     {"memmap": MemmapTensor(1000, 640, 640, 3, dtype=torch.uint8, prefix="/tmp/")}, [1000]
... )

Finally, still on the main node, we call the function *on the remote node* and then
check that the data has been written where needed:

>>> idx = [4, 5, 6, 7, 998]
>>> t0 = time.time()
>>> out = rpc.rpc_sync(
...     worker_info,
...     fill_tensordict_cp,
...     args=(tensordict, idx),
... )
>>> print("time elapsed:", time.time() - t0)
>>> print("check all ones", out["memmap"][idx, :1, :1, :1].clone())

Although the call to :obj:`rpc.rpc_sync` involved passing the entire tensordict,
updating specific indices of this object and return it to the original worker,
the execution of this snippet is extremely fast (even more so if the reference
to the memory location is already passed beforehand, see `torchrl's distributed
replay buffer documentation <https://github.com/pytorch/rl/blob/main/examples/distributed/distributed_replay_buffer.py>`_ to learn more).

The script contains additional RPC configuration steps that are beyond the
purpose of this document.
