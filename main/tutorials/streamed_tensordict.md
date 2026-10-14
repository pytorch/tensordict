Note

Go to the end
to download the full example code.

# Building tensordicts from streams

**Author**: [Vincent Moens](https://github.com/vmoens)

Sensors, robots, and transaction logs produce streams of data, often at
different rates. To process a stream in batches, you group the incoming values
into buckets, one for each fixed time window. The buckets have different
lengths, so they do not fit in one tensor. This tutorial groups two streams,
at 7 Hz and 3 Hz, into one-second buckets with a lazy stack of tensordicts. It
is for readers who batch data of variable size.

What you will learn:

- How to stack tensordicts whose entries have different shapes, with
[`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack).
- How to read streams with `asyncio` and write them into a tensordict at
regular intervals.
- How to convert a lazy stack into nested tensors with
[`densify()`](../reference/generated/tensordict.TensorDictBase.densify.html#tensordict.TensorDictBase.densify).

Before you start, read [Changing the shape of a TensorDict](tensordict_shapes.html#sphx-glr-tutorials-tensordict-shapes-py), which
introduces lazy stacks.

## Stacking tensordicts with different shapes

One bucket can hold more values than another. For example, the first two
buckets can be as follows:

```
import torch
from tensordict import TensorDict

bucket0 = TensorDict({"stream0": torch.randn(5), "stream1": torch.randn(4)})
bucket1 = TensorDict({"stream0": torch.randn(4), "stream1": torch.randn(5)})
```

You cannot stack these two tensordicts into one block of memory, because the
shapes of the streams differ. A [`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict)
groups tensordicts with different tensor shapes. To create one, call
[`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack):

```
from tensordict import lazy_stack

data = lazy_stack([bucket0, bucket1], dim=0)
print(data)
```

```
LazyStackedTensorDict(
 fields={
 stream0: Tensor(shape=torch.Size([2, -1]), device=cpu, dtype=torch.float32, is_shared=False),
 stream1: Tensor(shape=torch.Size([2, -1]), device=cpu, dtype=torch.float32, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0)
```

The result presents the two tensordicts as if they were stacked along
dimension 0. The shape `-1` in the printed output marks a dimension whose
size differs between the stacked tensordicts.
[`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) supports most operations of
[`TensorDictBase`](../reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase). Here are some examples:

```
data_select = data.select("stream0")
data_plus_1 = data + 1
data_apply = data.apply(lambda x: x + 1)
```

Indexing the stack returns the original tensordicts:

```
assert data[0] is bucket0
```

To store the data in contiguous memory, call
[`densify()`](../reference/generated/tensordict.TensorDictBase.densify.html#tensordict.TensorDictBase.densify). It stacks the tensors that have
the same shape, and stores the others as nested tensors. A nested tensor
holds tensors of different lengths in one tensor. With
`layout=torch.jagged`, all the values of an entry are in one buffer, and
`offsets()` gives the start of each tensor in this buffer:

```
data_cont = data.densify(layout=torch.jagged)
print(data_cont)
print("lengths of stream0:", data_cont["stream0"].offsets().diff())
```

```
TensorDict(
 fields={
 stream0: NestedTensor(shape=torch.Size([2, j2]), device=cpu, dtype=torch.float32, is_shared=False),
 stream1: NestedTensor(shape=torch.Size([2, j1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
lengths of stream0: tensor([5, 4])
```

## Asynchronous streams of data

Next, write a function that streams data at a given frequency. Here the data
are integers that increase by 1 at each step.

The function and its reader run as `asyncio` tasks, in one thread. To pass
the data between tasks, the function puts each value into a queue that it
receives as input:

```
import asyncio
from typing import List

async def generate_numbers(frequency: float, queue: asyncio.Queue) -> None:
 i = 0
 while True:
 await asyncio.sleep(1 / frequency)
 await queue.put(i)
 i += 1
```

The `collect_data` function reads the data from the queue for `timeout`
seconds, then returns the values that it read:

```
async def collect_data(queue: asyncio.Queue, timeout: float) -> List[int]:
 values = []

 # The nested `collect` task lets wait_for (below) stop the reads
 # when the timeout ends.
 async def collect():
 nonlocal values
 while True:
 value = await queue.get()
 values.append(value)

 task = asyncio.create_task(collect())
 try:
 await asyncio.wait_for(task, timeout=timeout)
 except asyncio.TimeoutError:
 task.cancel()
 return values
```

The `wait7hz` function runs a 7 Hz generator for one second and prints
what `collect_data` received:

```
async def wait7hz() -> None:
 queue = asyncio.Queue()
 generate_task = asyncio.create_task(generate_numbers(7, queue))
 collect_data_task = asyncio.create_task(collect_data(queue, timeout=1))
 values = await collect_data_task
 # generate_task still runs: stop it
 generate_task.cancel()
 print(values)

asyncio.run(wait7hz())
```

```
[0, 1, 2, 3, 4, 5]
```

Next, write a class that inherits from
[`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict). It reads the data of several
streams and writes each bucket into a new tensordict. A
[`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) can grow one tensordict at a time
with `append`, so the class appends each new bucket to the stack until it
has enough buckets. Here is the `StreamedTensorDict` class:

```
from typing import Callable, Dict

from tensordict import LazyStackedTensorDict, NestedKey, TensorDictBase

class StreamedTensorDict(LazyStackedTensorDict):
 """A lazy stack class that can be built from a dictionary of streams."""

 @classmethod
 async def from_streams(
 cls,
 streams: Dict[NestedKey, Callable],
 timeout: float,
 batch_size: int,
 densify: bool = True,
 ) -> TensorDictBase:
 td = cls(stack_dim=0)

 # one queue for each stream
 queues = [asyncio.Queue() for _ in range(len(streams))]
 tasks = []
 for stream, queue in zip(streams.values(), queues):
 task = asyncio.create_task(stream(queue))
 tasks.append(task)
 for _ in range(batch_size):
 values_tasks = []
 for queue in queues:
 values_task = asyncio.create_task(collect_data(queue, timeout))
 values_tasks.append(values_task)
 values = await asyncio.gather(*values_tasks)
 td.append(TensorDict(dict(zip(streams.keys(), values))))

 # Cancel the generator tasks
 for task in tasks:
 task.cancel()
 if densify:
 return td.densify(layout=torch.jagged)
 return td
```

Last, the `main` function passes the streaming functions `stream0` (7 Hz)
and `stream1` (3 Hz) to `StreamedTensorDict.from_streams`. This method
collects `batch_size` buckets of data, for `timeout=1` second each:

```
async def main() -> TensorDictBase:
 def stream0(queue):
 return generate_numbers(frequency=7, queue=queue)

 def stream1(queue):
 return generate_numbers(frequency=3, queue=queue)

 # This takes about 5 seconds: 5 buckets of 1 second
 return await StreamedTensorDict.from_streams(
 {"bucket0": stream0, "bucket1": stream1}, timeout=1, batch_size=5
 )

td = asyncio.run(main())

print("TensorDict from stream", td)
```

```
TensorDict from stream TensorDict(
 fields={
 bucket0: NestedTensor(shape=torch.Size([5, j3]), device=cpu, dtype=torch.int64, is_shared=False),
 bucket1: NestedTensor(shape=torch.Size([5, j4]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([5]),
 device=None,
 is_shared=False)
```

Print the data of both streams. Each stream counts up from 0, so the values
of each stream are `0, 1, 2, ...`. In 5 buckets of 1 second, the 7 Hz
stream gives about 35 values, and the 3 Hz stream about 15. The lengths of
the buckets come from `offsets()`:

```
print("bucket0 (7 Hz, about 35 values)", td["bucket0"].values())
print("bucket1 (3 Hz, about 15 values)", td["bucket1"].values())
print("values per bucket of bucket0 (about 7)", td["bucket0"].offsets().diff())
```

```
bucket0 (7 Hz, about 35 values) tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33])
bucket1 (3 Hz, about 15 values) tensor([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
values per bucket of bucket0 (about 7) tensor([6, 7, 7, 7, 7])
```

## Conclusion

You can now batch data of variable size. Group the tensordicts with
[`lazy_stack()`](../reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack), grow the
[`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) one bucket at a time with
`append`, and convert it into nested tensors with `densify`. The same
pattern works for any stream that you read in fixed time windows.

## Further reading

- [Stacking and variable-size data](../concepts/stacking.html#stacking) explains lazy stacks and data of different sizes.
- [Changing the shape of a TensorDict](tensordict_shapes.html#sphx-glr-tutorials-tensordict-shapes-py) compares lazy stacks and
dense stacks.
- The [`LazyStackedTensorDict`](../reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) reference lists the
operations of a lazy stack.
- The [torch.nested documentation](https://pytorch.org/docs/stable/nested.html)
explains the jagged layout of nested tensors.

**Total running time of the script:** (0 minutes 6.020 seconds)

[`Download Jupyter notebook: streamed_tensordict.ipynb`](../_downloads/ea350b8ef859f0797dd9ceffacb742e6/streamed_tensordict.ipynb)

[`Download Python source code: streamed_tensordict.py`](../_downloads/cf148f8c4b7f2a4a3f315989a6f7ce2c/streamed_tensordict.py)

[`Download zipped: streamed_tensordict.zip`](../_downloads/c1f74b953ed757c75e2d2bc64666eae1/streamed_tensordict.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)