# -*- coding: utf-8 -*-

"""
Building tensordicts from streams
=================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

Sensors, robots, and transaction logs produce streams of data, often at
different rates. To process a stream in batches, you group the incoming values
into buckets, one for each fixed time window. The buckets have different
lengths, so they do not fit in one tensor. This tutorial groups two streams,
at 7 Hz and 3 Hz, into one-second buckets with a lazy stack of tensordicts. It
is for readers who batch data of variable size.

What you will learn:

- How to stack tensordicts whose entries have different shapes, with
  :func:`~tensordict.lazy_stack`.
- How to read streams with ``asyncio`` and write them into a tensordict at
  regular intervals.
- How to convert a lazy stack into nested tensors with
  :meth:`~tensordict.TensorDictBase.densify`.

Before you start, read :ref:`sphx_glr_tutorials_tensordict_shapes.py`, which
introduces lazy stacks.

Stacking tensordicts with different shapes
------------------------------------------

One bucket can hold more values than another. For example, the first two
buckets can be as follows:

"""

import torch
from tensordict import TensorDict

bucket0 = TensorDict({"stream0": torch.randn(5), "stream1": torch.randn(4)})
bucket1 = TensorDict({"stream0": torch.randn(4), "stream1": torch.randn(5)})

############################################
# You cannot stack these two tensordicts into one block of memory, because the
# shapes of the streams differ. A :class:`~tensordict.LazyStackedTensorDict`
# groups tensordicts with different tensor shapes. To create one, call
# :func:`~tensordict.lazy_stack`:

from tensordict import lazy_stack

data = lazy_stack([bucket0, bucket1], dim=0)
print(data)

############################################
# The result presents the two tensordicts as if they were stacked along
# dimension 0. The shape ``-1`` in the printed output marks a dimension whose
# size differs between the stacked tensordicts.
# :class:`~tensordict.LazyStackedTensorDict` supports most operations of
# :class:`~tensordict.TensorDictBase`. Here are some examples:

data_select = data.select("stream0")
data_plus_1 = data + 1
data_apply = data.apply(lambda x: x + 1)

############################################
# Indexing the stack returns the original tensordicts:

assert data[0] is bucket0

############################################
# To store the data in contiguous memory, call
# :meth:`~tensordict.TensorDictBase.densify`. It stacks the tensors that have
# the same shape, and stores the others as nested tensors. A nested tensor
# holds tensors of different lengths in one tensor. With
# ``layout=torch.jagged``, all the values of an entry are in one buffer, and
# ``offsets()`` gives the start of each tensor in this buffer:

data_cont = data.densify(layout=torch.jagged)
print(data_cont)
print("lengths of stream0:", data_cont["stream0"].offsets().diff())

############################################
# Asynchronous streams of data
# ----------------------------
#
# Next, write a function that streams data at a given frequency. Here the data
# are integers that increase by 1 at each step.
#
# The function and its reader run as ``asyncio`` tasks, in one thread. To pass
# the data between tasks, the function puts each value into a queue that it
# receives as input:

import asyncio
from typing import List


async def generate_numbers(frequency: float, queue: asyncio.Queue) -> None:
    i = 0
    while True:
        await asyncio.sleep(1 / frequency)
        await queue.put(i)
        i += 1


############################################
# The ``collect_data`` function reads the data from the queue for ``timeout``
# seconds, then returns the values that it read:


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


############################################
# The ``wait7hz`` function runs a 7 Hz generator for one second and prints
# what ``collect_data`` received:


async def wait7hz() -> None:
    queue = asyncio.Queue()
    generate_task = asyncio.create_task(generate_numbers(7, queue))
    collect_data_task = asyncio.create_task(collect_data(queue, timeout=1))
    values = await collect_data_task
    # generate_task still runs: stop it
    generate_task.cancel()
    print(values)


asyncio.run(wait7hz())

############################################
# Next, write a class that inherits from
# :class:`~tensordict.LazyStackedTensorDict`. It reads the data of several
# streams and writes each bucket into a new tensordict. A
# :class:`~tensordict.LazyStackedTensorDict` can grow one tensordict at a time
# with ``append``, so the class appends each new bucket to the stack until it
# has enough buckets. Here is the ``StreamedTensorDict`` class:
#

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


############################################
# Last, the ``main`` function passes the streaming functions ``stream0`` (7 Hz)
# and ``stream1`` (3 Hz) to ``StreamedTensorDict.from_streams``. This method
# collects ``batch_size`` buckets of data, for ``timeout=1`` second each:


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

############################################
# Print the data of both streams. Each stream counts up from 0, so the values
# of each stream are ``0, 1, 2, ...``. In 5 buckets of 1 second, the 7 Hz
# stream gives about 35 values, and the 3 Hz stream about 15. The lengths of
# the buckets come from ``offsets()``:

print("bucket0 (7 Hz, about 35 values)", td["bucket0"].values())
print("bucket1 (3 Hz, about 15 values)", td["bucket1"].values())
print("values per bucket of bucket0 (about 7)", td["bucket0"].offsets().diff())

############################################
# Conclusion
# ----------
#
# You can now batch data of variable size. Group the tensordicts with
# :func:`~tensordict.lazy_stack`, grow the
# :class:`~tensordict.LazyStackedTensorDict` one bucket at a time with
# ``append``, and convert it into nested tensors with ``densify``. The same
# pattern works for any stream that you read in fixed time windows.
#
# Further reading
# ---------------
#
# - :ref:`stacking` explains lazy stacks and data of different sizes.
# - :ref:`sphx_glr_tutorials_tensordict_shapes.py` compares lazy stacks and
#   dense stacks.
# - The :class:`~tensordict.LazyStackedTensorDict` reference lists the
#   operations of a lazy stack.
# - The `torch.nested documentation <https://pytorch.org/docs/stable/nested.html>`__
#   explains the jagged layout of nested tensors.
