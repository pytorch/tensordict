# -*- coding: utf-8 -*-

"""
Building tensordicts from streams
=================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

In many real-world applications, data is generated continuously and at varying frequencies.

For example, sensor readings from IoT devices, financial transactions, or social media updates can all produce streams
of data that need to be processed and analyzed in real-time.

When working with such data streams, it's often necessary to "bucketize" the incoming data into discrete chunks,
allowing for efficient processing and analysis. However, this can be challenging when dealing with data streams that
have different frequencies or formats.

In this tutorial, we'll explore how to use TensorDict to build and manipulate data streams.
We'll learn how to create lazy stacks of tensors, handle asynchronous data streams, and densify our data for efficient
storage and processing.

In this tutorial, you will learn:
- How to read streams of data and write them at regular intervals within a tensordict;
- How to build TensorDict that stack contents with heterogeneous shapes together;
- How to densify these tensors in single storages using ``nested_tensor`` if required.

Stacking heterogeneous tensordicts together
-------------------------------------------

In many real-life scenarios, data come in streams that have different defined frequencies.

Our goal in this tutorial is to "bucketize" the upcoming data such that it can be read and processed at a given
slower frequency.
The challenge in this scenario is that the data may not be representable in regular "rectangular" format (i.e., where
each dimension of the tensor is well-defined), but it could be the case that one bucket of data has more element than
another, in which case we cannot simply stack them together. Typically, consider the case where the first and second
buckets of data are as follows:

"""

import torch
from tensordict import TensorDict

bucket0 = TensorDict(stream0=torch.randn(5), stream1=torch.randn(4))
bucket1 = TensorDict(stream0=torch.randn(4), stream1=torch.randn(5))

############################################
# In principle, we cannot stack these two tensordict contiguously in memory as the shape of the two streams differ.
# Fortunately, TensorDict offers a tool to group instances with heterogeneous tensor shapes together:
# :class:`~tensordict.LazyStackedTensorDict`.
# To create a lazy stack, one can just call :meth:`~tensordict.TensorDict.lazy_stack`:

data = TensorDict.lazy_stack([bucket0, bucket1], dim=0)
print(data)

############################################
# The resulting data is just a representation of the two tensordicts as if they had been stacked together along
# dimension 0. :class:`~tensordict.LazyStackedTensorDict` supports most common operations of the
# :class:`~tensordict.TensorDictBase` class, here are some examples:

data_select = data.select("stream0")
data_plus_1 = data + 1
data_apply = data.apply(lambda x: x + 1)

############################################
# Moreover, indexing it will return the original data we used to create the stack

assert data[0] is bucket0

############################################
# Still, in some instances, one could wish to have a contiguous representation of the underlying data.
# To do this, :class:`~tensordict.TensorDictBase` offers a :meth:`~tensordict.TensorDictBase.densify` method that
# will stack the tensors that can be stacked, and attempt to represent the rest as ``nested_tensor`` instances:

data_cont = data.densify()

############################################
# Asynchronous streams of data
# ----------------------------
#
# Let us now switch to a more concrete example, where we create a function that streams data (in this case, just
# integers incremented by 1 at each iteration) at a given frequency.
#
# To pass the data across threads, the function will use a queue received as input:

import asyncio
from typing import List


async def generate_numbers(frequency: float, queue: asyncio.Queue) -> None:
    i = 0
    while True:
        await asyncio.sleep(1 / frequency)
        await queue.put(i)
        i += 1


############################################
# The ``collect_data`` function reads the data from the queue for a given amount of time.
# As soon as ``timeout`` has passed, the function returns:


async def collect_data(queue: asyncio.Queue, timeout: float) -> List[int]:
    values = []

    # We create a nested `collect` async function in order to be able to stop it as
    #  soon as timeout is passed (see wait_for below).
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
# The ``wait7hz`` function reads the data from the queue for a given amount of time.
#


async def wait7hz() -> None:
    queue = asyncio.Queue()
    generate_task = asyncio.create_task(generate_numbers(7, queue))
    collect_data_task = asyncio.create_task(collect_data(queue, timeout=1))
    values = await collect_data_task
    # The ``generate_task`` has not been terminated
    generate_task.cancel()
    print(values)


asyncio.run(wait7hz())

from typing import Callable, Dict

############################################
# We can now design a class that inherits from :class:`~tensordict.LazyStackedTensorDict` and reads data coming
# from different streams and registers them in separate tensordicts.
# A nice feature of :class:`~tensordict.LazyStackedTensorDict` is that it can be built incrementally too, such that
# we can simply register the new data coming in by extending the lazy stack up until we have collected enough data.
# Here is an implementation of this ``StreamedTensorDict`` class:
#

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

        # We construct a queue for each stream
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
            return td.densify(layout=torch.strided)
        return td


############################################
# Finally, the ``main`` function will compose the streaming functions ``stream0`` and ``stream1`` and pass them to the
# ``StreamedTensorDict.from_streams`` method which will collect ``batch_size`` batches of data for ``timeout=1`` second
# each:


async def main() -> TensorDictBase:
    def stream0(queue):
        return generate_numbers(frequency=7, queue=queue)

    def stream1(queue):
        return generate_numbers(frequency=3, queue=queue)

    # Running this should take about 10 seconds
    return await StreamedTensorDict.from_streams(
        {"bucket0": stream0, "bucket1": stream1}, timeout=1, batch_size=10
    )


td = asyncio.run(main())

print("TensorDict from stream", td)
############################################
# Let's represent the data from both streams - should be equal to torch.arange() for batch_size * timeout * Hz
#  <=> 1 * 10 secs * 3 or 7
print("bucket0 (7Hz, around 70 values)", td["bucket0"].values())
print("bucket1 (3Hz, around 30 values)", td["bucket1"].values())
print("shapes of bucket0 (7Hz, around 70 values)", td["bucket0"]._nested_tensor_size())

############################################
# Conclusion
# ----------
#
# In this tutorial, we've explored the basics of working with TensorDict and asynchronous data streams.
# We've learned how to create lazy stacks of tensors, handle asynchronous data streams using asyncio, and densify our
# data for efficient storage and processing.
#
# We've also seen how :class:`~tensordict.TensorDict` and :class:`~tensordict.LazyStackedTensorDict` can be used to
# simplify complex data processing tasks, such as bucketizing data streams with different frequencies.
# By leveraging the power of TensorDict and asyncio, you can build scalable and efficient data processing pipelines
# that can handle even the most demanding real-world applications.
#
# Thanks for following along with this tutorial! We hope you've found it helpful and informative.
#
