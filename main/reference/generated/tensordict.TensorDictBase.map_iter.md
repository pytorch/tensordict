# TensorDictBase.map_iter

TensorDictBase.map_iter(*fn: Callable[[[TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)], [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [None](https://docs.python.org/3/builtins/constants.html#None)]*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *num_workers: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *shuffle: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *chunksize: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *num_chunks: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *pool: mp.Pool | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *generator: [Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *max_tasks_per_child: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *worker_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*, *index_with_generator: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *pbar: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *mp_start_method: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Iterator[T]

Maps a function to splits of the tensordict across one dimension iteratively.

This is the iterable version of [`map()`](tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map).

This method will apply a function to a tensordict instance by chunking
it in tensordicts of equal size and dispatching the operations over the
desired number of workers. It will yield the results one at a time.

The function signature must be `Callable[[TensorDict], TensorDict | Tensor]`.
The function must be serializable.

Note

This method is particularly useful when working with large
datasets stored on disk (for example, memory-mapped tensordicts).
The chunks are slices of the original data that are not copied,
and the processes receive them without a data transfer. This way,
you can process datasets larger than the memory (for example,
over 1 TB).

Note

You can use this function to represent a dataset and load from
it, as a dataloader does.

Parameters:

- **fn** (*callable*) - function to apply to the tensordict.
Signatures similar to `Callable[[TensorDict], TensorDict | Tensor]`
are supported.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dim along which the tensordict will be chunked.
- **num_workers** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of workers. Exclusive with `pool`.
If none is provided, the number of workers will be set to the
number of cpus available.

Keyword Arguments:

- **shuffle** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - whether the indices should be globally shuffled.
If `True`, each batch will contain non-contiguous samples.
If `index_with_generator=False` and shuffle=True`, an error will be raised.
Defaults to `False`.
- **chunksize** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - The size of each chunk of data.
A `chunksize` of 0 will unbind the tensordict along the
desired dimension and restack it after the function is applied,
whereas `chunksize>0` will split the tensordict and call
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) on the resulting list of tensordicts.
If none is provided, the number of chunks will equate the number
of workers. For very large tensordicts, such large chunks
may not fit in memory for the operation to be done and
more chunks may be needed to make the operation practically
doable. This argument is exclusive with `num_chunks`.
- **num_chunks** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the number of chunks to split the tensordict
into. If none is provided, the number of chunks will equate the number
of workers. For very large tensordicts, such large chunks
may not fit in memory for the operation to be done and
more chunks may be needed to make the operation practically
doable. This argument is exclusive with `chunksize`.
- **pool** (*mp.Pool**,**optional*) - a multiprocess Pool instance to use
to execute the job. If none is provided, a pool will be created
within the `map` method.
- **generator** ([*Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,**optional*) - 

a generator to use for seeding.
A base seed will be generated from it, and each worker
of the pool will be seeded with the provided seed incremented
by a unique integer from `0` to `num_workers`. If no generator
is provided, a random integer will be used as seed.
To work with unseeded workers, a pool should be created separately
and passed to [`map()`](tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map) directly.

Note

Caution should be taken when providing a low-valued seed as
this can cause autocorrelation between experiments, example:
if 8 workers are asked and the seed is 4, the workers seed will
range from 4 to 11. If the seed is 5, the workers seed will range
from 5 to 12. These two experiments will have an overlap of 7
seeds, which can have unexpected effects on the results.

Note

The goal of seeding the workers is to have independent seed on
each worker, and NOT to have reproducible results across calls
of the map method. In other words, two experiments may and
probably will return different results as it is impossible to
know which worker will pick which job. However, we can make sure
that each worker has a different seed and that the pseudo-random
operations on each will be uncorrelated.
- **max_tasks_per_child** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the maximum number of jobs picked
by every child process. Defaults to `None`, i.e., no restriction
on the number of jobs.
- **worker_threads** ([*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the number of threads for the workers.
Defaults to `1`.
- **index_with_generator** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - 

if `True`, the splitting / chunking
of the tensordict will be done during the query, sparing init time.
Note that [`chunk()`](tensordict.TensorDictBase.chunk.html#tensordict.TensorDictBase.chunk) and [`split()`](tensordict.TensorDictBase.split.html#tensordict.TensorDictBase.split) are much more
efficient than indexing (which is used within the generator)
so a gain of processing time at init time may have a negative
impact on the total runtime. Defaults to `True`.

Note

The default value of `index_with_generator` differs for `map_iter`
and `map` and the former assumes that it is prohibitively expensive to
store a split version of the TensorDict in memory.
- **pbar** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a progress bar will be displayed.
Requires tqdm to be available. Defaults to `False`.
- **mp_start_method** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the start method for multiprocessing.
If not provided, the default start method will be used.
Accepted strings are `"fork"` and `"spawn"`. Keep in mind that
`"cuda"` tensors cannot be shared between processes with the
`"fork"` start method. This is without effect if the `pool`
is passed to the `map` method.

Examples

Run this example as a script, not in a REPL: the worker processes
must import `process_data`, and the `if __name__ == "__main__"`
block keeps them from running the main code again.

```
import torch
from tensordict import TensorDict

def process_data(data):
 # The chunks of a memory-mapped tensordict are locked:
 # return a new tensordict instead of adding a key to data.
 return data.empty().set("y", data["x"] + 1)

if __name__ == "__main__":
 data = TensorDict({"x": torch.zeros(1, 1_000)}, [1, 1_000]).memmap_()
 for sample in data.map_iter(process_data, dim=1, num_workers=2, chunksize=5):
 print(sample["y"])
 # tensor([[1., 1., 1., 1., 1.]])
 break
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).