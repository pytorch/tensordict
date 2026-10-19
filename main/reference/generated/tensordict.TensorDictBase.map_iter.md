# TensorDictBase.map_iter

TensorDictBase.map_iter(*fn: Callable[[[TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)], [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection) | [None](https://docs.python.org/3/builtins/constants.html#None)]*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *num_workers: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *shuffle: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *chunksize: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *num_chunks: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *pool: mp.Pool | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *generator: [Generator](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *max_tasks_per_child: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *worker_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 1*, *index_with_generator: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *pbar: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *mp_start_method: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Iterator[T]

Applies a function to chunks of the tensordict in worker processes, and yields the results.

The iterator version of [`map()`](tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map): the tensordict is split along
`dim` into chunks, `fn` is called on each chunk in a pool of worker
processes, and the result for each chunk is yielded as soon as it is
ready, in the order of the chunks (unless `shuffle=True`). This can
serve as a data loader over a tensordict.

The chunks are slices of the tensordict, not copies. On a memory-mapped
tensordict (see [`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_)), the workers read them from the
files without a copy, which makes it possible to process data that is
larger than memory.

Parameters:

- **fn** (*Callable*) - the function called on each chunk, a tensordict
that holds a slice of the tensordict along `dim`. It returns
a tensordict, a tensor, or `None`. It must be picklable: define
it at the top level of a module or script, not as a lambda.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension along which the tensordict is
split into chunks. Defaults to `0`.
- **num_workers** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of worker processes.
Ignored when `pool` is given. Defaults to `None`, which uses
one worker per CPU.

Keyword Arguments:

- **shuffle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the chunks take their
elements from a random permutation of the tensordict along
`dim`, so a chunk holds elements that are not contiguous, and
the results are yielded in the order the workers finish them.
Needs `index_with_generator=True`: otherwise it raises
`RuntimeError`. Defaults to `False`.
- **chunksize** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the size of each chunk along `dim`.
`0` passes the elements one by one, unbound along `dim`.
Exclusive with `num_chunks`. Defaults to `None`.
- **num_chunks** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of chunks. Exclusive with
`chunksize`. Defaults to `None`: if neither is given, the
tensordict is split into one chunk per worker. Use more chunks
when a chunk per worker does not fit in memory.
- **pool** (*multiprocessing.Pool**,**optional*) - a pool of processes to run
the job. When it is given, `num_workers`, `generator`,
`max_tasks_per_child`, `worker_threads` and
`mp_start_method` are ignored. Defaults to `None`, which
creates a pool for the call and closes it afterwards.
- **generator** ([*Generator*](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator)*,**optional*) - the generator that draws the
base seed of the workers. Worker `i` seeds torch with the
base seed plus `i`, and NumPy with a seed derived from both,
so that the workers draw different random numbers. This makes the workers independent,
not the results reproducible: which worker gets which chunk
changes from call to call. Defaults to `None`, which uses a
new [`torch.Generator`](https://docs.pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator), so the base seed is the same on
every call, and emits a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning). Starting with
TensorDict 0.17, the default will change to drawing the base
seed from the global torch RNG, as
[`DataLoader`](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) does, which also advances
it by one draw. Pass `generator=torch.default_generator` for
that behaviour now, or `generator=torch.Generator()` to keep
the current seeds. To leave the workers unseeded, pass a
`pool`.
- **max_tasks_per_child** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of chunks a worker
process handles before it is replaced by a new one. Defaults to
`None`, which keeps the workers for the whole call.
- **worker_threads** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the number of torch threads in each
worker. Defaults to `1`.
- **index_with_generator** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, each chunk is
indexed when a worker asks for it, instead of all at once
before the job starts. This saves the memory and the time of
splitting the whole tensordict upfront, but indexing is slower
than [`split()`](tensordict.TensorDictBase.split.html#tensordict.TensorDictBase.split) and [`chunk()`](tensordict.TensorDictBase.chunk.html#tensordict.TensorDictBase.chunk). Defaults to `True`,
unlike [`map()`](tensordict.TensorDictBase.map.html#tensordict.TensorDictBase.map).
- **pbar** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, shows a progress bar. Needs
`tqdm`; without it, no bar is shown. Defaults to `False`.
- **mp_start_method** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the start method of the worker
processes, for example `"fork"` or `"spawn"`. CUDA tensors
cannot be shared with processes started with `"fork"`.
Defaults to `None`, which uses the default start method of
[`torch.multiprocessing`](https://docs.pytorch.org/docs/stable/multiprocessing.html#module-torch.multiprocessing).

Yields:

The result of `fn` for each chunk.

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
 # generator=torch.default_generator seeds the workers from
 # the global RNG, the default from TensorDict 0.17.
 for sample in data.map_iter(
 process_data,
 dim=1,
 num_workers=2,
 chunksize=5,
 generator=torch.default_generator,
 ):
 print(sample["y"])
 # tensor([[1., 1., 1., 1., 1.]])
 break
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).