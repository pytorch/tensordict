# TensorDictFuture

*class*tensordict.utils.TensorDictFuture(*futures*, *resulting_td*)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A tensordict that is still being written by background threads.

[`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap),
[`memmap_()`](tensordict.TensorDictBase.memmap_.html#tensordict.TensorDictBase.memmap_),
[`memmap_like()`](tensordict.TensorDictBase.memmap_like.html#tensordict.TensorDictBase.memmap_like),
[`save()`](tensordict.TensorDictBase.save.html#tensordict.TensorDictBase.save) and
[`consolidate()`](tensordict.TensorDictBase.consolidate.html#tensordict.TensorDictBase.consolidate) return one when called
with `return_early=True` and `num_threads > 1`. Call `result()`
to wait for the writes and get the tensordict. You do not build one
yourself.

Parameters:

- **futures** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*Future*](https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.Future)) - the writes to wait for.
- **resulting_td** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict that `result()`
returns.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... future = td.memmap(path, num_threads=2, return_early=True)
... result = future.result()
... print(result.is_memmap(), result["b"].tolist())
True [1.0, 1.0, 1.0]
```

result()

Waits for the writes to finish and returns the tensordict.

If a write failed, raises the first error of the writes.