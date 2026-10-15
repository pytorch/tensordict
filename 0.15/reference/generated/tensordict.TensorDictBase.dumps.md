# TensorDictBase.dumps

TensorDictBase.dumps(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *copy_existing: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, ***, *num_threads: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *return_early: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *share_non_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *robust_key: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = True*, *archive: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *compression: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Saves the tensordict to disk, and returns a memory-mapped copy of it.

`save` (and its alias `dumps`) is [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap): it takes the
same arguments. `self` is not changed. Read the result back with
[`load()`](tensordict.TensorDictBase.load.html#tensordict.TensorDictBase.load). See [Save and load tensordicts](../../saving.html#saving).

Parameters:

**prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory where the tensordict is
written, one file per leaf. If `prefix` ends with `".tdz"`
or `archive=True`, a single archive file is written instead.

Keyword Arguments:

- **archive** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, write a single archive file.
Defaults to `None`, which writes an archive if `prefix`
ends with `".tdz"`.
- **compression** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](tensordict.TensorDictBase.int.html#tensordict.TensorDictBase.int)*,**optional*) - the compression of the
archive. See [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap).

All the other arguments are those of [`memmap()`](tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap).

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), nested=TensorDict(b=torch.ones(3), batch_size=[3]), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(path)
... loaded = TensorDict.load(path)
... print((loaded == td).all())
True
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).