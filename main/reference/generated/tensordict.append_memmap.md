# append_memmap

tensordict.append_memmap(*prefix: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [Path](https://docs.python.org/3/library/pathlib.html#pathlib.Path)*, *rows: [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)*, ***, *allow_pickle: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Appends rows to a tensordict saved in a directory, along its first batch dimension.

Each leaf file grows by the bytes of the new rows, and the `meta.json`
files are rewritten last, each one through a temporary file, so that a
failed append leaves the saved data as it was. Readers that mapped the
files before the append keep seeing the old number of rows, and
[`load_memmap()`](tensordict.load_memmap.html#tensordict.load_memmap) after the append sees the new one. Use one
writer at a time. The `meta.json` files of nested levels are replaced
one at a time, so an append that is killed midway, or a
[`load_memmap()`](tensordict.load_memmap.html#tensordict.load_memmap) that runs during an append, can see
nested levels with different numbers of rows: don't read a directory
while you append to it. Each append maps the whole saved tree, as
[`load_memmap()`](tensordict.load_memmap.html#tensordict.load_memmap) does, to read its structure, so many small
appends to a save with many leaves are slow. See [Save and load tensordicts](../../saving.html#saving).

`rows` must have the keys, the classes, the dtypes and the trailing
shapes of the saved tensordict, and its batch size may differ from the
saved one in its first dimension only. A non-tensor entry stays as
saved, and its value in `rows` must be equal to it: either one value or
a [`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of equal values. A saved
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack), a lazy stack, a nested tensor and an
archive cannot grow, and raise. Everything is checked before the first
file is written.

Parameters:

- **prefix** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the directory that [`save()`](tensordict.save.html#tensordict.save) or
[`memmap()`](tensordict.memmap.html#tensordict.memmap) wrote.
- **rows** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass*) - the rows to append.

Keyword Arguments:

**allow_pickle** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether the pickled non-tensor entries
of the saved tensordict, which are compared with those of
`rows`, may be unpickled, as in
[`load_memmap()`](tensordict.load_memmap.html#tensordict.load_memmap). Defaults to `None`.

Raises:

[**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if `prefix` is not a directory, or if `rows` cannot be
 appended to the saved tensordict.

Examples

```
>>> import tempfile
>>> import torch
>>> import tensordict
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3, 2), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... _ = tensordict.save(td, tmpdir)
... tensordict.append_memmap(tmpdir, TensorDict(a=torch.ones(2, 2), batch_size=[2]))
... print(tensordict.load_memmap(tmpdir)["a"].sum(-1).tolist())
[0.0, 0.0, 0.0, 2.0, 2.0]
```