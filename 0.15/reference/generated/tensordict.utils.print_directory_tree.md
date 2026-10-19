# print_directory_tree

tensordict.utils.print_directory_tree(*path*, *indent=''*, *display_metadata=True*) → [str](https://docs.python.org/3/builtins/stdtypes.html#str)

Logs the tree of files and directories under `path`.

Each line is logged at the `INFO` level with the tensordict logger
([`tensordict.utils.logger`](../utils.html#tensordict.utils.logger)) and the tree is returned as a string.
The entries of a directory are in the order of [`os.listdir()`](https://docs.python.org/3/library/os.html#os.listdir).

Parameters:

- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - the directory (or file) to show.
- **indent** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the string put before each line of the tree;
each level adds four spaces to it. Defaults to `""`.
- **display_metadata** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the first line gives
the total size of the files under `path`. Defaults to `True`.

Returns:

The lines that were logged, joined by newlines.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.utils import print_directory_tree
>>> td = TensorDict(a=torch.zeros(3), b=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.memmap(path)
... tree = print_directory_tree(path, display_metadata=False)
>>> sorted(line.strip() for line in tree.splitlines()[1:])
['a.memmap', 'b.memmap', 'meta.json']
```