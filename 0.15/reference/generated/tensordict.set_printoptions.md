# set_printoptions

*class*tensordict.set_printoptions(***, *verbose: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***kwargs*)

Controls which attributes appear in TensorDict's `__repr__` output.

Use it as a context manager or a decorator to change the options for a
block of code, or call `set_printoptions(...).set()` to change them
for the rest of the program. [Printing and display](../../printing.html#printing) describes every option
with examples.

Parameters:

- **verbose** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, the device, dtype and
`is_shared` attributes are hidden, unless a keyword argument
sets them. Defaults to `True`.
- ****kwargs** - the options to set: `show_batch_size`, `show_device`,
`show_is_shared`, `show_shape`, `show_field_device`,
`show_dtype`, `show_field_is_shared`, `show_grad`,
`show_is_contiguous`, `show_is_view`, `show_storage_size`,
`plain` and `sort_keys`. An unknown option raises a
`TypeError`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, set_printoptions
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3])
>>> with set_printoptions(verbose=False):
... print(td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]))},
 batch_size=torch.Size([3]))
```