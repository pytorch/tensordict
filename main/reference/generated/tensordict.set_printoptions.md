# set_printoptions

*class*tensordict.set_printoptions(***, *verbose: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, ***kwargs*)

Sets the options that control what the repr of a tensordict shows.

Use it as a context manager or a decorator to change the options for a
block of code; the previous options come back at the end of the block.
Call `set_printoptions(...).set()` to change them for the rest of the
program. The options that are not passed keep their current value.
[Printing and display](../../printing.html#printing) describes every option with examples, and
[`get_printoptions()`](tensordict.get_printoptions.html#tensordict.get_printoptions) returns the current values.

Keyword Arguments:

- **verbose** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `False`, hides the device, dtype and
`is_shared` attributes, unless a keyword argument sets them.
Defaults to `True`.
- ****kwargs** - the options to set: `show_batch_size`, `show_device`,
`show_is_shared`, `show_shape`, `show_field_device`,
`show_dtype`, `show_field_is_shared`, `show_grad`,
`show_is_contiguous`, `show_is_view`, `show_storage_size`,
`plain`, `sort_keys`, `compact` and `html`. `compact=True`
prints each entry on one line as its dtype and its feature
dimensions (`f32[3]`). `html=False` turns off the HTML display
of notebooks.

Raises:

[**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if an option is unknown.

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