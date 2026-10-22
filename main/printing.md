# Printing and display

`print(td)` shows the structure of a [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
without the values of its tensors: the dtype and shape of each tensor,
whatever its size. [`set_printoptions`](reference/generated/tensordict.set_printoptions.html#tensordict.set_printoptions) chooses the attributes
that the display shows, and [`parse_tensor_dict_string()`](reference/generated/tensordict.parse_tensor_dict_string.html#tensordict.parse_tensor_dict_string)
rebuilds a tensordict from a printed one. All the options are in
Available options.

## What `print` shows

By default, each entry prints as its dtype and its shape after the batch
dims, and the batch size prints once, at the start. The display stays on one
line if it fits in 80 columns, and otherwise puts one entry on each line:

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... {
... "image": torch.randn(32, 3, 64, 64),
... "label": torch.randint(10, (32,)),
... },
... batch_size=[32],
... )
>>> print(td)
TensorDict(batch_size=[32], fields={image: f32[3,64,64], label: i64[]})
>>> sample = TensorDict(
... points=torch.zeros(1000, 3),
... cells=torch.zeros(1000, 4, dtype=torch.int64),
... data=TensorDict(pressure=torch.zeros(1000), mask=torch.zeros(1000, dtype=torch.bool)),
... batch_size=[1000],
... device="cpu",
... )
>>> print(sample)
TensorDict(batch_size=[1000], device=cpu, fields={
 cells: i64[4],
 data: {mask: bool[], pressure: f32[]},
 points: f32[3]})
```

The dtype codes are those of JAX and polars: `f32` for `torch.float32`,
`i64` for `torch.int64`, `u8`, `c64`, `bf16`, `bool`. The rules:

- A nested tensordict prints as `{...}`, and with its class name when it is
not a [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) (`Mesh{...}` for a tensorclass). A
nested tensordict with more batch dims than its parent shows them first:
`sub: [5]{y: f32[3]}`.
- In a lazy stack, a dim whose size differs between the members prints as
`-1`, the header shows `stack_dim=`, and the keys that only some members
have follow under `exclusive_fields=`. A lazy stack nested in another
entry shows only the keys that all its members have; the detailed display
(`compact=False`) shows the others.
- A non-tensor entry prints as the type name of its value (`str`). In a
[`PersistentTensorDict`](reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict), it prints as `NonTensorData` or
`NonTensorStack` instead, since reading the type of the value would read
the file.
- When the tensordict has no device (`device=None`), a leaf that is not on
the CPU shows its device: `f32[3]@cuda:0`.
- `show_batch_size`, `show_device`, `show_dtype` and `sort_keys`
apply; `show_dtype=False` (or `verbose=False`) prints the shapes only,
as `[3]`. The other options (`html` aside) only apply to the detailed
display: passing one of them without `compact`, such as
`set_printoptions(show_grad=True)`, switches to the detailed display. With
an explicit `compact=True`, the compact display ignores them.
- The display reads no data and syncs no device: it uses only the shape,
dtype and device of each leaf, and stacks no leaf of a lazy stack.
- A [`TensorDictStore`](reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) and a
[`NonTensorStack`](reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) keep their usual display; nested in
another tensordict, a store prints as its class name (`TensorDictStore`),
since its entries would read the server.

### The detailed display: `compact=False`

With `compact=False`, the display shows the full shape, device, dtype, and
`is_shared` of each entry on its own line, and the batch size, device, and
`is_shared` of the tensordict. This was the default display in earlier releases:

```
>>> from tensordict import set_printoptions
>>> with set_printoptions(compact=False):
... print(td)
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([32]),
 device=None,
 is_shared=False)
```

## Choose what the display shows

[`set_printoptions`](reference/generated/tensordict.set_printoptions.html#tensordict.set_printoptions) selects the format and the attributes
that appear. Most options choose the attributes of the detailed display: if
all your work is on the CPU and nothing is shared, `device=cpu` and
`is_shared=False` repeat on every line of it and tell you nothing. It
works for the whole process, as a context manager, or as a decorator, as
[`torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad) does for the last two. Its role is similar to
[`torch.set_printoptions()`](https://docs.pytorch.org/docs/stable/generated/torch.set_printoptions.html#torch.set_printoptions), which is only a function.

### For the whole process

Call `set_printoptions(...).set()` to change the defaults for the rest of
the process. `show_device` and `show_is_shared` control the lines of the
tensordict, and `show_field_device` and `show_field_is_shared` the
attributes of each field:

```
>>> set_printoptions(
... compact=False,
... show_device=False,
... show_is_shared=False,
... show_field_device=False,
... show_field_is_shared=False,
... ).set()
>>> print(td)
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), dtype=torch.float32),
 label: Tensor(shape=torch.Size([32]), dtype=torch.int64)},
 batch_size=torch.Size([32]))
```

`set()` changes only the options that it gets, so
`set_printoptions().set()` without arguments does not reset anything. To go
back to the defaults, set the changed options again:

```
>>> set_printoptions(
... compact=True,
... show_device=True,
... show_is_shared=True,
... show_field_device=True,
... show_field_is_shared=True,
... ).set()
```

### For one block of code

Use the context manager to change the options for one block of code. At the
end of the block, the previous options come back:

```
>>> with set_printoptions(compact=False, show_dtype=False, show_field_is_shared=False):
... print(td) # dtype and per-field is_shared hidden
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu),
 label: Tensor(shape=torch.Size([32]), device=cpu)},
 batch_size=torch.Size([32]),
 device=None,
 is_shared=False)
>>> print(td) # back to defaults
TensorDict(batch_size=[32], fields={image: f32[3,64,64], label: i64[]})
```

### For one function

Decorate a function to use the options for every `repr` call in it. Here,
`show_is_shared=False` hides the `is_shared=` line of the tensordict:

```
>>> @set_printoptions(compact=False, show_is_shared=False)
... def summarize(td):
... print(td)
>>> summarize(td)
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([32]),
 device=None)
```

### Shapes only: `verbose=False`

To see only the shapes, pass `verbose=False`. This one argument hides the
dtype in the default display, and the device, the dtype, and `is_shared` in
the detailed one:

```
>>> with set_printoptions(verbose=False):
... print(td)
TensorDict(batch_size=[32], fields={image: [3,64,64], label: []})
>>> with set_printoptions(compact=False, verbose=False):
... print(td)
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64])),
 label: Tensor(shape=torch.Size([32]))},
 batch_size=torch.Size([32]))
```

`verbose=True` (the default) changes nothing. Explicit keyword arguments
take precedence over `verbose`, so you can mix the two:

```
>>> with set_printoptions(compact=False, verbose=False, show_dtype=True):
... print(td)
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), dtype=torch.float32),
 label: Tensor(shape=torch.Size([32]), dtype=torch.int64)},
 batch_size=torch.Size([32]))
```

### Notebook display

In a notebook (Jupyter, VS Code, Colab), a tensordict or a tensorclass that
ends a cell displays as HTML: the entries of the compact display, one on each
line, with each nested tensordict in a block that you can fold. The header
shows the class name, the batch size, the dim names and the device, and a
nested tensordict shows its class name and number of entries. The display
uses the colors of the notebook theme, so it reads in light and dark themes.

- It reads the same metadata as the compact display: no data, no device
sync, and no stacking of a lazy stack. `show_batch_size`,
`show_device`, `show_dtype` and `sort_keys` apply.
- It shows at most 200 entries; the rest of each block shows as
`... and K more`.
- A [`TensorDictStore`](reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore), a
[`NonTensorStack`](reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack), a [`NonTensorData`](reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData),
and any tensordict or tensorclass with its own text display (such as
[`TensorDictParams`](reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams), or a class that defines
`__repr__`) keep their text display, as does any tensordict whose HTML
display fails. Nested in another tensordict, a
[`TensorDictStore`](reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) shows as its class name.
- `set_printoptions(html=False)` turns the HTML display off, and the
notebook shows the text display instead.

### Available options

These options choose the format:

| Option | Default | Description |
| --- | --- | --- |
| `compact` | `True` | Print each entry as its dtype and its shape after the batch dims (`f32[3]`), on one line when it fits. See What print shows. With `False`, print the detailed display that the options below describe. |
| `html` | `True` | Display tensordicts as HTML in notebooks. See Notebook display. |

These options control the lines of the outer `TensorDict(...)` block of the
detailed display:

| Option | Default | Description |
| --- | --- | --- |
| `show_batch_size` | `True` | Show the `batch_size=` line. |
| `show_device` | `True` | Show the `device=` line. |
| `show_is_shared` | `True` | Show the `is_shared=` line. |

These options control the attributes in the `Tensor(...)` description of
each field in the detailed display (`show_dtype` also applies to the compact
one):

| Option | Default | Description |
| --- | --- | --- |
| `show_shape` | `True` | Show the `shape=` attribute. |
| `show_field_device` | `True` | Show the `device=` attribute. |
| `show_dtype` | `True` | Show the `dtype=` attribute. |
| `show_field_is_shared` | `True` | Show the `is_shared=` attribute. |

This option is a shortcut for several of the options above:

| Option | Default | Description |
| --- | --- | --- |
| `verbose` | `True` | When `False`, hides `show_device`, `show_is_shared`, `show_field_device`, `show_dtype`, and `show_field_is_shared`. Explicit keyword arguments override `verbose`. |

These options add attributes that are hidden by default, and change the order
of the keys:

| Option | Default | Description |
| --- | --- | --- |
| `show_grad` | `False` | Show `requires_grad=`. |
| `show_is_contiguous` | `False` | Show `is_contiguous=`. |
| `show_is_view` | `False` | Show `is_view=` (whether `._base` is set). |
| `show_storage_size` | `False` | Show `storage_size=` (bytes). |
| `plain` | `False` | Append a short value summary (mean/std for floats, min/max for ints). |
| `sort_keys` | `"alphabetical"` | Key ordering: `"alphabetical"` (default), `"insertion"` (dict order), or a callable passed as `key` to [`sorted()`](https://docs.python.org/3/builtins/functions.html#sorted). |

### Read the current options

[`get_printoptions()`](reference/generated/tensordict.get_printoptions.html#tensordict.get_printoptions) returns a dict with the current values:

```
>>> from tensordict import get_printoptions
>>> get_printoptions()
{'show_batch_size': True, 'show_device': True, ...}
```

## Rebuild a tensordict from its printed form

When you debug, you sometimes have only the printed form of a tensordict, for
example from a log file or from the terminal of a colleague.
[`parse_tensor_dict_string()`](reference/generated/tensordict.parse_tensor_dict_string.html#tensordict.parse_tensor_dict_string) builds a
[`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) from that string. The result has the keys,
shapes, dtypes, batch size, and device of the printed tensordict. All the
values are zeros, because the repr does not contain the data. It reads the
default display:

```
>>> from tensordict import parse_tensor_dict_string
>>> td = parse_tensor_dict_string(
... "TensorDict(batch_size=[32], fields={image: f32[3,64,64], label: i64[]})"
... )
>>> td["image"].shape, td["label"].dtype
(torch.Size([32, 3, 64, 64]), torch.int64)
```

and the detailed one, which older logs hold:

```
>>> s = """TensorDict(
... fields={
... image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
... label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
... batch_size=torch.Size([32]),
... device=cpu,
... is_shared=False)"""
>>> td = parse_tensor_dict_string(s)
>>> td.batch_size
torch.Size([32])
>>> td["image"].shape
torch.Size([32, 3, 64, 64])
```

[`parse_tensor_dict_string()`](reference/generated/tensordict.parse_tensor_dict_string.html#tensordict.parse_tensor_dict_string) is a debugging aid. It reads
both formats with the default options: in the detailed one, `shape=`,
`device=`, `dtype=`, and `is_shared=` for each field. It reads nested tensordicts at any depth, keys
with any character, such as `my-key`, and devices with an index, such as
`cuda:0`. It has these limits:

- It reads only tensor fields and nested tensordicts. For a non-tensor field,
a nested tensorclass or lazy stack, or an entry of a lazy stack whose shape
differs between the members (`-1`), it raises
`ValueError: Cannot parse the field ...`.
- If [`set_printoptions`](reference/generated/tensordict.set_printoptions.html#tensordict.set_printoptions) hid a field attribute or added one
(for example `show_dtype=False`, or `plain=True` in the detailed
display), it raises the same `ValueError`.
- The compact display does not show the device of a nested tensordict, so the
parser puts a nested tensordict, and the leaves without an `@device`
suffix, on the device of its parent, or on the CPU. It shows an
[`UnbatchedTensor`](reference/generated/tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor) by its full shape, which the parser
reads as the shape after the batch dims of a batched tensor.
- It makes each tensor on the printed device. To parse the repr of a CUDA
tensordict, run it on a machine with CUDA.
- It does not keep `is_shared`, and it returns a
[`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) for the repr of a lazy stack. For a lazy
stack with exclusive keys, it raises
`ValueError: Cannot parse the exclusive_fields of a lazy stack ...`.

## See also

- The reference entries of [`set_printoptions`](reference/generated/tensordict.set_printoptions.html#tensordict.set_printoptions),
[`get_printoptions()`](reference/generated/tensordict.get_printoptions.html#tensordict.get_printoptions), and
[`parse_tensor_dict_string()`](reference/generated/tensordict.parse_tensor_dict_string.html#tensordict.parse_tensor_dict_string).
- [Devices, dtypes, and memory](concepts/devices.html#devices): what `device=None` in the repr means.
- [Non-tensor data and unbatched entries](concepts/non_tensor.html#non-tensor-data): how non-tensor entries appear in the repr.