# Printing and display

`print(td)` shows the structure of a [`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
without the values of its tensors: one line for each tensor, whatever its
size. [`set_printoptions`](reference/generated/tensordict.set_printoptions.html#tensordict.set_printoptions) chooses the attributes
that the display shows, and [`parse_tensor_dict_string()`](reference/generated/tensordict.parse_tensor_dict_string.html#tensordict.parse_tensor_dict_string)
rebuilds a tensordict from a printed one. All the options are in
Available options.

## What `print` shows

By default, the display shows the shape, device, dtype, and `is_shared` of
each entry, and the batch size, device, and `is_shared` of the tensordict:

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
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([32]),
 device=None,
 is_shared=False)
```

## Choose what the display shows

If all your work is on the CPU and nothing is shared, `device=cpu` and
`is_shared=False` repeat on every line and tell you nothing.
[`set_printoptions`](reference/generated/tensordict.set_printoptions.html#tensordict.set_printoptions) selects the attributes that appear. It
works for the whole process, as a context manager, or as a decorator, as
[`torch.no_grad`](https://docs.pytorch.org/docs/stable/generated/torch.no_grad.html#torch.no_grad) does for the last two. Its role is similar to
[`torch.set_printoptions()`](https://docs.pytorch.org/docs/stable/generated/torch.set_printoptions.html#torch.set_printoptions), which is only a function.

### For the whole process

Call `set_printoptions(...).set()` to change the defaults for the rest of
the process. `show_device` and `show_is_shared` control the lines of the
tensordict, and `show_field_device` and `show_field_is_shared` the
attributes of each field:

```
>>> from tensordict import set_printoptions
>>> set_printoptions(
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
>>> with set_printoptions(show_dtype=False, show_field_is_shared=False):
... print(td) # dtype and per-field is_shared hidden
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu),
 label: Tensor(shape=torch.Size([32]), device=cpu)},
 batch_size=torch.Size([32]),
 device=None,
 is_shared=False)
>>> print(td) # back to defaults
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 label: Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([32]),
 device=None,
 is_shared=False)
```

### For one function

Decorate a function to use the options for every `repr` call in it. Here,
`show_is_shared=False` hides the `is_shared=` line of the tensordict:

```
>>> @set_printoptions(show_is_shared=False)
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
device, the dtype, and `is_shared`:

```
>>> with set_printoptions(verbose=False):
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
>>> with set_printoptions(verbose=False, show_dtype=True):
... print(td)
TensorDict(
 fields={
 image: Tensor(shape=torch.Size([32, 3, 64, 64]), dtype=torch.float32),
 label: Tensor(shape=torch.Size([32]), dtype=torch.int64)},
 batch_size=torch.Size([32]))
```

### Available options

These options control the lines of the outer `TensorDict(...)` block:

| Option | Default | Description |
| --- | --- | --- |
| `show_batch_size` | `True` | Show the `batch_size=` line. |
| `show_device` | `True` | Show the `device=` line. |
| `show_is_shared` | `True` | Show the `is_shared=` line. |

These options control the attributes in the `Tensor(...)` description of
each field:

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
values are zeros, because the repr does not contain the data:

```
>>> from tensordict import parse_tensor_dict_string
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
the default print format, with `shape=`, `device=`, `dtype=`, and
`is_shared=` for each field. It reads nested tensordicts at any depth, keys
with any character, such as `my-key`, and devices with an index, such as
`cuda:0`. It has these limits:

- It reads only tensor fields and nested tensordicts. For a non-tensor field,
it raises `ValueError: Cannot parse the field ...`.
- If [`set_printoptions`](reference/generated/tensordict.set_printoptions.html#tensordict.set_printoptions) hid a field attribute or added one
(for example `plain=True`), it raises the same `ValueError`.
- It makes each tensor on the printed device. To parse the repr of a CUDA
tensordict, run it on a machine with CUDA.
- It does not keep `is_shared`, and it returns a
[`TensorDict`](reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) for the repr of a lazy stack. For a lazy
stack with exclusive keys, it raises a `RuntimeError`
([#1956](https://github.com/pytorch/tensordict/issues/1956)).

## See also

- The reference entries of [`set_printoptions`](reference/generated/tensordict.set_printoptions.html#tensordict.set_printoptions),
[`get_printoptions()`](reference/generated/tensordict.get_printoptions.html#tensordict.get_printoptions), and
[`parse_tensor_dict_string()`](reference/generated/tensordict.parse_tensor_dict_string.html#tensordict.parse_tensor_dict_string).
- [Devices, dtypes, and memory](concepts/devices.html#devices): what `device=None` in the repr means.
- [Non-tensor data and unbatched entries](concepts/non_tensor.html#non-tensor-data): how non-tensor entries appear in the repr.