# to_safetensors

tensordict.to_safetensors(*data: [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *path: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | Path*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Saves a tensordict or a tensorclass to a safetensors file.

Each tensor is saved under its nested key joined with `"."`, the names
that a model's `state_dict` uses. The structure of the tensordict (the
batch size and dim names of each level, the class of each tensorclass,
the non-tensor values, and which tensors are
[`UnbatchedTensor`](tensordict.UnbatchedTensor.html#tensordict.UnbatchedTensor)) is saved as one JSON string in the metadata of
the file, so that [`from_safetensors()`](tensordict.from_safetensors.html#tensordict.from_safetensors) rebuilds it.

Tensors that share memory are saved as separate copies, and a tensordict
on another device is saved through the CPU. Requires the `safetensors`
package.

Parameters:

- **data** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict or tensorclass to save.
- **path** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or**Path*) - the path of the file.

Raises:

- [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if a value cannot be stored in safetensors: a nested,
 jagged or sparse tensor, a non-tensor value that is not JSON
 (`str`, `int`, `float`, `bool`, `None`, `list` or
 `dict`), a stack of non-tensor values, a lazy stack whose
 entries have different keys, or a bare non-tensor value. A lazy
 stack of tensordicts whose entries have different shapes cannot
 be saved either.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if two keys have the same name once joined with `"."`,
 for example `"a.b"` and `("a", "b")`.

Examples

```
>>> import tempfile
>>> from pathlib import Path
>>> import torch
>>> from tensordict import TensorDict, from_safetensors, to_safetensors
>>> td = TensorDict(
... {"weight": torch.ones(2, 3), "sub": {"bias": torch.zeros(2)}},
... batch_size=[2],
... )
>>> with tempfile.TemporaryDirectory() as tmp:
... to_safetensors(td, Path(tmp) / "td.safetensors")
... loaded = from_safetensors(Path(tmp) / "td.safetensors")
>>> loaded.batch_size, loaded["sub", "bias"]
(torch.Size([2]), tensor([0., 0.]))
```