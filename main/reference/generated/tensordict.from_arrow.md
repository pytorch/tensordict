# from_arrow

tensordict.from_arrow(*table: [Any](https://docs.python.org/3/library/typing.html#typing.Any)*, ***, *copy: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *nulls: [Literal](https://docs.python.org/3/library/typing.html#typing.Literal)['raise', 'fill', 'mask'] = 'raise'*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDict](tensordict.TensorDict.html#tensordict.TensorDict)

Converts an Arrow table to a tensordict, with one entry per column, viewing the data where it can.

Requires pyarrow 15 or later. The columns convert as follows:

- integer and floating-point columns become tensors that view the Arrow
data through DLPack. Boolean columns are copied, since Arrow packs them
as bits.
- `list` and `large_list` columns of one of these types become jagged
nested tensors (`layout=torch.jagged`) that view the values.
- `fixed_size_list` and `arrow.fixed_shape_tensor` columns become
tensors with trailing dimensions.
- `struct` columns become nested tensordicts.
- all the other columns, such as strings, binary, dictionaries,
timestamps, and lists of lists or of strings, become a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack) of the Python objects of
`pyarrow.Array.to_pylist`.

A column made of more than one chunk is first concatenated with
`combine_chunks`, which copies it. The tensors that view the Arrow data
keep its buffers alive, and so the whole table that they come from.

Parameters:

**table** (*pyarrow.Table**or**pyarrow.RecordBatch*) - the table. Other objects
go through `pyarrow.table()` first: for example a Polars
DataFrame, or any object that has an `__arrow_c_stream__`
method. A Hugging Face dataset holds its table in
`dataset.data.table`: after `select`, `shuffle` or
`filter`, call `dataset = dataset.flatten_indices()` first,
since this table holds all the rows of the original dataset.

Keyword Arguments:

- **copy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether to copy the data. `None` views the
buffers that Arrow marks as mutable, and copies the immutable
ones, such as those of a memory-mapped file (for example a Hugging
Face dataset loaded from disk). `True` copies every column.
`False` views every column that it can: an in-place write into a
view of an immutable buffer then crashes the process, since torch
has no read-only tensors. Defaults to `None`.
- **nulls** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - what to do with null values. `"raise"`
raises a `ValueError`. `"fill"` fills them with `NaN` in
floating-point tensors, `0` (`False`) in the other tensors, an
empty list in jagged tensors, and `None` in a
[`NonTensorStack`](tensordict.NonTensorStack.html#tensordict.NonTensorStack). A null struct makes each of
its fields null. `"mask"` fills them as `"fill"` does, and adds
the boolean entry `("masks", *key)` for each entry `key` that
has nulls, `True` where the row is not null, as
[`pad_sequence()`](tensordict.pad_sequence.html#tensordict.pad_sequence) does with `return_mask=True`.
Filling copies the column. Defaults to `"raise"`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of the
result. Defaults to `None`, which keeps the tensors on the CPU
and gives a tensordict without a device.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) with the batch size `[num_rows]`.

See also

[`to_arrow()`](tensordict.to_arrow.html#tensordict.to_arrow), the reverse conversion.

Examples

```
>>> import pyarrow as pa
>>> from tensordict import from_arrow
>>> table = pa.table(
... {
... "x": pa.array([1.0, 2.0, 3.0]),
... "ids": pa.array([[1, 2], [3], []]),
... "text": pa.array(["a", "b", "c"]),
... }
... )
>>> td = from_arrow(table)
>>> td["x"]
tensor([1., 2., 3.], dtype=torch.float64)
>>> td["ids"].unbind(0)
(tensor([1, 2]), tensor([3]), tensor([], dtype=torch.int64))
>>> td.get("text").tolist()
['a', 'b', 'c']
>>> td = from_arrow(pa.table({"y": pa.array([1.0, None])}), nulls="mask")
>>> td["y"], td["masks", "y"]
(tensor([1., nan], dtype=torch.float64), tensor([ True, False]))
```