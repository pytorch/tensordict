# TensorDictBase.to_struct_array

TensorDictBase.to_struct_array() → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

Converts the tensordict to a NumPy structured array, with one field per entry.

The shape of the array is the batch size. A nested tensordict becomes
a nested structured field, and the feature dimensions of a tensor
become the shape of its field. String values become `"U"` fields of
at least 10 characters. Unlike [`from_struct_array()`](tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array), this
method copies the data: the array does not share memory with the
tensordict.

Returns:

A `np.ndarray` with a structured dtype.

See also

[`from_struct_array()`](tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array), which converts back, and
[`numpy()`](tensordict.TensorDictBase.numpy.html#tensordict.TensorDictBase.numpy) to convert to a dict of NumPy arrays.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({'a': torch.tensor([1, 2, 3]), 'b': torch.tensor([4.0, 5.0, 6.0])}, batch_size=[3])
>>> arr = td.to_struct_array()
>>> print(arr)
[(1, 4.) (2, 5.) (3, 6.)]
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).