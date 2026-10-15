# TensorDictBase.to_struct_array

TensorDictBase.to_struct_array() → [ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html#numpy.ndarray)

Converts a tensordict to a numpy structured array.

In a [`from_struct_array()`](tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array) - `to_struct_array()` loop, the content of the input and output arrays should match.
However, to_struct_array will not keep the memory content of the original arrays.

See also

[`from_struct_array()`](tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array) for more information.

See also

[`numpy()`](tensordict.TensorDictBase.numpy.html#tensordict.TensorDictBase.numpy) to convert to a dictionary of numpy arrays.

Returns:

A numpy structured array representation of the input TensorDict.

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