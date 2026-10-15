# from_struct_array

tensordict.from_struct_array(*struct_array*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Converts a structured numpy array to a TensorDict.

See also

[`TensorDictBase.from_struct_array()`](tensordict.TensorDictBase.from_struct_array.html#tensordict.TensorDictBase.from_struct_array) for more information.

Examples

```
>>> import numpy as np
>>> from tensordict import from_struct_array
>>> x = np.array(
... [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
... dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
... )
>>> td = from_struct_array(x)
>>> x_recon = td.to_struct_array()
>>> assert (x_recon == x).all()
>>> assert x_recon.shape == x.shape
>>> # Try modifying x age field and check effect on td
>>> x["age"] += 1
>>> assert (td["age"] == np.array([10, 4])).all()
```