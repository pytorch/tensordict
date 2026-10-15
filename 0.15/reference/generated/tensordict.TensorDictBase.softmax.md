# TensorDictBase.softmax

TensorDictBase.softmax(*dim: [int](https://docs.python.org/3/builtins/functions.html#int)*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Apply a softmax function to the tensordict elements.

Parameters:

- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*tuple*](https://docs.python.org/3/builtins/stdtypes.html#tuple)*of**ints*) - A tensordict dimension along which softmax will be computed.
- **dtype** ([*dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - the desired data type of returned tensor.
If specified, the input tensor is cast to dtype before the operation is performed.
This is useful for preventing data type overflows.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).