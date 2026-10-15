# assert_allclose_td

tensordict.assert_allclose_td(*actual: T*, *expected: T*, *rtol: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *atol: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *equal_nan: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *intersection: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *msg: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = ''*, *prefix: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) = ()*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Asserts that two tensordicts, actual and expected, are element-wise equal within a tolerance for all entries.

This function checks if the elements of the actual tensor are close to the corresponding elements
of the expected tensordict, within a relative tolerance (rtol) and an absolute tolerance (atol).

It is similar to the [`assert_close()`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close) function in PyTorch, but with tensordicts inputs.

Parameters:

- **actual** (*T*) - The tensordict containing actual values.
- **expected** (*T*) - The tensordict containing expected values.
- **rtol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*|**None**,**optional*) - The relative tolerance parameter. Default is None.
- **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*|**None**,**optional*) - The absolute tolerance parameter. Default is None.
- **equal_nan** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - If True, `NaNs` will be considered equal to `NaNs`. Default is `True`.
- **intersection** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - If True, only the intersection of the two tensordicts will be compared.
Default is `False`.
- **msg** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - An optional message to include in the assertion error if the check fails.
- **prefix** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - a prefix to add to the key for error messages.

Returns:

True if the tensors are close within the specified tolerances, raise an exception otherwise.

Return type:

[*bool*](https://docs.python.org/3/builtins/functions.html#bool)

Raises:

[**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) - If the tensordicts are not close within the specified tolerances.