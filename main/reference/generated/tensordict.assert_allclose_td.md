# assert_allclose_td

tensordict.assert_allclose_td(*actual: T*, *expected: T*, *rtol: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *atol: [float](https://docs.python.org/3/builtins/functions.html#float) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *equal_nan: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*, *intersection: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *msg: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = ''*, *prefix: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey) = ()*) → [bool](https://docs.python.org/3/builtins/functions.html#bool)

Asserts that two tensordicts have the same keys and close values.

Each pair of tensors is compared with [`torch.testing.assert_close()`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close),
which also checks that the dtypes, shapes and devices match. Nested
tensordicts are compared entry by entry, and two lazy stacks are compared
stack by stack. Non-tensor data is not compared. Tensorclasses are
accepted.

`assert_allclose_td` is another name for this function.

Parameters:

- **actual** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to check.
- **expected** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict with the expected values.
- **rtol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the relative tolerance. Defaults to `None`,
which uses the default of [`torch.testing.assert_close()`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close) for
the dtype.
- **atol** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,**optional*) - the absolute tolerance. Defaults to `None`,
which uses the default of [`torch.testing.assert_close()`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close) for
the dtype.
- **equal_nan** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, two `NaN` values are equal.
Defaults to `True`.
- **intersection** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, compares only the keys
that both tensordicts have. If `False`, different key sets
raise a `KeyError`. Defaults to `False`.
- **msg** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - a message appended to the error message.
Defaults to `""`.
- **prefix** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)*,**optional*) - a key prepended to the keys in the error
messages. Defaults to `()`.

Returns:

`True` if the check passes.

Raises:

- [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `actual` or `expected` is not a tensordict or a
 tensorclass.
- [**KeyError**](https://docs.python.org/3/builtins/exceptions.html#KeyError) - if the keys differ and `intersection=False`.
- [**AssertionError**](https://docs.python.org/3/builtins/exceptions.html#AssertionError) - if two values are not close. The message names the
 key and the root mean squared error, followed by
 [`torch.testing.assert_close()`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)'s description of the mismatch.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the shapes of two tensors cannot be broadcast
 together.

Examples

```
>>> import torch
>>> from tensordict import TensorDict, assert_close
>>> a = TensorDict(x=torch.zeros(3), y=torch.ones(3))
>>> assert_close(a, a.clone())
True
>>> b = TensorDict(x=torch.full((3,), 0.01), y=torch.ones(3))
>>> assert_close(a, b, atol=0.1, rtol=0)
True
>>> assert_close(a, b) 
Traceback (most recent call last):
 ...
AssertionError: key ('x',) does not match, got mse = 0.0100
Tensor-likes are not close!
...
```