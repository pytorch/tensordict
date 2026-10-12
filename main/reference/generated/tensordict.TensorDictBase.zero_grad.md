# TensorDictBase.zero_grad

TensorDictBase.zero_grad(*set_to_none: [bool](https://docs.python.org/3/builtins/functions.html#bool) = True*) → Self

Zeros all the gradients of the TensorDict recursively.

Parameters:

**set_to_none** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, tensor.grad will be `None`,
otherwise `0`.
Defaults to `True`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).