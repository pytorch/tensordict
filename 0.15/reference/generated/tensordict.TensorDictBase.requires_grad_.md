# TensorDictBase.requires_grad

TensorDictBase.requires_grad_(*requires_grad=True*) → Self

Change if autograd should record operations on this tensor: sets this tensor's requires_grad attribute in-place.

Returns this tensordict.

Parameters:

**requires_grad** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - whether or not autograd should record operations on this tensordict.
Defaults to `True`.

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).