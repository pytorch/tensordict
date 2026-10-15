# TensorDictBase.batch_dims

*property*TensorDictBase.batch_dims*: [int](https://docs.python.org/3/builtins/functions.html#int)*

The number of batch dimensions, that is `len(self.batch_size)`.

`ndim`, `dim()` and `ndimension()` return the same value.

Examples

```
>>> from tensordict import TensorDict
>>> TensorDict(batch_size=[3, 4]).batch_dims
2
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).