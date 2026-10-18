# TensorDictBase.unlock

TensorDictBase.unlock_() → Self

Unlocks the tensordict and its nested tensordicts, and returns it.

After this call, entries can be added, removed and renamed again. See
[`lock_()`](tensordict.TensorDictBase.lock_.html#tensordict.TensorDictBase.lock_) for what a lock blocks.

Can be used as a context manager: the tensordict is locked again when
the `with` block exits, also if the block raises an exception.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the tensordict, or one of its nested tensordicts,
 is also an entry of another locked tensordict. Unlock that
 parent first. The tensordict stays locked.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), batch_size=[3], lock=True)
>>> with td.unlock_():
... td["b"] = torch.ones(3)
>>> td.is_locked
True
>>> sorted(td.keys())
['a', 'b']
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `unlock_`

![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict