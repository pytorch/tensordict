# TensorDictBase.replace

TensorDictBase.replace(**args*, ***kwargs*)

Returns a shallow copy of the tensordict with some entries replaced.

The copy is made with [`copy()`](tensordict.TensorDictBase.copy.html#tensordict.TensorDictBase.copy), and the new entries are written
into it with [`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update): nested tensordicts are merged, and the
tensordict does not change. With no entry to write, `replace`
returns `self`.

When both a positional argument and `**kwargs` are given, the
keyword entries are merged into a copy of the positional argument
(with `dict` unpacking for a dictionary, and with [`copy()`](tensordict.TensorDictBase.copy.html#tensordict.TensorDictBase.copy) and
[`update()`](tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) for a tensordict), so the argument does not change.
A file or a store, such as a
[`PersistentTensorDict`](tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict), or an indexed view of one,
shares its storage with its copy. So where the keyword entries go (the
argument itself, each member of a lazy stack, and each nested
tensordict that they merge into), a file or a store is replaced by a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) that holds its entries; the other
entries stay views of the file or store. Called on a file or a
store, or on a tensordict that holds one as a nested entry that the
new entries merge into, `replace` writes the new entries into the
file or store.

Parameters:

- ***args** ([*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - at most one, the entries to write.
- ****kwargs** - top-level entries to write, by key. They take
precedence over the entries of the positional argument.

Returns:

A shallow copy of the tensordict, or `self` if there is no entry
to write.

Raises:

[**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if more than one positional argument is passed, or if
 it is not a dictionary or a tensordict.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b=torch.zeros(3), batch_size=[3])
>>> new = td.replace(a=torch.ones(3))
>>> new["a"], td["a"], new["b"] is td["b"]
(tensor([1., 1., 1.]), tensor([0., 0., 0.]), True)
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `replace`

![](../../_images/sphx_glr_tensorclass_particles_thumb.png)

[Simulating particles with a tensorclass](../../tutorials/tensorclass_particles.html)

Simulating particles with a tensorclass