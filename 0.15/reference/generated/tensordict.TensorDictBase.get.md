# TensorDictBase.get

TensorDictBase.get(*key*)

TensorDictBase.get(*key*, *default*)

Returns the value stored at `key`, or `default` if the key is missing.

`td.get(key)` is the same as `td[key]`, except for a missing key:
`td[key]` raises `KeyError`, and `get` returns `default`.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key to read. A tuple reads a nested entry:
`td.get(("a", "b"))` is `td["a"]["b"]`. See [Keys, nested keys, and locking](../../concepts/keys.html#keys).
- **default** (*optional*) - the value returned if the key is missing.
Defaults to `None`, as for [`dict.get()`](https://docs.python.org/3/builtins/stdtypes.html#dict.get). A nested key
that runs through a tensor is missing too: `get` returns
the default passed to it, and without one raises a
`ValueError` that names the tensor.

Changed in version 0.7: A missing key returns `None` instead of raising `KeyError`.
`set_get_defaults_to_none(False)` and `TD_GET_DEFAULTS_TO_NONE=0`
restore the old behavior. They are deprecated and will be removed
in 0.17.

Keyword Arguments:

- **as_list** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a lazy stack returns the
entries as a list, one item per stacked tensordict.
Exclusive with `as_padded_tensor` and `as_nested_tensor`.
Defaults to `False`.
- **as_padded_tensor** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a lazy stack pads
the entries to the largest shape and returns them as one
tensor: each dim is padded to its largest size, also across
the levels of a lazy stack of lazy stacks. `padding_value`
sets the padding value and `padding_side` sets the side of
every padded dim.
Exclusive with `as_list` and `as_nested_tensor`.
Defaults to `False`.
- **as_nested_tensor** ([*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - if `True`, a lazy stack returns
the entries as a nested tensor ([`torch.nested`](https://docs.pytorch.org/docs/stable/nested.html#module-torch.nested)). `layout`
sets its layout. A lazy stack of lazy stacks raises a
`ValueError`, as a nested tensor has a single nested level.
Exclusive with `as_list` and `as_padded_tensor`.
Defaults to `False`.
- **layout** ([*layout*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout)*,**optional*) - the layout of the nested tensor
when `as_nested_tensor=True`: `torch.jagged` or
`torch.strided`. Defaults to `torch.jagged`.
- **padding_side** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)) - The side of padding. Must be "left" or "right". Defaults to "right".
- **padding_value** (*scalar**or*[*bool*](tensordict.TensorDictBase.bool.html#tensordict.TensorDictBase.bool)*,**optional*) - The padding value. Defaults to 0.0.

A lazy stack reads the entry of every stacked tensordict and stacks
the results into a new tensor. If the entry shapes differ, the plain
call raises a `RuntimeError`: pass `as_list`, `as_padded_tensor`
or `as_nested_tensor` to get a list, a padded tensor or a nested
tensor instead.

Note

A dense tensordict ignores the keyword arguments today and
warns for `as_list=True` and `as_nested_tensor=True`. Starting
with TensorDict 0.17, the default will change to returning a list
or a nested tensor, as a lazy stack does.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict({"x": 1, "nested": {"y": 2}}, batch_size=[])
>>> td.get("x")
tensor(1)
>>> td.get(("nested", "y"))
tensor(2)
>>> print(td.get("z"))
None
>>> td.get("z", 0)
0
```

```
>>> import torch
>>> from tensordict import lazy_stack
>>> td = lazy_stack([
... TensorDict({"x": torch.ones(1,)}),
... TensorDict({"x": torch.ones(2,) * 2}),
... ])
>>> nested = td.get("x", as_nested_tensor=True)
>>> nested.is_nested, nested.offsets()
(True, tensor([0, 1, 3]))
>>> td.get("x", as_padded_tensor=True)
tensor([[1., 0.],
 [2., 2.]])
```

Every tensordict class and every tensorclass has this method. For the
other methods, see [Methods by task](../td.html#api-tensordict-methods).

## Tutorials that use `get`

![](../../_images/sphx_glr_tensordict_keys_thumb.png)

[Getting, setting, and nesting keys](../../tutorials/tensordict_keys.html)

Getting, setting, and nesting keys
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict