Note

Go to the end
to download the full example code.

# Getting, setting, and nesting keys

**Author**: [Tom Begley](https://github.com/tcbegley)

A [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) stores tensors under string keys, as a
Python `dict` does, and a value can be another `TensorDict`. If you know
`dict`, most of this tutorial is familiar. It shows where a `TensorDict`
is different: what a missing key returns, what a `for` loop gives, and how
tuples of strings address nested entries.

What you will learn:

- How to set and get entries, and what happens when a key is missing.
- How to find out if a key exists, and how to iterate over the keys.
- How to read and write nested entries with tuples of strings.
- How to flatten nested keys, and how to select and exclude keys.

## Setting and getting keys

Set and get entries with the same syntax as for a Python `dict`:

```
import torch
from tensordict import TensorDict

tensordict = TensorDict()

# set a key
a = torch.rand(10)
tensordict["a"] = a

# retrieve the value stored under "a"
assert tensordict["a"] is a
```

Note

Unlike the keys of a Python `dict`, all the keys of a `TensorDict` must
be strings. Tuples of strings refer to nested values, as shown below.

The `.get()` and `.set()` methods do the same thing:

```
tensordict = TensorDict()

# set a key
a = torch.rand(10)
tensordict.set("a", a)

# retrieve the value stored under "a"
assert tensordict.get("a") is a
```

Like `dict.get`, `get` returns `None` if the key is not in the
`TensorDict`. Indexing with `tensordict[key]` raises a `KeyError`
instead. Use `tensordict[key]` when the key must exist, and `get` when it
can be missing:

```
assert tensordict.get("banana") is None
try:
 tensordict["banana"]
except KeyError:
 print("tensordict['banana'] raises a KeyError")
```

```
tensordict['banana'] raises a KeyError
```

If you give `get` a default value, it returns this value for a missing
key:

```
assert tensordict.get("banana", a) is a
```

Like `dict.setdefault`,
[`setdefault()`](../reference/generated/tensordict.TensorDictBase.setdefault.html#tensordict.TensorDictBase.setdefault) returns the value of a key. If
the key is not found, it stores the default value under the key and returns
it.

```
assert tensordict.setdefault("banana", a) is a
# a is now stored under "banana"
assert tensordict["banana"] is a
```

To delete a key, use the `del` statement, as for a Python `dict`. The
[`TensorDict.del_`](../reference/generated/tensordict.TensorDictBase.del_.html#tensordict.TensorDictBase.del_) method does the same.

```
del tensordict["banana"]
```

To write new values into the existing tensor of a key, pass
`inplace=True` to `.set()`, or use the `.set_()` method. The tensor
object stays the same, and only its values change.

```
tensordict.set("a", torch.zeros(10), inplace=True)

# all the entries of the "a" tensor are now zero
assert (tensordict.get("a") == 0).all()
# but it is still the same tensor as before
assert tensordict.get("a") is a

# set_ does the same
tensordict.set_("a", torch.ones(10))
assert (tensordict.get("a") == 1).all()
assert tensordict.get("a") is a
```

## Renaming keys

To rename a key, use the
[`TensorDict.rename_key_`](../reference/generated/tensordict.TensorDictBase.rename_key_.html#tensordict.TensorDictBase.rename_key_) method. The
value stays in the [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), under the new key.

```
tensordict.rename_key_("a", "b")
assert tensordict.get("b") is a
print(tensordict)
```

```
TensorDict(batch_size=[], fields={b: f32[10]})
```

## Updating multiple values

[`TensorDict.update`](../reference/generated/tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) copies the entries of
another [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), or of a `dict`, into a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict). It overwrites the keys that exist, and
creates the keys that do not exist.

The second argument of the constructor below, `[10]`, is the batch size:
the leading dimensions that all the entries share (see [Batch size, shapes, and indexing](../concepts/batch_size.html#batch-size)).

```
tensordict = TensorDict({"a": torch.rand(10), "b": torch.rand(10)}, [10])
tensordict.update(TensorDict({"a": torch.zeros(10), "c": torch.zeros(10)}, [10]))
assert (tensordict["a"] == 0).all()
assert (tensordict["b"] != 0).all()
assert (tensordict["c"] == 0).all()
print(tensordict)
```

```
TensorDict(batch_size=[10], fields={a: f32[], b: f32[], c: f32[]})
```

## Nested values

A value of a `TensorDict` can be another `TensorDict`. To create nested
values, pass a `TensorDict` or a nested `dict` to the constructor. A
nested tensordict has the batch dimensions of its parent, and can have more.

```
# creating nested values with a nested dict
nested_tensordict = TensorDict(
 {"a": torch.rand(2, 3), "double_nested": {"a": torch.rand(2, 3)}}, [2, 3]
)
# creating nested values with a TensorDict
tensordict = TensorDict({"a": torch.rand(2), "nested": nested_tensordict}, [2])

print(tensordict)
```

```
TensorDict(batch_size=[2], fields={
 a: f32[],
 nested: [3]{a: f32[], double_nested: {a: f32[]}}})
```

To read a nested value, use a tuple of strings:

```
double_nested_a = tensordict["nested", "double_nested", "a"]
nested_a = tensordict.get(("nested", "a"))
```

To write a nested value, also use a tuple of strings:

```
tensordict["nested", "double_nested", "b"] = torch.rand(2, 3)
tensordict.set(("nested", "b"), torch.rand(2, 3))

print(tensordict)
```

```
TensorDict(batch_size=[2], fields={
 a: f32[],
 nested: [3]{a: f32[], b: f32[], double_nested: {a: f32[], b: f32[]}}})
```

## Iterating over the contents of a TensorDict

To iterate over the keys of a `TensorDict`, use the `.keys()` method.

Note

Do not write `for key in tensordict`. A `TensorDict` iterates like a
tensor: this loop gives one sub-tensordict for each element of the first
batch dimension. If the tensordict has no batch dimensions, the loop
raises a `TypeError`. In the same way, `len(tensordict)` is the size of
the first batch dimension, not the number of keys.

```
for key in tensordict.keys():
 print(key)
```

```
a
nested
```

By default, `.keys()` gives only the top-level keys of the `TensorDict`.
With `include_nested=True`, it also gives the keys of the nested
tensordicts, at all levels, as tuples of strings.

```
for key in tensordict.keys(include_nested=True):
 print(key)
```

```
a
('nested', 'a')
('nested', 'double_nested', 'a')
('nested', 'double_nested', 'b')
('nested', 'double_nested')
('nested', 'b')
nested
```

To get only the keys of tensor values, and not the keys of nested
tensordicts, also pass `leaves_only=True`.

```
for key in tensordict.keys(include_nested=True, leaves_only=True):
 print(key)
```

```
a
('nested', 'a')
('nested', 'double_nested', 'a')
('nested', 'double_nested', 'b')
('nested', 'b')
```

As for `dict`, the `.values()` and `.items()` methods exist. They take the
same keyword arguments as `.keys()`.

```
for key, value in tensordict.items(include_nested=True):
 if isinstance(value, TensorDict):
 print(f"{key} is a TensorDict")
 else:
 print(f"{key} is a Tensor")
```

```
a is a Tensor
nested is a TensorDict
('nested', 'a') is a Tensor
('nested', 'double_nested') is a TensorDict
('nested', 'double_nested', 'a') is a Tensor
('nested', 'double_nested', 'b') is a Tensor
('nested', 'b') is a Tensor
```

## Testing if a key exists

To find out if a key exists in a `TensorDict`, use the `in` operator. It
also accepts nested keys as tuples of strings. Each `in` test is a `dict`
lookup at each level of nesting, so its cost does not grow with the number
of keys.

```
assert "a" in tensordict
assert ("nested", "a") in tensordict
assert ("nested", "banana") not in tensordict
```

## Flattening and unflattening nested keys

The `.flatten_keys()` method turns nested keys into top-level keys. It
joins the parts of each nested key with a separator:

```
print(tensordict, end="\n\n")
print(tensordict.flatten_keys(separator="."))
```

```
TensorDict(batch_size=[2], fields={
 a: f32[],
 nested: [3]{a: f32[], b: f32[], double_nested: {a: f32[], b: f32[]}}})

TensorDict(batch_size=[2], fields={
 a: f32[],
 nested.a: f32[3],
 nested.b: f32[3],
 nested.double_nested.a: f32[3],
 nested.double_nested.b: f32[3]})
```

The `.unflatten_keys()` method does the inverse:

```
flattened_tensordict = tensordict.flatten_keys(separator=".")
print(flattened_tensordict, end="\n\n")
print(flattened_tensordict.unflatten_keys(separator="."))
```

```
TensorDict(batch_size=[2], fields={
 a: f32[],
 nested.a: f32[3],
 nested.b: f32[3],
 nested.double_nested.a: f32[3],
 nested.double_nested.b: f32[3]})

TensorDict(batch_size=[2], fields={
 a: f32[],
 nested: {a: f32[3], b: f32[3], double_nested: {a: f32[3], b: f32[3]}}})
```

`unflatten_keys` works on any flat key names with a separator. For
example, the parameter names of a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) use `"."`. This
code builds a [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with the structure of the
module:

```
import torch.nn as nn

module = nn.Sequential(
 nn.Sequential(nn.Linear(100, 50), nn.Linear(50, 10)),
 nn.Linear(10, 1),
)
params = TensorDict(dict(module.named_parameters()), []).unflatten_keys(".")

print(params)
```

```
TensorDict(batch_size=[], fields={
 0: {
 0: {bias: f32[50], weight: f32[50,100]},
 1: {bias: f32[10], weight: f32[10,50]}},
 1: {bias: f32[1], weight: f32[1,10]}})
```

For modules, [`TensorDict.from_module`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.from_module)
gives the same result in one call, and also includes the buffers of the
module. The [Functional calls and ensembles with TensorDict](functional.html#sphx-glr-tutorials-functional-py) tutorial uses it.

```
params_from_module = TensorDict.from_module(module)
assert set(params_from_module.keys(True, True)) == set(params.keys(True, True))
```

## Selecting and excluding keys

Two methods return a new [`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with a subset of
the keys:

- [`TensorDict.select`](../reference/generated/tensordict.TensorDictBase.select.html#tensordict.TensorDictBase.select) keeps only the
keys that you name.
- [`TensorDict.exclude`](../reference/generated/tensordict.TensorDictBase.exclude.html#tensordict.TensorDictBase.exclude) keeps all the
keys except the keys that you name.

```
print("Select:")
print(tensordict.select("a", ("nested", "a")), end="\n\n")
print("Exclude:")
print(tensordict.exclude(("nested", "b"), ("nested", "double_nested")))
```

```
Select:
TensorDict(batch_size=[2], fields={a: f32[], nested: [3]{a: f32[]}})

Exclude:
TensorDict(batch_size=[2], fields={a: f32[], nested: [3]{a: f32[]}})
```

## Conclusion

You can now set, read, rename, and delete the entries of a
[`TensorDict`](../reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) at any depth, with tuples of strings for the
nested keys. You can test keys with `in`, list them with `keys()`,
`values()`, and `items()`, flatten them, and keep or drop a subset with
`select` and `exclude`.

Two behaviors differ from a `dict`. `get` returns `None` for a missing
key, and `tensordict[key]` raises a `KeyError`. A `for` loop over a
`TensorDict` goes over its first batch dimension, so use `keys()` to go
over the keys.

## Further reading

- [Keys, nested keys, and locking](../concepts/keys.html#keys) explains keys, nested keys, and locking.
- [Non-tensor data and unbatched entries](../concepts/non_tensor.html#non-tensor-data) explains how to store values that are not tensors.
- [Changing the shape of a TensorDict](tensordict_shapes.html#sphx-glr-tutorials-tensordict-shapes-py) shows the shape operations.
- [Keys, values, and entries](../reference/td.html#api-methods-keys) lists all the key methods.

**Total running time of the script:** (0 minutes 0.010 seconds)

[`Download Jupyter notebook: tensordict_keys.ipynb`](../_downloads/97d8d73d25e56588177c9d8bc7211dc9/tensordict_keys.ipynb)

[`Download Python source code: tensordict_keys.py`](../_downloads/3d2b145d2e55d1ee5cc84110745e45ea/tensordict_keys.py)

[`Download zipped: tensordict_keys.zip`](../_downloads/30cf47ff46cc4701c97ed46f4d91a0f2/tensordict_keys.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)