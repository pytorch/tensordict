# LazyStackedTensorDict

`LazyStackedTensorDict` has the methods of [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase),
listed by task in [Methods by task](../td.html#api-tensordict-methods). This page lists only the
methods that `LazyStackedTensorDict` adds, and the overrides that have their own
documentation.

*class*tensordict.LazyStackedTensorDict(**tensordicts: T*, *stack_dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *hook_out: callable | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *hook_in: callable | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: Sequence[[int](https://docs.python.org/3/builtins/functions.html#int)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: Sequence[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *stack_dim_name: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *strict_shape: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Bases: [`TensorDictBase`](tensordict.TensorDictBase.html#tensordict.TensorDictBase)

A stack of tensordicts that keeps each input as a separate tensordict.

[`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) and [`tensordict.stack()`](tensordict.stack.html#tensordict.stack) return a dense
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict). They raise an error when the inputs have
different keys, entry shapes or batch sizes. A lazy stack accepts these
inputs: it keeps a list of the input tensordicts and stacks their entries
only when you read them. Use it for data whose size changes from one
sample to the next, such as text or episodes of different lengths.
[Stacking and variable-size data](../../concepts/stacking.html#stacking) explains when to use a dense or a lazy stack, and
[Which container should I use?](../../choosing_a_container.html#choosing-a-container) compares the containers.

Build a lazy stack with [`tensordict.lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) or
`LazyStackedTensorDict(*tds, stack_dim=dim)`. To get a dense
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) when the inputs allow it and a lazy stack
otherwise, use [`tensordict.maybe_dense_stack()`](tensordict.maybe_dense_stack.html#tensordict.maybe_dense_stack).

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> tds = [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.ones(4))]
>>> td_stack = LazyStackedTensorDict(*tds, stack_dim=0)
>>> td_stack.batch_size
torch.Size([2])
>>> td_stack[0] is tds[0]
True
>>> td_stack.get("a", as_padded_tensor=True)
tensor([[0., 0., 0., 0.],
 [1., 1., 1., 1.]])
```

A lazy stack behaves as follows:

- Indexing along the stack dimension returns the input objects:
`td_stack[i] is tds[i]`.
- Reading a key (`td_stack["a"]`) stacks the entries into a new tensor.
In-place changes to that tensor do not reach the inputs. Writes through
the lazy stack (`td_stack["a"] = value`, [`apply_()`](tensordict.TensorDictBase.apply_.html#tensordict.TensorDictBase.apply_))
do reach them.
- If the entry shapes differ, `td_stack["a"]` raises an error. Pass
`as_nested_tensor`, `as_padded_tensor` or `as_list` to [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get)
to read the entries as a nested tensor, a padded tensor or a list.
These arguments apply even when the shapes match.
- [`keys()`](tensordict.TensorDictBase.keys.html#tensordict.TensorDictBase.keys) lists only the keys that every
input has. The printed lazy stack shows the other keys under
`exclusive_fields`.
- The batch sizes of the inputs must have the same number of dimensions.
If a dimension differs between inputs, it is `-1` in the stacked
`batch_size`. Use `strict_shape=True` to raise an error instead.
An entry shape has `-1` in the same way for each dimension that
differs between inputs.

Parameters:

***tensordicts** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordicts to stack.

Keyword Arguments:

- **stack_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension of the batch size along which the
tensordicts are stacked. It must be between `-td.ndim - 1` and
`td.ndim`, where `td` is one of the inputs. Defaults to `0`.
- **hook_out** (*Callable**,**optional*) - 

a callable to execute after [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get).

Deprecated since version 0.15: `hook_out` belongs to the [`vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) support of lazy
stacks and has no replacement.
- **hook_in** (*Callable**,**optional*) - 

a callable to execute before [`set()`](tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set).

Deprecated since version 0.15: `hook_in` belongs to the [`vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) support of lazy
stacks and has no replacement.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - if no tensordict is
given, the batch size of the stacked tensordicts: the empty stack
has a `batch_size` with `0` at `stack_dim`. If tensordicts
are given, it must equal the stacked batch size, or the
constructor raises a `RuntimeError`.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device of an empty stack. If
tensordicts are given, the stack takes the device of the first
one and ignores this argument. Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the dimension names of an empty stack.
If tensordicts are given, the names come from them, and this
argument is ignored with a `FutureWarning`. To name the dimensions
of such a stack, pass `stack_dim_name`, or set `names` on the
stack. Starting with TensorDict 0.17, the default will change to
applying these names to the stack.
- **stack_dim_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the name of the stack dimension.
Defaults to `None`.
- **strict_shape** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the batch sizes of the
inputs must be equal, or the constructor raises a
`RuntimeError`. Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> tds = [TensorDict(a=torch.zeros(3, 4), batch_size=[3]) for _ in range(10)]
>>> td_stack = LazyStackedTensorDict.lazy_stack(tds, -1)
>>> td_stack.shape
torch.Size([3, 10])
>>> td_stack.get("a").shape
torch.Size([3, 10, 4])
>>> td_stack[:, 0] is tds[0]
True
```

Inputs with different batch sizes give `-1` in the batch size:

```
>>> td_stack = LazyStackedTensorDict(
... TensorDict(a=torch.zeros(3), batch_size=[3]),
... TensorDict(a=torch.zeros(4), batch_size=[4]),
... )
>>> td_stack.batch_size
torch.Size([2, -1])
```

Inputs with different keys keep their own keys:

```
>>> td_stack = LazyStackedTensorDict(
... TensorDict(a=torch.zeros(3)), TensorDict(a=torch.zeros(3), b=torch.ones(2))
... )
>>> list(td_stack.keys())
['a']
>>> td_stack[1]["b"]
tensor([1., 1.])
```

Note

You can assign a list to a key of a lazy stack. The list must
have the structure that `tensor.tolist()` returns: the length of
its first level must equal the first dimension of the lazy stack,
whether or not it is the stack dimension.

```
>>> td = LazyStackedTensorDict(TensorDict(), TensorDict(), stack_dim=0)
>>> td["a"] = [torch.ones(2), torch.zeros(1)]
>>> assert td[1]["a"] == torch.zeros(1)
>>> td["b"] = ["a string", "another string"]
>>> assert td[1]["b"] == "another string"
```

append(*tensordict: T*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Appends a tensordict to the lazy stack, like [`list.append()`](https://docs.python.org/3/builtins/stdtypes.html#list.append).

Same as `insert()` at the end of the stack.

Parameters:

**tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to append. It must have
the device of the other members and as many batch dimensions.
Its sizes must match theirs, except where the batch size of
the lazy stack is `-1`.

extend(*tensordict: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[T] | T*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Appends several tensordicts to the lazy stack, like [`list.extend()`](https://docs.python.org/3/builtins/stdtypes.html#list.extend).

Parameters:

**tensordict** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the
tensordicts to append. A tensordict is unbound along the stack
dimension and its parts are appended.

*classmethod*from_dict(*input_dict: [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[str](https://docs.python.org/3/builtins/stdtypes.html#str), [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [List](https://docs.python.org/3/library/typing.html#typing.List)[[Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Any](https://docs.python.org/3/library/typing.html#typing.Any)]]*, **other*, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *batch_size=None*, *device=None*, *batch_dims=None*, *stack_dim_name=None*, *stack_dim=0*, *names: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*)

Builds a lazy stack from a list of dicts, one tensordict per dict.

Each dict goes through [`tensordict.TensorDict.from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict), then the
results are stacked lazily, so the dicts can have different keys and
entry shapes. A dict whose keys are `"0"`, `"1"`, ... is read as
that list.

Parameters:

- **input_dict** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*, or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*of*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the data of each
stacked tensordict.
- ***other** - passed to [`tensordict.TensorDict.from_dict()`](tensordict.TensorDictBase.from_dict.html#tensordict.TensorDictBase.from_dict).

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
of each tensordict to the largest leading shape that all its
entries share. Defaults to `False`.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
each stacked tensordict, without the stack dimension. Defaults
to `None`, which gives each tensordict the batch size `[]`
unless its batch size is set automatically.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of each
stacked tensordict. Defaults to `None`, which keeps each leaf
where it is.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions of each tensordict. Defaults to
`None`.
- **stack_dim_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the name of the stack dimension.
Defaults to `None`.
- **stack_dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the stack dimension. Defaults to `0`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the dimensions of the
result, stack dimension included. Exclusive with
`stack_dim_name`. Without a `batch_size`, only
`names[stack_dim]` is used today and the members get no
names. Starting with TensorDict 0.17, the default will change
to giving the others to the members once their batch size is
known (a [`FutureWarning`](https://docs.python.org/3/builtins/exceptions.html#FutureWarning) is emitted when this changes
the result). Passing other entries whose number differs from
the member batch dimensions is deprecated and will be removed
in TensorDict 0.17 (a [`DeprecationWarning`](https://docs.python.org/3/builtins/exceptions.html#DeprecationWarning) is emitted);
use `stack_dim_name` instead.

Returns:

A `LazyStackedTensorDict`.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict
>>> td = LazyStackedTensorDict.from_dict(
... [{"a": torch.zeros(3)}, {"a": torch.zeros(4)}], stack_dim_name="t"
... )
>>> td.batch_size, td.names
(torch.Size([2]), ['t'])
>>> td[1]["a"].shape
torch.Size([4])
```

*classmethod*from_list(*input: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[Mapping](https://docs.python.org/3/library/typing.html#typing.Mapping)]*, ***, *auto_batch_size: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *batch_dims: [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *names: [list](https://docs.python.org/3/builtins/stdtypes.html#list)[[str](https://docs.python.org/3/builtins/stdtypes.html#str)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *lazy: [bool](https://docs.python.org/3/builtins/functions.html#bool) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → Self

Builds a lazy stack from a list of dicts, one tensordict per dict.

Same as [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict) with a list, stacked along dimension `0`.
The dicts can have different keys and entry shapes.

Parameters:

**input** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)) - the data of each stacked tensordict.

Keyword Arguments:

- **auto_batch_size** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, sets the batch size
of each tensordict to the largest leading shape that all its
entries share. Defaults to `None`, which turns it on when
`batch_dims` is given.
- **batch_size** ([*Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the batch size of
each stacked tensordict, without the stack dimension. Defaults
to `None`, which gives each tensordict the batch size `[]`
unless its batch size is set automatically.
- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*or*[*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the device of each
stacked tensordict. Defaults to `None`, which keeps each leaf
where it is.
- **batch_dims** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the maximum number of batch dimensions
of each tensordict when the batch size is set automatically.
Defaults to `None`.
- **names** ([*list*](https://docs.python.org/3/builtins/stdtypes.html#list)*of*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the names of the dimensions of the
result, stack dimension first. See [`from_dict()`](tensordict.from_dict.html#tensordict.from_dict). Defaults
to `None`.
- **lazy** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - ignored: the result is always a lazy stack.
Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict
>>> td = LazyStackedTensorDict.from_list(
... [{"a": torch.zeros(3)}, {"a": torch.zeros(4)}], auto_batch_size=True
... )
>>> td.batch_size
torch.Size([2, -1])
```

get_item_shape(*key*)

Returns the shape of an entry of the lazy stack, with `-1` where the members differ.

The shape of a tensor entry is read from the members without stacking
them. Other entries are read with [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get), which stacks them.

Parameters:

**key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the key of the entry.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> td = LazyStackedTensorDict(
... TensorDict(a=torch.zeros(3)), TensorDict(a=torch.zeros(4))
... )
>>> td.get_item_shape("a")
torch.Size([2, -1])
```

get_nestedtensor(*key: [NestedKey](tensordict.NestedKey.html#tensordict.NestedKey)*, *default: [Any](https://docs.python.org/3/library/typing.html#typing.Any) = NO_DEFAULT*, ***, *layout: [layout](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | [TensorCollection](tensordict.TensorCollection.html#tensordict.TensorCollection)

Returns an entry of the lazy stack as a nested tensor.

Deprecated since version 0.15: Use `get(key, as_nested_tensor=True)` instead. It differs in three
ways: it builds a `torch.jagged` nested tensor unless you pass
`layout=torch.strided`, whereas this method builds a
`torch.strided` one by default; when a tensor `default` is
given and some tensordicts lack `key`, it returns `default`
itself, whereas this method uses `default` as the value of the
tensordicts that lack `key`; and when the stack dim is not 0, it
nests the tensors along the stack dim and puts that dim first,
whereas this method raises.

Parameters:

- **key** ([*NestedKey*](tensordict.NestedKey.html#tensordict.NestedKey)) - the entry to nest.
- **default** (*Any**,**optional*) - the value for the tensordicts that lack
`key`. If it is a tensor, it is nested with the entries of
the other tensordicts. Otherwise, it is returned as is.
Without a default, a missing key raises a `KeyError`.

Keyword Arguments:

**layout** ([*layout*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout)*,**optional*) - the layout of the nested tensor.
Defaults to `None`, which builds a `torch.strided` one.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> td0 = TensorDict({"a": torch.zeros(4), "b": torch.zeros(4)}, [])
>>> td1 = TensorDict({"a": torch.ones(5)}, [])
>>> td = LazyStackedTensorDict.lazy_stack([td0, td1], 0)
>>> a = td.get_nestedtensor("a")
>>> # using a tensor as default uses this default to build the nested tensor
>>> b = td.get_nestedtensor("b", default=torch.ones(4))
>>> assert (b[1] == 1).all()
>>> # using anything else as default returns the default
>>> b2 = td.get_nestedtensor("b", None)
>>> assert b2 is None
```

*property*hook_in*: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The callable that the lazy stack applies to the values it writes.

Deprecated since version 0.15: `hook_in` belongs to the [`vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) support of lazy
stacks and has no replacement.

*property*hook_out*: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable) | [None](https://docs.python.org/3/builtins/constants.html#None)*

The callable that the lazy stack applies to the values it returns.

Deprecated since version 0.15: `hook_out` belongs to the [`vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) support of lazy
stacks and has no replacement.

insert(*index: [int](https://docs.python.org/3/builtins/functions.html#int)*, *tensordict: T*) → [None](https://docs.python.org/3/builtins/constants.html#None)

Inserts a tensordict into the lazy stack at `index`, like [`list.insert()`](https://docs.python.org/3/builtins/stdtypes.html#list.insert).

The lazy stack is changed in place and its batch size grows by one
along the stack dimension. The new tensordict is stored as is, not
copied.

Parameters:

- **index** ([*int*](https://docs.python.org/3/builtins/functions.html#int)) - the position of the new tensordict along the stack
dimension.
- **tensordict** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordict to insert. It must have
the device of the other members and as many batch dimensions.
Its sizes must match theirs, except where the batch size of
the lazy stack is `-1`.

Raises:

- [**TypeError**](https://docs.python.org/3/builtins/exceptions.html#TypeError) - if `tensordict` is not a tensordict.
- [**ValueError**](https://docs.python.org/3/builtins/exceptions.html#ValueError) - if the device of `tensordict` differs from that of
 the members, or if its batch size does not fit the lazy stack.
- [**RuntimeError**](https://docs.python.org/3/builtins/exceptions.html#RuntimeError) - if the lazy stack is locked.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> td = LazyStackedTensorDict(TensorDict(a=torch.zeros(3)))
>>> td.insert(0, TensorDict(a=torch.ones(3)))
>>> td.batch_size
torch.Size([2])
>>> td[0]["a"]
tensor([1., 1., 1.])
```

*classmethod*lazy_stack(*items: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, ***, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [int](https://docs.python.org/3/builtins/functions.html#int) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *out: T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *stack_dim_name: [str](https://docs.python.org/3/builtins/stdtypes.html#str) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *strict_shape: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *as_list: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *as_padded_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *as_nested_tensor: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *padding_side: [str](https://docs.python.org/3/builtins/stdtypes.html#str) = 'right'*, *layout: [layout](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *padding_value: [float](https://docs.python.org/3/builtins/functions.html#float) | [int](https://docs.python.org/3/builtins/functions.html#int) | [bool](https://docs.python.org/3/builtins/functions.html#bool) = 0.0*) → Self

Stacks tensordicts into a lazy stack.

The result keeps each input as a separate tensordict, so the inputs
can have different keys, entry shapes and batch sizes. See
`LazyStackedTensorDict` for the behavior of the
result, and [Stacking and variable-size data](../../concepts/stacking.html#stacking) for the choice between a dense and a lazy
stack.

If `items` holds tensorclass instances of one class, the result is
an instance of that class that holds a lazy stack. If `items` holds
tensors, they are stacked with [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack), unless one of
`as_list`, `as_padded_tensor` or `as_nested_tensor` is set.
These arguments, `layout`, `padding_side` and `padding_value`
apply only to a list of tensors. To read padded or nested entries from
a lazy stack of tensordicts, pass them to [`get()`](tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) instead.

Parameters:

- **items** (*Sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordicts to stack. It
must not be empty, or a `RuntimeError` is raised.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the dimension along which to stack. Defaults
to `0`. A `dim` below `-batch_dims - 1` warns and wraps
around once more (and raises if it is still out of range).
Starting with TensorDict 0.17, the default will change to
raising an `IndexError`, as for a tensor.

Keyword Arguments:

- **device** ([*device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - not used: the stack takes the
device of the first item.
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a lazy stack to write the data
into. Its batch size must be the stacked batch size. Defaults
to `None`.
- **stack_dim_name** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the name of the stack dimension.
Defaults to `None`.
- **strict_shape** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the batch sizes of the
items must be equal, or a `RuntimeError` is raised.
Defaults to `False`.
- **as_list** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - tensors only. If `True`, the tensors
are returned as a list.
Exclusive with `as_padded_tensor` and `as_nested_tensor`.
Defaults to `False`.
- **as_padded_tensor** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - tensors only. If `True`, the
tensors are padded to the largest shape and stacked into one tensor: each
dim is padded to its largest size. `padding_value` sets the
padding value and `padding_side` sets the side of every
padded dim.
Exclusive with `as_list` and `as_nested_tensor`.
Defaults to `False`.
- **as_nested_tensor** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - tensors only. If `True`, the
tensors are returned as a nested tensor ([`torch.nested`](https://docs.pytorch.org/docs/stable/nested.html#module-torch.nested)). `layout`
sets its layout. Tensors that are nested tensors themselves
raise a `ValueError`, as a nested tensor has a single nested
level.
Exclusive with `as_list` and `as_padded_tensor`.
Defaults to `False`.
- **layout** ([*layout*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.layout)*,**optional*) - the layout of the nested tensor
when `as_nested_tensor=True`: `torch.jagged` or
`torch.strided`. Defaults to `torch.jagged`.
- **padding_side** ([*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,**optional*) - the side to pad with
`as_padded_tensor`: `"left"` or `"right"`. Defaults to
`"right"`.
- **padding_value** ([*float*](https://docs.python.org/3/builtins/functions.html#float)*,*[*int*](https://docs.python.org/3/builtins/functions.html#int)*or*[*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - the padding value
with `as_padded_tensor`. Defaults to `0.0`.

Returns:

A `LazyStackedTensorDict`, or a tensorclass
instance if `items` holds tensorclass instances.

Examples

```
>>> import torch
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> td_stack = LazyStackedTensorDict.lazy_stack(
... [TensorDict(a=torch.zeros(3)), TensorDict(a=torch.zeros(4))]
... )
>>> type(td_stack).__name__
'LazyStackedTensorDict'
>>> td_stack.get("a", as_list=True)
[tensor([0., 0., 0.]), tensor([0., 0., 0., 0.])]
```

*classmethod*maybe_dense_stack(*items: [Sequence](https://docs.python.org/3/library/typing.html#typing.Sequence)[[TensorDictBase](tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *dim: [int](https://docs.python.org/3/builtins/functions.html#int) = 0*, *out: T | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *strict: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → Self

Stacks tensordicts densely if possible, or into a lazy stack otherwise.

Use this function when the inputs usually match, but can differ. If
every item has the same keys and the same entry shapes, the result is
the dense stack that [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) returns. Otherwise, the result
is the lazy stack that [`lazy_stack()`](tensordict.lazy_stack.html#tensordict.lazy_stack) returns. See [Stacking and variable-size data](../../concepts/stacking.html#stacking).

Parameters:

- **items** (*Sequence**of*[*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*or**tensorclass instances*) - the
tensordicts to stack.
- **dim** ([*int*](https://docs.python.org/3/builtins/functions.html#int)*,**optional*) - the new dimension. Defaults to `0`. A
`dim` outside `[-batch_dims - 1, batch_dims]` warns, and
is used on the dims of the leaves. Starting with TensorDict
0.17, the default will change to raising an `IndexError`,
as for a tensor.
- **out** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)*,**optional*) - a tensordict to write the result
into. Defaults to `None`.
- **strict** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - used only with `out`. If `True`,
`out` must have the same keys as the items, or a
`RuntimeError` is raised. Defaults to `False`.

Returns:

A [`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict) (or the tensorclass of the
items) if the items can be stacked densely, a
`LazyStackedTensorDict` otherwise.

Examples

```
>>> from tensordict import LazyStackedTensorDict, TensorDict
>>> td0 = TensorDict({"a": 0}, [])
>>> td1 = TensorDict({"b": 0}, [])
>>> out = LazyStackedTensorDict.maybe_dense_stack([td0, td0]) # same keys: dense stack
>>> assert type(out) is TensorDict and out.shape == (2,)
>>> out = LazyStackedTensorDict.maybe_dense_stack([td0, td1]) # different keys: lazy stack
>>> assert isinstance(out, LazyStackedTensorDict) and out.shape == (2,)
```

non_tensor_items(*include_nested: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*)

Returns the non-tensor leaves as `(key, value)` pairs, each stacked across the members.

Only the keys that every member holds are listed. If
`include_nested` is `True`, the nested tensordicts are searched as
well.

valid_keys(*include_nested: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *leaves_only: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *is_leaf: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Type](https://docs.python.org/3/library/typing.html#typing.Type)], [bool](https://docs.python.org/3/builtins/functions.html#bool)] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, ***, *sort: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → _LazyStackedTensorDictKeysView

Returns the keys of the lazy stack, like [`keys()`](tensordict.TensorDictBase.keys.html#tensordict.TensorDictBase.keys).

Deprecated since version 0.15: Use [`keys()`](tensordict.TensorDictBase.keys.html#tensordict.TensorDictBase.keys) instead, which takes the same arguments. Will be
removed in TensorDict 0.17.

## Tutorials that use `LazyStackedTensorDict`

![](../../_images/sphx_glr_streamed_tensordict_thumb.png)

[Building tensordicts from streams](../../tutorials/streamed_tensordict.html)

Building tensordicts from streams
![](../../_images/sphx_glr_tensordict_shapes_thumb.png)

[Changing the shape of a TensorDict](../../tutorials/tensordict_shapes.html)

Changing the shape of a TensorDict