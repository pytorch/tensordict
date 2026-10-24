# tensorclass

tensordict.tensorclass(*cls: T = None*, */*, ***, *autocast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *frozen: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *nocast: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *shadow: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *tensor_only: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *skip_post_init_on_rebuild: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*, *cast_filter: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable) | [None](https://docs.python.org/3/builtins/constants.html#None) = None*, *kw_only: [bool](https://docs.python.org/3/builtins/functions.html#bool) = False*) → T | [None](https://docs.python.org/3/builtins/constants.html#None)

A decorator that turns a class with annotated fields into a tensorclass.

A tensorclass is a dataclass whose instances behave like a
[`TensorDict`](tensordict.TensorDict.html#tensordict.TensorDict): they have a batch size, and they support
indexing, reshaping, `to(device)`, stacking and memory-mapping. For new
code, subclass [`TensorClass`](tensordict.TensorClass.html#tensordict.TensorClass) instead. It takes the same
flags, and type checkers understand it without a plugin. This decorator
is kept for existing code and for code that moves from
`@dataclass`. See [TensorClass: a dataclass of tensors](../../tensorclass.html#tensorclass-guide).

Parameters:

**cls** ([*type*](https://docs.python.org/3/builtins/functions.html#type)*,**optional*) - the class to decorate, passed by
`@tensorclass` without parentheses.

Keyword Arguments:

- **autocast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, when a field is set, the value
is converted to the annotated type. Exclusive with `nocast` and
`tensor_only`. Defaults to `False`.
- **frozen** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, assigning a field raises
`dataclasses.FrozenInstanceError`, as with
`@dataclass(frozen=True)`. To block writes on one instance only,
call `lock_()` on it or pass `lock=True` to the constructor:
writes then raise a `RuntimeError`. Defaults to `False`.
- **nocast** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, Tensor-compatible types such as
`int`, `np.ndarray` and the like are not cast to a tensor
type. Exclusive with `autocast` and `tensor_only`. Defaults
to `False`.
- **shadow** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, fields can have the name of a
tensorclass member (`batch_size`, `sum`, ...), and hide that
member. If `False`, such a name raises an `AttributeError`
when the class is defined. Defaults to `False`.
- **tensor_only** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, every field annotation must
be a tensor type or a tensor collection. Values that can be cast
to a tensor are cast. Reading a field then skips the lookup in the
non-tensor storage. Exclusive with `autocast` and `nocast`.
Defaults to `False`.
- **skip_post_init_on_rebuild** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`,
`__post_init__` runs in the generated `__init__`,
`from_tensordict`, `from_dict`, the other constructors such
as `from_any` or `fromkeys`, and when the data is loaded or
received from another process, but not when tensordict makes a
new instance from the data of an existing one, as in indexing,
`clone`, `to`, `apply` or `torch.stack`. Defaults to
`False`: `__post_init__` also runs on these rebuilds, except
on the shells that `empty()` builds. On a lazy stack it runs on
each stacked instance.
- **cast_filter** (*Callable**,**optional*) - the default `cast_filter` of
[`to()`](tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) and of the dtype shorthands
(`float()`, `bfloat16()`, ...) on the instances of the class: a
predicate `fn(key, value) -> bool` that selects the tensor leaves
a dtype cast applies to, `key` being the nested key as a tuple of
str. A `cast_filter` passed to the call overrides it, and
`cast_filter=None` there casts every leaf. Subclasses inherit it.
It applies to calls on the tensorclass itself, not when the
tensorclass is a value of another tensordict or tensorclass that
is cast. Defaults to `None`: every leaf is cast.
- **kw_only** ([*bool*](https://docs.python.org/3/builtins/functions.html#bool)*,**optional*) - if `True`, the fields of the class are
keyword-only arguments of `__init__`, as with
`@dataclass(kw_only=True)`. The fields of a subclass are not,
unless it sets the option too. Defaults to `False`.

Returns:

The tensorclass, or a decorator if the flags are given without a
class (`@tensorclass(autocast=True)`).

Examples

tensorclass can be used with or without arguments:

```
>>> from typing import Any
>>> import torch
>>> from tensordict import tensorclass
>>> @tensorclass
... class X:
... y: int
>>> X(torch.ones(())).y
tensor(1.)
>>> @tensorclass(autocast=False)
... class X:
... y: int
>>> X(torch.ones(())).y
tensor(1.)
>>> @tensorclass(autocast=True)
... class X:
... y: int
>>> X(torch.ones(())).y
1
>>> @tensorclass(nocast=True)
... class X:
... y: Any
>>> X(1).y
1
>>> @tensorclass(nocast=False)
... class X:
... y: Any
>>> X(1).y
tensor(1)
```

A tensorclass can have methods. Fields that are not tensors are stored
as [`NonTensorData`](tensordict.NonTensorData.html#tensordict.NonTensorData):

```
>>> @tensorclass
... class MyData:
... X: torch.Tensor
... y: torch.Tensor
... z: str
... def expand_and_mask(self):
... X = self.X.unsqueeze(-1).expand_as(self.y)
... X = X[self.y]
... return X
...
>>> data = MyData(
... X=torch.ones(3, 4, 1),
... y=torch.zeros(3, 4, 2, 2, dtype=torch.bool),
... z="test",
... batch_size=[3, 4])
>>> print(data)
MyData(batch_size=[3,4], fields={X: f32[1], y: bool[2,2], z: str})
>>> print(data.expand_and_mask())
tensor([])
```

A tensorclass instance can be a field of another tensorclass:

```
>>> @tensorclass
... class NestingMyData:
... nested: MyData
...
>>> nesting_data = NestingMyData(nested=data, batch_size=[3, 4])
>>> # although the data is stored as a TensorDict, the type hint helps us
>>> # to appropriately cast the data to the right type
>>> assert isinstance(nesting_data.nested, type(data))
```

`skip_post_init_on_rebuild` checks the data once, when it comes in:

```
>>> @tensorclass(skip_post_init_on_rebuild=True)
... class Ragged:
... offsets: torch.Tensor
... indices: torch.Tensor
... def __post_init__(self):
... if int(self.offsets[-1]) != len(self.indices):
... raise ValueError("the last offset must be len(indices)")
>>> r = Ragged(offsets=torch.tensor([0, 2, 3]), indices=torch.zeros(3))
>>> r.clone().offsets # not checked again
tensor([0, 2, 3])
```

`cast_filter` keeps the integer and boolean fields out of dtype casts:

```
>>> from tensordict import is_inexact
>>> @tensorclass(cast_filter=is_inexact)
... class Mesh:
... points: torch.Tensor
... cells: torch.Tensor
>>> mesh = Mesh(points=torch.zeros(4, 3), cells=torch.zeros(2, 3, dtype=torch.long))
>>> mesh.bfloat16().points.dtype, mesh.bfloat16().cells.dtype
(torch.bfloat16, torch.int64)
>>> mesh.bfloat16(cast_filter=None).cells.dtype # cast every leaf
torch.bfloat16
```