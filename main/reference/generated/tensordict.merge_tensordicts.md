# merge_tensordicts

tensordict.merge_tensordicts(**tensordicts: T*, *callback_exist: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → T

Merges tensordicts together.

Parameters:

***tensordicts** (*sequence**of*[*TensorDict*](tensordict.TensorDict.html#tensordict.TensorDict)*or**equivalent*) - the list of tensordicts to merge together.

Keyword Arguments:

**callback_exist** ([*callable*](https://docs.python.org/3/builtins/functions.html#callable)*or**Dict**[*[*str*](https://docs.python.org/3/builtins/stdtypes.html#str)*,*[*callable*](https://docs.python.org/3/builtins/functions.html#callable)*]**,**optional*) - a callable in case an entry exists in each and every tensordict.
If the entry is present in some but not all tensordicts, or if `callback_exist` is not passed,
update is used and the first non-`None` value in the tensordict sequence will be used.
If a dictionary of callables is passed, it will contain the associated callback function for some of the
nested keys in the tensordicts passed to the function.

Examples

```
>>> import torch
>>> from tensordict import merge_tensordicts, TensorDict
>>> td0 = TensorDict({"a": {"b0": 0}, "c": {"d": {"e": 0}}, "common": 0})
>>> td1 = TensorDict({"a": {"b1": 1}, "f": {"g": {"h": 1}}, "common": 1})
>>> td2 = TensorDict({"a": {"b2": 2}, "f": {"g": {"h": 2}}, "common": 2})
>>> td = merge_tensordicts(td0, td1, td2, callback_exist=lambda *v: torch.stack(list(v)))
>>> print(td)
TensorDict(
 fields={
 a: TensorDict(
 fields={
 b0: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 b2: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 c: TensorDict(
 fields={
 d: TensorDict(
 fields={
 e: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 common: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int64, is_shared=False),
 f: TensorDict(
 fields={
 g: TensorDict(
 fields={
 h: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(td["common"])
tensor([0, 1, 2])
```