# merge_tensordicts

tensordict.merge_tensordicts(**tensordicts: T*, *callback_exist: [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [Any](https://docs.python.org/3/library/typing.html#typing.Any)] | [Dict](https://docs.python.org/3/library/typing.html#typing.Dict)[[NestedKey](tensordict.NestedKey.html#tensordict.NestedKey), [Callable](https://docs.python.org/3/library/typing.html#typing.Callable)[[[Any](https://docs.python.org/3/library/typing.html#typing.Any)], [Any](https://docs.python.org/3/library/typing.html#typing.Any)]] | [None](https://docs.python.org/3/builtins/constants.html#None) = None*) → T

Merges the entries of several tensordicts into a new tensordict.

The result has the batch size of the first tensordict and every key of
every input. The values are not copied. For a key that several inputs
hold, the value comes from the first input that holds it, unless
`callback_exist` combines them.

Parameters:

***tensordicts** ([*TensorDictBase*](tensordict.TensorDictBase.html#tensordict.TensorDictBase)) - the tensordicts to merge, at least two,
or a `RuntimeError` is raised.

Keyword Arguments:

**callback_exist** (*Callable**or*[*dict*](https://docs.python.org/3/builtins/stdtypes.html#dict)*of**NestedKey to Callable**,**optional*) - a function called as `callback_exist(*values)` for a leaf that
every input holds; its result becomes the value of the leaf. A
dict gives a function per key (`"a"` or `("a", "b")`), and
the keys it lacks take the first value. Leaves that only some
inputs hold, and non-tensor entries, always take the first value.
Defaults to `None`, which takes the first value for every key.

Examples

```
>>> import torch
>>> from tensordict import merge_tensordicts, TensorDict
>>> td0 = TensorDict({"a": {"b0": 0}, "common": 0})
>>> td1 = TensorDict({"a": {"b1": 1}, "f": 1, "common": 1})
>>> td2 = TensorDict({"a": {"b2": 2}, "common": 2})
>>> td = merge_tensordicts(td0, td1, td2)
>>> list(td.keys(include_nested=True, leaves_only=True, sort=True))
[('a', 'b0'), ('a', 'b1'), ('a', 'b2'), 'common', 'f']
>>> td["common"]
tensor(0)
>>> td = merge_tensordicts(td0, td1, td2, callback_exist=lambda *v: torch.stack(v))
>>> td["common"]
tensor([0, 1, 2])
```