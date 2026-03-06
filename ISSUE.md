
## 1. TensorClass `.get(key)` returns unwrapped values, but `is_non_tensor()` only detects `NonTensorData`

```python
@tensorclass
class Transition:
    obs: torch.Tensor
    label: str

t = Transition(obs=torch.randn(4), label="cat", batch_size=[])
t.get("label")           # returns "cat" (plain str)
is_non_tensor(t.get("label"))  # False!
```

Compare with plain TensorDict:
```python
td = TensorDict({"label": "cat"}, [])
td.get("label")           # returns NonTensorData(data='cat', ...)
is_non_tensor(td.get("label"))  # True
```

So the same logical operation — "is this field a non-tensor leaf?" — gives different answers depending on whether data lives in a TensorDict vs a TensorClass. This makes it hard to write generic code that classifies keys.

**Suggestion:** Either (a) have a utility like `is_non_tensor_key(td, key)` that checks the internal representation regardless of unwrapping, or (b) document that `is_non_tensor` should not be used on TensorClass `.get()` results, providing an anative.

I worked around this by using `isinstance(val, torch.Tensor)` as the positive check (tensor leaf) and treating everything else as non-tensor.

---

## 2. `TensorClass.select()` preserves the type with None defaults

```python
t = Transition(obs=torch.randn(4), action=torch.tensor([1.0]), label="cat", batch_size=[])
t.select("obs", "action")
# Returns: Transition(obs=..., action=..., label=None, batch_size=[])
```

This is sensible behavior in general, but it means you can't get a "plain TensorDict subset" from a TensorClass without the unselected fields appearing as `None`. When passing this to `TensorDictStore.from_tensordict()`, the `from_tensordict` method detects the TensorClass type and stores the class path. Later reads from that store might try to reconstruct the full TensorClass even though only a subset of keys was stored.

**Not necessarily a bug** — just worth being aware of. A `select(..., as_tensordict=True)` or `to_tensordict().select(...)` would be useful for cases where you explicly want to drop the TensorClass wrapper.

---

## 3. The C extension build is fragile with Python 3.13

The `_C.so` pybind11 extension repeatedly failed with:
```
symbol not found in flat namespace '__PyThreadState_UncheckedGet'
```

This is `_PyThreadState_UncheckedGet` being removed in Python 3.13's stable ABI. The fix was upgrading pybind11 from 3.0.1 to 3.0.2 and doing a clean rebuild (`rm -rf build tensordict/_C.so`). But pip's editable install doesn't always trigger a C extension rebuild when the cached `.so` is stale, so the error kept reappearing between install attempts.

**Suggestion:** Pin `pybind11>=3.0.2` in the build requirements for Python 3.13+ compatibility. Consider adding a version check or a clearer error message when the C extension fails to load.

---

## Summary

| Priority | Issue | Impact |
|----------|-------|--------|
| **Medium** | `is_non_tensor` inconsistency with TensorClass (#1) | Makes generic key classification unreliable |
| **Low** | `select()` preserving TensorClass type (#2) | Workaround: call `.to_tensordict()` first |
| **Low** | pybind11/Python 3.13 build (#3) | One-time fix, just needs a version bump |
