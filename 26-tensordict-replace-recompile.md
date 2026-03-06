# Fix `replace()` Recompiles Under `torch.compile`

**Status:** Open
**Priority:** High
**Category:** Performance / torch.compile
**Difficulty:** Medium
**Upstream:** tensordict

## Problem

`MjTensorClass.replace(**kwargs)` causes O(N) recompiles under `torch.compile`,
where N is the number of unique kwarg patterns across all call sites. After
exceeding `cache_size_limit` (default 8), Dynamo gives up on `replace` entirely
and falls back to eager — turning every subsequent `replace` call into a **graph
break**.

In a single `mujoco_torch.step()` call, `replace` is invoked from ~15 distinct
sites with different kwargs (e.g. `replace(qvel=...)`, `replace(crb=...)`,
`replace(qfrc_smooth=..., qacc_smooth=...)`, etc.). This far exceeds the default
cache limit.

### Root cause

The current implementation:

```python
def replace(self, **kwargs):
    clone = self.clone(recurse=False)
    clone._tensordict._tensordict.update(kwargs)
    return clone
```

When Dynamo traces `dict.update(kwargs)`, it installs `DICT_KEYS_MATCH` guards
on both the target dict and the source kwargs dict
(`torch/_dynamo/variables/dicts.py`, `ConstDictVariable.call_method("update")`):

```python
self.install_dict_keys_match_guard()         # target dict
args[0].install_dict_keys_match_guard()      # source (kwargs) dict
```

`DICT_KEYS_MATCH` guards on `len(dict)` and the exact key names. When `replace`
is called from a different site with different kwargs, the source-dict guard
fails, triggering a recompile.

After 8 unique kwarg patterns, Dynamo marks `replace` as "skip" (falls back to
eager), and every subsequent call becomes a graph break.

### Precedent

`NNModuleHooksDictVariable` in Dynamo solves the identical problem for
`nn.Module` hook dicts by subclassing `ConstDictVariable` and overriding the
guard installation to be a no-op:

```python
class NNModuleHooksDictVariable(ConstDictVariable):
    def install_dict_keys_match_guard(self) -> None:
        pass
    def install_dict_contains_guard(self, tx, args) -> None:
        pass
```

### Key facts

- `_StringOnlyDict` (tensordict's internal storage) is just `dict`.
- `torch.compiler.is_compiling()` is constant-folded to `True` during Dynamo
  tracing — safe to branch on without introducing a guard.
- `torch.compiler.allow_in_graph` makes a function opaque to Dynamo — it is
  recorded as a single graph node, no tracing inside, no guards on args.
- `CALL_FUNCTION_KW` (keyword calls like `d.replace(qvel=v)`) extracts keys
  from a bytecode constant tuple — no `DICT_KEYS_MATCH` at the call site.
  The guard is installed **inside** `replace` when it operates on `kwargs`.

## Minimal Reproducible Example

```python
#!/usr/bin/env python3
"""MRE: replace() recompiles under torch.compile.

Run:
    TORCH_LOGS="+recompiles,graph_breaks" python this_script.py
"""
import time
import torch
from tensordict import TensorClass


class State(TensorClass["nocast"]):
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    w: torch.Tensor
    v: torch.Tensor

    def replace(self, **kwargs):
        clone = self.clone(recurse=False)
        clone._tensordict._tensordict.update(kwargs)
        return clone


def step(s: State) -> State:
    """Mimics a physics step that calls replace with different kwargs."""
    s = s.replace(x=s.x + 1)
    s = s.replace(y=s.y + 2)
    s = s.replace(z=s.z + 3)
    s = s.replace(x=s.x * 0.9, y=s.y * 0.9)
    s = s.replace(w=s.w + s.x)
    s = s.replace(v=s.v - 1, w=s.w + 1)
    s = s.replace(x=s.x + s.v, y=s.y + s.w, z=s.z + 0.1)
    s = s.replace(v=torch.zeros_like(s.v))
    s = s.replace(w=torch.ones_like(s.w))
    s = s.replace(x=s.x + s.y + s.z)
    return s


def main():
    s = State(
        x=torch.randn(4),
        y=torch.randn(4),
        z=torch.randn(4),
        w=torch.randn(4),
        v=torch.randn(4),
        batch_size=[4],
    )

    # ── Eager baseline ────────────────────────────────────────────────
    t0 = time.perf_counter()
    for _ in range(2000):
        s_out = step(s)
    eager_dt = time.perf_counter() - t0
    print(f"Eager:    {2000 / eager_dt:.0f} steps/s")

    # ── Compiled ──────────────────────────────────────────────────────
    step_c = torch.compile(step)

    # Warmup (triggers recompiles)
    print("\nCompiling (watch for recompiles / graph breaks in logs)...")
    s_out_c = step_c(s)

    # Check correctness
    s_out_eager = step(s)
    for field in ("x", "y", "z", "w", "v"):
        torch.testing.assert_close(
            getattr(s_out_c, field), getattr(s_out_eager, field),
            msg=lambda m: f"Field {field}: {m}",
        )
    print("Correctness: OK")

    # Benchmark compiled
    t0 = time.perf_counter()
    for _ in range(2000):
        s_out = step_c(s)
    compiled_dt = time.perf_counter() - t0
    print(f"Compiled: {2000 / compiled_dt:.0f} steps/s")
    print(f"Speedup:  {eager_dt / compiled_dt:.2f}x")


if __name__ == "__main__":
    main()
```

**Expected output** (current behavior): many `[__recompiles] Recompiling function
replace` lines and `[__graph_breaks]` lines. Compiled throughput is comparable
to or worse than eager because the graph is shattered.

## Proposed Solutions

An evaluation agent should implement **all five** variants below and benchmark
them. Each variant replaces only the `replace` method (or adds a decorator).
Measure:

1. Number of `[__recompiles]` lines from `replace` (target: 0)
2. Number of `[__graph_breaks]` lines from `replace` (target: 0)
3. Eager throughput (steps/s) — must not regress vs baseline
4. Compiled throughput (steps/s) — higher is better
5. Correctness — outputs match eager baseline

### Solution A: `allow_in_graph` on `replace`

Mark the entire `replace` method as opaque to Dynamo. Dynamo records each call
as a single graph node without tracing inside — no kwargs guards, no recompiles.

```python
@torch.compiler.allow_in_graph
def replace(self, **kwargs):
    clone = self.clone(recurse=False)
    clone._tensordict._tensordict.update(kwargs)
    return clone
```

**Pro:** Simplest change; no kwargs tracing means zero recompiles.
**Risk:** `allow_in_graph` may not support user-defined object (TensorClass)
inputs/outputs. Test whether FakeTensor propagation works.

### Solution B: `is_compiling()` + `allow_in_graph` inner helper

Keep the eager path unchanged. Under compile, delegate the dict mutation to an
`allow_in_graph` helper, passing kwargs as a positional dict (not `**`-unpacked)
to avoid `CALL_FUNCTION_EX` guards:

```python
def replace(self, **kwargs):
    clone = self.clone(recurse=False)
    if torch.compiler.is_compiling():
        _replace_update(clone._tensordict._tensordict, kwargs)
    else:
        clone._tensordict._tensordict.update(kwargs)
    return clone

@torch.compiler.allow_in_graph
def _replace_update(target_dict, source_dict):
    target_dict.update(source_dict)
```

**Pro:** Eager path is untouched; compiled path avoids kwargs guards.
**Risk:** Same `allow_in_graph` + user-defined-object risk as A, but the opaque
boundary is smaller (only the dict mutation, not clone/return).

### Solution C: `_specialize_in_compile` decorator

When compiling, dispatch to a per-keyset specialized function. Each specialized
function captures the kwargs keys in its closure (compile-time constant) and
takes values as positional args — eliminating the kwargs dict inside the hot
function:

```python
import functools

_replace_cache: dict[tuple[str, ...], callable] = {}

def _get_replace_fn(keys: tuple[str, ...]):
    fn = _replace_cache.get(keys)
    if fn is not None:
        return fn

    def _do_replace(self, *values):
        clone = self.clone(recurse=False)
        td = clone._tensordict._tensordict
        for k, v in zip(keys, values):
            td[k] = v
        return clone

    _replace_cache[keys] = _do_replace
    return _do_replace

def replace(self, **kwargs):
    if not torch.compiler.is_compiling():
        clone = self.clone(recurse=False)
        clone._tensordict._tensordict.update(kwargs)
        return clone
    keys = tuple(sorted(kwargs))
    fn = _get_replace_fn(keys)
    return fn(self, *(kwargs[k] for k in keys))
```

**Pro:** Each inner function has a fixed signature → stable Dynamo guards.
**Con:** The outer `replace` still recompiles per unique keyset (the kwargs
dict is examined to compute `keys`). Need enough cache entries to cover all
patterns. Test whether `cache_size_limit` still bites.

### Solution D: Individual `__setitem__` under `is_compiling()`

Replace `dict.update(kwargs)` with individual dict assignments. The hypothesis
is that `__setitem__` with a constant string key produces a lighter guard than
`update` with a variable-keyed source dict:

```python
def replace(self, **kwargs):
    clone = self.clone(recurse=False)
    td = clone._tensordict._tensordict
    if torch.compiler.is_compiling():
        for k, v in kwargs.items():
            td[k] = v
    else:
        td.update(kwargs)
    return clone
```

**Pro:** Minimal change; easy to test.
**Con:** `kwargs.items()` still installs `DICT_KEYS_MATCH` on the kwargs dict,
so this may produce the same recompiles. Worth testing to confirm.

### Solution E: Reconstruct via `{**base, **overrides}`

Instead of clone-then-mutate, build a fresh internal dict and assign it to the
clone. This avoids calling `dict.update` on the target entirely:

```python
def replace(self, **kwargs):
    clone = self.clone(recurse=False)
    if torch.compiler.is_compiling():
        clone._tensordict._tensordict = {
            **clone._tensordict._tensordict, **kwargs
        }
    else:
        clone._tensordict._tensordict.update(kwargs)
    return clone
```

**Pro:** No mutation of the target dict; no `DICT_KEYS_MATCH` on the target.
**Con:** `{**d1, **d2}` may still install guards on both source dicts during
unpacking. Test to confirm.

## Evaluation Instructions

The evaluation agent should:

1. Start from the MRE above.
2. For each solution (A through E), replace the `replace` method with the
   proposed variant.
3. Run with `TORCH_LOGS="+recompiles,graph_breaks"` and count:
   - `[__recompiles]` lines mentioning `replace`
   - `[__graph_breaks]` lines mentioning `replace`
4. Run the timing benchmark (eager + compiled throughput).
5. Verify correctness (the `assert_close` check in the MRE).
6. Report a table:

   | Solution | Recompiles | Graph breaks | Eager steps/s | Compiled steps/s | Correct |
   |----------|-----------|--------------|--------------|-----------------|---------|
   | Baseline | ? | ? | ? | ? | Yes |
   | A | ? | ? | ? | ? | ? |
   | ... | | | | | |

7. Select the solution with: zero recompiles from replace, zero graph breaks
   from replace, no eager regression, best compiled throughput.
8. If multiple solutions tie, prefer the simplest.
9. If no solution achieves zero recompiles AND zero graph breaks, report which
   comes closest and what the remaining issue is.
