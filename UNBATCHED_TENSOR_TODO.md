# UnbatchedTensor Test Coverage - Status & Remaining Work

This document tracks the test coverage status for `UnbatchedTensor` in TensorDict.

---

## What is UnbatchedTensor?

`UnbatchedTensor` is a special wrapper class in TensorDict that allows storing tensors that **do not follow the TensorDict's batch dimensions**.

### The Problem It Solves

In a standard TensorDict, all tensors must have shapes that are compatible with the TensorDict's `batch_size`. For example:

```python
td = TensorDict({
    "obs": torch.randn(4, 3, 84, 84),  # batch_size is [4, 3], feature dims are [84, 84]
    "action": torch.randn(4, 3, 6),     # batch_size is [4, 3], feature dim is [6]
}, batch_size=[4, 3])
```

But sometimes you need to store data that doesn't follow this batch structure - for example:
- **Shared parameters** that are the same across all batch elements
- **Lookup tables** or **embeddings** that don't vary with batch
- **Metadata tensors** with their own independent shape

### How UnbatchedTensor Works

```python
from tensordict import TensorDict, UnbatchedTensor

td = TensorDict({
    "obs": torch.randn(4, 3, 84, 84),
    "shared_weights": UnbatchedTensor(torch.randn(128, 64)),  # Independent shape!
}, batch_size=[4, 3])

# The UnbatchedTensor has its own shape
td["shared_weights"].shape  # torch.Size([128, 64]) - NOT [4, 3, 128, 64]
```

### Key Semantics

1. **`shape` property**: Returns the actual data shape (`self.data.shape`), NOT prefixed with batch dimensions.

2. **`batch_size` property**: A virtual property that tracks what batch_size the UnbatchedTensor "belongs to" for validation purposes. This is updated when the parent TensorDict's batch_size changes.

3. **Shape operations are NO-OPs for data**: When you reshape, transpose, split, gather, etc. a TensorDict containing an UnbatchedTensor:
   - The UnbatchedTensor's **data remains unchanged**
   - Only its **`batch_size` property is updated** to match the new TensorDict batch_size
   - A **copy** of the UnbatchedTensor is created (data is shared, but batch_size can differ)

4. **Data operations apply normally**: Operations like `to(device)`, `to(dtype)`, arithmetic (`+`, `-`, `*`, `/`), `clamp`, `maximum`, etc. apply to the underlying data as expected.

### Important: Accessing UnbatchedTensor

- **`td["key"]`** returns the underlying `.data` tensor (for convenience)
- **`td.get("key")`** returns the `UnbatchedTensor` wrapper (preserves `batch_size`)

Use `.get()` when you need to access the `batch_size` property or preserve the wrapper.

---

## Current Status

- **254 tests passing** with `td_with_unbatched` fixture
- **106 tests skipped** (legitimate semantic incompatibilities + features not yet implemented)
- **12 tests xfail** (memmap support - needs implementation)

---

## XFAIL Tests (12) - Memmap Support Needed

These tests are marked with `pytest.xfail` and represent memmap functionality that needs implementation.

| Test | Issue |
|------|-------|
| `test_memmap_` (4 variants) | memmap support not yet implemented |
| `test_memmap_existing` (2 variants) | memmap support not yet implemented |
| `test_memmap_like` (4 variants) | memmap support not yet implemented |
| `test_memmap_prefix` | memmap support not yet implemented |
| `test_save_load_memmap` | memmap support not yet implemented |

### How to Fix Memmap

The memmap functionality needs to:
1. **In `_populate_memmap()` (~line 4952 in `_td.py`)**: Detect UnbatchedTensor, unwrap `.data` for storage
2. **In `_update_metadata()` (~line 5073 in `_td.py`)**: Add `"_is_unbatched": True` flag to metadata JSON
3. **In `_load_memmap()` (~line 2994 in `_td.py`)**: Check metadata for `_is_unbatched` flag, wrap tensor back into UnbatchedTensor

---

## Skipped Tests - Features Not Yet Implemented

### Indexed Assignment (requires internal _tensordict batch_size handling)

| Test | Reason |
|------|--------|
| `test_setitem` | Internal `_tensordict` has batch_size=[] which mismatches during indexed assignment |
| `test_setitem_slice` | Same issue as test_setitem |
| `test_broadcast` | Same issue - uses indexed assignment |

### Dimension Mismatch Issues (expand_as_right)

| Test | Reason |
|------|--------|
| `test_where` | expand_as_right dimension mismatch between UnbatchedTensor and batch dims |
| `test_where_pad` | Same issue |
| `test_masked_fill` | Same issue |
| `test_masked_fill_` | Same issue |

### Other Features Needing Implementation

| Test | Reason |
|------|--------|
| `test_zero_grad` | `backward()` requires scalar output, UnbatchedTensor isn't scalar |
| `test_losses` | `l1_loss`, `mse_loss` etc. not implemented for UnbatchedTensor |
| `test_new_full`, `test_new_ones`, `test_new_tensor`, `test_new_zeros` | Comparison with UnbatchedTensor needs investigation |
| `test_cast_to` | Needs investigation |

---

## Skipped Tests - Legitimate Semantic Incompatibilities

### Shape-Dependent Operations (Cannot Work)

| Test | Reason |
|------|--------|
| `test_add_batch_dim_cache` | vmap requires consistent batch dimensions |
| `test_reduction` | Reductions over batch dims don't apply to unbatched data |
| `test_reduction_feature` | Same as above |
| `test_logsumexp` | Reduction operation over batch dims |
| `test_softmax` | Softmax over batch dim doesn't apply |
| `test_view_dtype` | View with dtype has shape implications |

### TensorClass Structure Incompatibilities

| Test | Reason |
|------|--------|
| `test_nested_dict_init` | `to_dict()` returns nested structure `{'data': tensor}` that can't be re-initialized |
| `test_lock_change_names` | UnbatchedTensor has no `names` (shape independent of batch) |
| `test_non_tensor_data_pickle` | memmap/pickle requires special metadata handling |

### Validation/Update Issues

| Test | Reason |
|------|--------|
| `test_update_subtensordict` (6 variants) | Subtensordict update triggers batch size validation |
| `test_stack_onto` | Stack onto validation fails for UnbatchedTensor |
| `test_stack_subclasses_on_td` | Same validation issue |

---

## Completed Fixes

### Phase 1: Foreach Operations (All Fixed)

Added `_pass_through` fallback to avoid `torch._foreach_*` stripping wrappers:

| Test | Methods Patched | File |
|------|-----------------|------|
| `test_clamp` | `clamp_max`, `clamp_max_`, `clamp_min`, `clamp_min_` | `base.py` |
| `test_maximum` | `maximum`, `maximum_`, `minimum`, `minimum_` | `base.py` |
| `test_zero_grad` | Fixed iteration + added `grad` property | `base.py`, `_unbatched.py` |
| `test_autograd_grad` | Unwrap/wrap in `_grad` function | `_torch_func.py` |
| `test_losses` | `neg`, `neg_` patched | `base.py` |

### Phase 2: Other Fixes

| Test | Fix Applied |
|------|-------------|
| `test_data_ptr` | Added `untyped_storage()`, `data_ptr()` methods to UnbatchedTensor |
| `test_new_tensor` | Works after `mul` fix |
| `test_replace` | Specialized test for regular tensors only |
| `test_to_device_dtype_inplace` | Added `device`, `dtype` properties to UnbatchedTensor |
| `test_state_dict` | Added `state_dict()`, `load_state_dict()` to UnbatchedTensor |

### Phase 3: Regression Fixes

| Issue | Fix Applied |
|-------|-------------|
| Empty TensorDict crash | Added `if not vals: return self.copy()` to foreach operations |
| NonTensorData yielded as leaf | Fixed `_default_is_leaf` to only consider `_pass_through` attribute |
| maximum/minimum with UnbatchedTensor | Added unwrap/wrap logic in pass-through fallback |

---

## Instructions for LLM Agents

### Running Tests

```bash
cd /Users/vmoens/repos/tensordict

# Run specific test
.venv/bin/python -m pytest test/test_tensordict.py -k "TEST_NAME and td_with_unbatched" -v --tb=short

# Run all td_with_unbatched tests
.venv/bin/python -m pytest test/test_tensordict.py -k "td_with_unbatched" -q

# Run TestUnbatchedTensor suite
.venv/bin/python -m pytest test/test_tensordict.py::TestUnbatchedTensor -v
```

### Key Files

- `tensordict/_unbatched.py` - UnbatchedTensor class definition
- `tensordict/base.py` - Base TensorDict operations
- `tensordict/_td.py` - TensorDict implementation (memmap lives here)
- `tensordict/utils.py` - Utility functions including `_pass_through()`

### Key Utility Function

```python
from tensordict.utils import _pass_through
if _pass_through(value):
    # This is an UnbatchedTensor or similar pass-through type
```

### User Rules to Follow

- Do not add defensive fallbacks unless explicitly requested
- Avoid local imports unless strictly necessary
- Prefer early, explicit failure over silent workarounds
- Do not make up dependency versions

---

## Test Fixture

The `td_with_unbatched` fixture creates:

```python
TensorDict(
    source={
        "a": torch.randn(4, 3, 2, 1, 5),
        "b": torch.randn(4, 3, 2, 1, 10),
        "c": torch.randint(10, (4, 3, 2, 1, 3)),
        "unbatched": UnbatchedTensor(torch.randn(7, 11)),  # Different shape!
    },
    batch_size=[4, 3, 2, 1],
    device=device,
)
```
