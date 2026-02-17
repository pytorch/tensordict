# Optimizing Indexed Operations for RedisTensorDict

## Problem Statement

Indexed read (`td[idx]`) and indexed write (`td[idx] = subtd`) on a `RedisTensorDict`
transfer **the entire tensor** for every leaf key, even when only a small slice is needed.
This is a critical performance bottleneck because indexed access is the single most common
operation in TensorDict-based data pipelines (replay buffers, data collectors, batched envs).

### Concrete example

A replay buffer stores 1 million experience frames:

```python
buffer = RedisTensorDict(batch_size=[1_000_000])
buffer["obs"] = torch.randn(1_000_000, 3, 224, 224)   # ~560 GB in Redis
buffer["action"] = torch.randn(1_000_000, 6)            # ~22 MB in Redis
```

Sampling a minibatch of 256 frames:

```python
idx = torch.randint(0, 1_000_000, (256,))
minibatch = buffer[idx]          # READ: should transfer ~147 MB, currently ~560 GB
buffer[idx] = new_experience     # WRITE: should transfer ~147 MB, currently ~1.12 TB
```

The read downloads the full 560 GB tensor and indexes locally.
The write downloads 560 GB, patches 256 rows in memory, and re-uploads 560 GB.

With 2 leaf keys the transfer balloons to ~2.24 TB for an operation that logically touches
~294 MB of data. **That is a ~7,600x overhead.**

---

## Current Code Path

### Write: `td[idx] = subtd`

```
__setitem__(idx, subtd)                          # redis.py:543
 └─ _get_sub_tensordict(idx)                     # base.py:655  → _SubTensorDict
     └─ sub_td.update(subtd, inplace=True)       # _td.py:3977
         └─ for each leaf key:
             _set_at_str(key, val, idx)           # redis.py:714
              ├─ _aget_tensor(key_path)           # GET full tensor bytes + metadata
              ├─ existing[idx] = val              # patch in local memory
              └─ _aset_tensor(key_path, existing) # SET full tensor bytes + metadata
```

**Cost per leaf key:** 1 GET (full tensor) + 1 SET (full tensor) = 2 round-trips, transferring `2 × full_size` bytes.
**Cost for N keys:** 2N round-trips, 2N × full_size bytes total.

### Read: `subtd = td[idx]`

```
__getitem__(idx)                                 # base.py:655
 └─ _get_sub_tensordict(idx)                     # returns lazy _SubTensorDict
     └─ sub_td["obs"]                            # triggers _SubTensorDict._get_str
         └─ source._get_at_str("obs", idx)       # base.py:7883
             ├─ source._get_str("obs")           # GET full tensor from Redis
             └─ out[idx]                         # index locally
```

**Cost per leaf key:** 1 GET (full tensor) = 1 round-trip, transferring `full_size` bytes.
Reads are lazy (per-key), so every key access triggers a separate full download.

---

## Solution: Byte-Range Operations (`GETRANGE` / `SETRANGE`)

Redis `GETRANGE key start end` and `SETRANGE key offset value` operate on byte sub-ranges
of a string value without touching the rest. Since our tensors are stored as raw contiguous
bytes, indexing along the batch dimension(s) maps directly to byte offsets.

### Why it works

For a C-contiguous tensor of shape `[B, *rest]` with element size `e`:

```
row_size  = prod(rest) × e           # bytes per batch element
offset(i) = i × row_size             # byte offset of row i
```

| Index pattern            | Byte ranges                              | Contiguous? |
|--------------------------|------------------------------------------|-------------|
| `td[5]`                  | `[offset(5), offset(5) + row_size)`      | Single range |
| `td[2:8]`               | `[offset(2), offset(8))`                 | Single range |
| `td[::2]`               | `offset(0), offset(2), offset(4), ...`   | Multiple ranges |
| `td[[1, 7, 42]]`        | `offset(1), offset(7), offset(42)`       | Multiple ranges |
| `td[bool_mask]`          | Convert to int indices, then as above    | Multiple ranges |

Every case decomposes into a list of `(byte_offset, byte_length)` pairs.
Each pair becomes one `GETRANGE` or `SETRANGE` command.
All commands are **pipelined** into a single round-trip.

### Cost comparison

| Scenario | Current | With byte-range ops |
|----------|---------|---------------------|
| `td[0:10]`, 1 key, 1M×[3,224,224] f32 | 2 RT, ~1.12 TB | 2 RT, ~5.9 MB |
| `td[idx]` 256 random, 1 key, 1M×[3,224,224] f32 | 2 RT, ~1.12 TB | 2 RT, ~150 MB |
| `td[0:10] = v`, 100 keys, 1M×[3,224,224] f32 | 200 RT, ~112 TB | 2 RT, ~590 MB |
| `td[0]`, 10 keys, 10×[3] f32 | 10 RT, 1.2 KB | 2 RT, ~1.2 KB |

*RT = Redis round-trips. "2 RT" = 1 pipeline for metadata + 1 pipeline for byte-range ops.*

---

## Detailed Design

### 1. Byte-offset computation

```python
def _compute_byte_ranges(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> list[tuple[int, int]]:
    """Return [(byte_offset, byte_length), ...] for the given index.

    Only indexes into the first dimension (batch dim).
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()
    row_size = elem_size
    for s in shape[1:]:
        row_size *= s

    # Normalize idx to a list of integer positions
    if isinstance(idx, int):
        positions = [idx % shape[0]]
    elif isinstance(idx, slice):
        positions = list(range(*idx.indices(shape[0])))
    elif isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            positions = idx.nonzero(as_tuple=False).squeeze(-1).tolist()
        else:
            positions = idx.tolist()
    elif isinstance(idx, (list, range)):
        positions = list(idx)
    else:
        return None  # Signal: can't compute, fall back

    return [(p * row_size, row_size) for p in positions]
```

For **multi-dimensional batch sizes** `[B1, B2, ...]` with a tuple index `(i, j, ...)`:
the offset computation generalizes via strides, same as NumPy/PyTorch C-contiguous layout.
Non-contiguous access (e.g., `td[:, 5]` on batch `[B1, B2]`) produces a list of
non-adjacent ranges — still pipelineable, just more GETRANGE/SETRANGE commands.

If `_compute_byte_ranges` returns `None` (unsupported exotic index), fall back to the
current full read-modify-write (pipelined across keys, see fallback below).

### 2. Pipelined batch write: `_abatch_set_at`

```python
async def _abatch_set_at(
    self,
    items: dict[str, tuple[torch.Tensor, Any]],  # {key_path: (value, idx)}
):
    key_paths = list(items.keys())

    # ── Round-trip 1: fetch metadata for all keys ──
    pipe = self._client.pipeline()
    for kp in key_paths:
        pipe.hgetall(self._meta_key(kp))
    all_meta = await pipe.execute()

    # ── Compute byte ranges ──
    setrange_cmds = []          # (redis_key, byte_offset, chunk_bytes)
    fallback_keys = []          # keys that need full read-modify-write
    for kp, raw_meta in zip(key_paths, all_meta):
        meta = _decode_meta(raw_meta)
        shape = json.loads(meta["shape"])
        dtype = _str_to_dtype(meta["dtype"])
        value, idx = items[kp]
        ranges = _compute_byte_ranges(shape, dtype, idx)
        if ranges is None:
            fallback_keys.append(kp)
            continue
        value_bytes = _tensor_to_bytes(value.contiguous())
        offset = 0
        for byte_offset, byte_length in ranges:
            setrange_cmds.append((
                self._data_key(kp), byte_offset,
                value_bytes[offset:offset + byte_length],
            ))
            offset += byte_length

    # ── Round-trip 2: pipeline all SETRANGE commands ──
    if setrange_cmds:
        pipe = self._client.pipeline()
        for redis_key, byte_offset, chunk in setrange_cmds:
            pipe.setrange(redis_key, byte_offset, chunk)
        await pipe.execute()

    # ── Fallback: pipelined full read-modify-write for exotic indices ──
    if fallback_keys:
        await self._abatch_read_modify_write(fallback_keys, items)
```

**Total: 2 round-trips** (metadata + SETRANGE), transferring only slice-sized data.

### 3. Pipelined batch read: `_abatch_get_at`

```python
async def _abatch_get_at(
    self,
    key_paths: list[str],
    idx,
) -> dict[str, torch.Tensor]:
    # ── Round-trip 1: fetch metadata ──
    pipe = self._client.pipeline()
    for kp in key_paths:
        pipe.hgetall(self._meta_key(kp))
    all_meta = await pipe.execute()

    # ── Compute byte ranges, issue GETRANGE ──
    pipe = self._client.pipeline()
    range_plan = []   # [(kp, shape, dtype, n_ranges)]
    for kp, raw_meta in zip(key_paths, all_meta):
        meta = _decode_meta(raw_meta)
        shape = json.loads(meta["shape"])
        dtype = _str_to_dtype(meta["dtype"])
        ranges = _compute_byte_ranges(shape, dtype, idx)
        for byte_offset, byte_length in ranges:
            pipe.getrange(self._data_key(kp), byte_offset,
                          byte_offset + byte_length - 1)
        range_plan.append((kp, shape, dtype, len(ranges)))

    # ── Round-trip 2: execute ──
    results_flat = await pipe.execute()

    # ── Reassemble tensors ──
    result = {}
    flat_idx = 0
    for kp, shape, dtype, n_ranges in range_plan:
        chunks = results_flat[flat_idx:flat_idx + n_ranges]
        flat_idx += n_ranges
        data = b"".join(chunks)
        result_shape = _getitem_result_shape(shape, idx)
        result[kp] = _bytes_to_tensor(data, result_shape, dtype)
    return result
```

### 4. Fallback: pipelined read-modify-write

When byte-range ops can't be used (exotic multi-dim indices, non-tensor values),
fall back to a **pipelined** version of the current approach:

```python
async def _abatch_read_modify_write(self, key_paths, items):
    # ── Pipeline GET all ──
    pipe = self._client.pipeline()
    for kp in key_paths:
        pipe.get(self._data_key(kp))
        pipe.hgetall(self._meta_key(kp))
    results = await pipe.execute()

    # ── Patch all in memory ──
    pipe = self._client.pipeline()
    for i, kp in enumerate(key_paths):
        data, raw_meta = results[2*i], results[2*i + 1]
        meta = _decode_meta(raw_meta)
        shape = json.loads(meta["shape"])
        dtype = _str_to_dtype(meta["dtype"])
        existing = _bytes_to_tensor(data, shape, dtype)
        value, idx = items[kp]
        existing[idx] = value
        pipe.set(self._data_key(kp), _tensor_to_bytes(existing))
    await pipe.execute()
```

**2 round-trips** instead of 2N, still transferring full tensors but eliminating per-key
latency.

### 5. Hooking into TensorDictBase

Override `__setitem__` for index-based assignment to bypass `_SubTensorDict`:

```python
def __setitem__(self, index, value):
    # Key-based assignment — unchanged
    index_unravel = _unravel_key_to_tuple(index)
    if index_unravel:
        return self.set(index_unravel, value, inplace=True)

    # Index-based — collect all leaf (key_path, tensor, idx) and batch
    if not isinstance(value, TensorDictBase):
        value = TensorDict.from_dict(value, batch_size=[])
    items = {}
    for key in value.keys(include_nested=True, leaves_only=True):
        key_path = self._full_key_path(
            _KEY_SEP.join(_unravel_key_to_tuple(key))
        )
        items[key_path] = (value.get(key), index)
    self._run_sync(self._abatch_set_at(items))
```

Similarly, override `_get_at_str` / add `_get_at_batch` for reads:

```python
def _get_at_str(self, key, idx, default=NO_DEFAULT, **kwargs):
    key_path = self._full_key_path(key)
    result = self._run_sync(self._abatch_get_at([key_path], idx))
    tensor = result.get(key_path)
    if tensor is None:
        if default is not NO_DEFAULT:
            return default
        raise KeyError(key)
    return tensor
```

### 6. Optional: metadata caching

Since we store shape/dtype in Redis hashes AND we know them at write time, we can
cache metadata locally in a `dict`:

```python
self._meta_cache: dict[str, tuple[list[int], torch.dtype]] = {}
```

Updated on every `_aset_tensor` call, consulted before hitting Redis in `_abatch_set_at`
/ `_abatch_get_at`. This eliminates the metadata round-trip entirely, reducing to
**1 round-trip** for both reads and writes.

**Trade-off:** if multiple clients write to the same `RedisTensorDict`, the cache can
become stale. An optional `cache_metadata=True` flag (default `True`, disable for
multi-writer scenarios) would give the user control.

---

## Edge Cases and Fallbacks

| Case | Handling |
|------|----------|
| Non-contiguous index (e.g., `td[::3]`) | Multiple GETRANGE/SETRANGE per key, all pipelined |
| Fancy index (tensor/list) | Convert to positions, pipeline individual ranges |
| Boolean mask | `mask.nonzero()` → integer indices → ranges |
| Multi-dim batch index `td[i, j]` | Stride-based offset, single range |
| Non-first-dim index `td[:, j]` on batch `[B1, B2]` | Multiple non-adjacent ranges, all pipelined |
| Non-tensor values (JSON/pickle) | Full read-modify-write (pipelined fallback) |
| Empty index `td[torch.tensor([])]` | No-op |
| Key doesn't exist | `KeyError` (same as today) |
| Index out of bounds | Let it fail at byte computation time |
| Multi-dim tensor with extra dims beyond batch | Ranges still computed on batch dims only; the "row" includes all extra dims |

---

## Implementation Plan

### Phase 1 — Pipelined read-modify-write (quick win)

Replace the per-key serial read-modify-write with the batched `_abatch_read_modify_write`.
No byte-range logic yet. Cuts round-trips from 2N to 2. Data transfer unchanged.

**Impact:** eliminates network latency as a scaling factor in N (number of keys).

### Phase 2 — GETRANGE / SETRANGE for integer and contiguous slices

Add `_compute_byte_ranges` for integer indices and step-1 slices.
Wire `_abatch_set_at` and `_abatch_get_at` into `__setitem__` and `_get_at_str`.

**Impact:** reduces data transfer from O(full_size) to O(slice_size) for the most common
access patterns (single element, contiguous slice).

### Phase 3 — Fancy indexing (tensor/list/bool)

Extend `_compute_byte_ranges` to handle arbitrary integer lists, tensor indices,
and boolean masks. Each selected row becomes one GETRANGE/SETRANGE in the pipeline.

**Impact:** replay buffer random sampling (`td[randint_idx]`) becomes efficient.

### Phase 4 — Metadata caching

Cache shape/dtype locally on write, skip metadata round-trip on read.
Add `cache_metadata` flag to constructor.

**Impact:** cuts round-trips from 2 to 1 for all indexed operations.

### Phase 5 — Multi-dimensional batch indices

Generalize byte-offset computation for tuple indices on multi-dimensional batch sizes.
`td[i, j]` on batch `[B1, B2]` computes offset from strides.

**Impact:** correctness for all batch indexing patterns.

---

## Open Questions

1. **Should `td[idx]` return a materialized `TensorDict` or a lazy view?**
   Currently it returns a `_SubTensorDict` (lazy). With GETRANGE, returning a materialized
   `TensorDict` is natural and avoids the full-tensor download on subsequent key access.
   But it changes semantics: mutations on the result wouldn't propagate back to Redis.
   A `_SubTensorDict` that uses GETRANGE internally is more faithful but more complex.

2. **Should we cache metadata by default?**
   It halves the round-trips but introduces staleness risk for multi-client setups.
   Default-on with an opt-out flag seems reasonable.

3. **What about very large slices?**
   If `idx` is `td[:999_999]` out of 1M, GETRANGE transfers almost the full tensor anyway.
   Not worse than today, but also not a win. Should we detect this and fall back to full
   GET for simplicity? (Probably not — GETRANGE is never slower than GET.)

4. **Thread-safety of `_abatch_set_at` / `_abatch_get_at`?**
   The current design uses a single async event loop on a background thread. Multiple
   Python threads calling `_run_sync` concurrently should be safe because each call gets
   its own `Future`, but the pipeline construction itself needs to be atomic (no interleaving).
   The current per-call pipeline creation should be fine.

5. **`SETRANGE` on non-existent keys?**
   Redis `SETRANGE` on a missing key pads with zero bytes. We should ensure the full tensor
   is written via `SET` first (which already happens in `_aset_tensor`) and only use
   `SETRANGE` for subsequent indexed updates. The existence check is implicit: if the key
   exists in `__keys__`, the full tensor was already written.
