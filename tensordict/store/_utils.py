# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import torch

__all__ = [
    "_LUA_GETRANGES",
    "_LUA_SETRANGES",
    "_bytes_to_tensor",
    "_compute_byte_ranges",
    "_compute_covering_range",
    "_decode_meta",
    "_dtype_to_str",
    "_get_local_idx",
    "_getitem_result_shape",
    "_is_scattered_index",
    "_normalize_index",
    "_normalize_non_tensor_index",
    "_prepare_indexed_value",
    "_str_to_dtype",
    "_tensor_to_bytes",
]

# Lua scripts executed server-side to batch byte-range operations into a
# single round-trip per key. Each script takes ONE Redis key and a flat
# argument list encoding the byte ranges.

# GETRANGES: ARGV = [offset1, length1, offset2, length2, ...]
# Returns the *concatenated* bytes from all ranges.
_LUA_GETRANGES = """\
local key = KEYS[1]
local parts = {}
for i = 1, #ARGV, 2 do
    local off = tonumber(ARGV[i])
    local len = tonumber(ARGV[i + 1])
    parts[#parts + 1] = redis.call('GETRANGE', key, off, off + len - 1)
end
return table.concat(parts)
"""

# SETRANGES: ARGV = [offset1, data1, offset2, data2, ...]
# Applies all SETRANGEs atomically; returns "OK".
_LUA_SETRANGES = """\
local key = KEYS[1]
for i = 1, #ARGV, 2 do
    redis.call('SETRANGE', key, tonumber(ARGV[i]), ARGV[i + 1])
end
return redis.status_reply('OK')
"""


def _dtype_to_str(dtype: torch.dtype) -> str:
    """Convert a torch.dtype to its string representation."""
    return str(dtype)


def _str_to_dtype(s: str) -> torch.dtype:
    """Convert a string representation back to a torch.dtype."""
    return getattr(torch, s.split(".")[-1])


def _tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize a tensor to raw bytes."""
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def _bytes_to_tensor(
    data: bytes,
    shape: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Deserialize raw bytes to a tensor using torch.frombuffer."""
    return torch.frombuffer(bytearray(data), dtype=dtype).reshape(shape)


def _decode_meta(raw_meta: dict) -> dict[str, str]:
    """Decode a Redis hash response to a ``{str: str}`` dict."""
    return {
        k.decode() if isinstance(k, bytes) else k: (
            v.decode() if isinstance(v, bytes) else v
        )
        for k, v in raw_meta.items()
    }


def _normalize_index(idx: int, size: int) -> int:
    """Check an integer index and convert a valid negative index to an offset."""
    if not -size <= idx < size:
        raise IndexError(
            f"index {idx} is out of bounds for dimension 0 with size {size}"
        )
    return idx + size if idx < 0 else idx


def _normalize_non_tensor_index(idx):
    """Convert a row mask to positions before indexing a Python list."""
    if isinstance(idx, tuple) and len(idx) == 1:
        idx = idx[0]
    if isinstance(idx, list):
        idx = torch.tensor(idx)
    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            idx = idx.nonzero(as_tuple=False).squeeze(-1)
        idx = idx.reshape(-1).tolist()
    return idx


def _compute_byte_ranges(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> list[tuple[int, int]] | None:
    """Compute per-row ``(byte_offset, byte_length)`` pairs for the write path."""
    if isinstance(idx, tuple):
        if len(idx) == 1:
            idx = idx[0]
        else:
            return None
    if isinstance(idx, list):
        idx = torch.tensor(idx)

    if idx is Ellipsis:
        idx = slice(None)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    row_size = elem_size
    for s in shape[1:]:
        row_size *= s

    if isinstance(idx, int):
        pos = _normalize_index(idx, shape[0])
        return [(pos * row_size, row_size)]

    if isinstance(idx, slice):
        positions = range(*idx.indices(shape[0]))
        if len(positions) == 0:
            return []
        if positions.step == 1:
            return [(positions[0] * row_size, len(positions) * row_size)]
        return [(p * row_size, row_size) for p in positions]

    if isinstance(idx, range):
        return [(_normalize_index(int(p), shape[0]) * row_size, row_size) for p in idx]

    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            if idx.ndim == 1 and idx.shape[0] != shape[0]:
                raise IndexError(
                    f"The shape of the mask {list(idx.shape)} does not match "
                    f"dimension 0 with size {shape[0]}"
                )
            positions = idx.nonzero(as_tuple=False).squeeze(-1).tolist()
        else:
            positions = idx.reshape(-1).tolist()
        return [
            (_normalize_index(int(p), shape[0]) * row_size, row_size) for p in positions
        ]

    return None


def _compute_covering_range(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> tuple[int, int] | None:
    """Compute a single ``(byte_offset, byte_length)`` for the read path."""
    if isinstance(idx, tuple):
        idx = idx[0] if len(idx) == 1 else None
    if idx is None:
        return None
    if idx is Ellipsis:
        idx = slice(None)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    row_size = elem_size
    for s in shape[1:]:
        row_size *= s

    if isinstance(idx, int):
        pos = _normalize_index(idx, shape[0])
        return (pos * row_size, row_size)

    if isinstance(idx, slice):
        positions = range(*idx.indices(shape[0]))
        if len(positions) == 0:
            return (0, 0)
        start = positions[0]
        stop = positions[-1] + 1
        return (start * row_size, (stop - start) * row_size)

    return None


def _get_local_idx(idx, shape_0: int):
    """Return a local post-index to apply after fetching a covering range."""
    if isinstance(idx, tuple):
        idx = idx[0] if len(idx) == 1 else idx
    if idx is Ellipsis:
        return None
    if isinstance(idx, int):
        return None
    if isinstance(idx, slice):
        positions = range(*idx.indices(shape_0))
        if len(positions) == 0 or positions.step == 1:
            return None
        return slice(None, None, positions.step)
    return None


def _is_scattered_index(idx) -> bool:
    """Return True when *idx* selects rows with a tensor, list, or range."""
    if isinstance(idx, tuple):
        idx = idx[0] if len(idx) == 1 else idx
    if idx is Ellipsis:
        return False
    if isinstance(idx, (int, slice)):
        return False
    if isinstance(idx, (list, range, torch.Tensor)):
        return True
    return False


def _getitem_result_shape(
    shape: list[int],
    idx,
) -> list[int]:
    """Compute the result shape of ``tensor[idx]`` without creating a tensor."""
    if isinstance(idx, tuple):
        if len(idx) == 1:
            idx = idx[0]
        else:
            return list(torch.zeros(shape)[idx].shape)
    if isinstance(idx, list):
        idx = torch.tensor(idx)

    if idx is Ellipsis:
        idx = slice(None)

    rest = list(shape[1:])

    if isinstance(idx, int):
        return rest

    if isinstance(idx, slice):
        n = len(range(*idx.indices(shape[0])))
        return [n] + rest

    if isinstance(idx, range):
        return [len(idx)] + rest

    if isinstance(idx, torch.Tensor):
        if idx.dtype == torch.bool:
            n = int(idx.sum().item())
            return [n] + rest
        return list(idx.shape) + rest

    return list(torch.zeros(shape)[idx].shape)


def _prepare_indexed_value(
    value: torch.Tensor, shape: list[int], dtype: torch.dtype, idx
) -> torch.Tensor:
    """Match the selected shape and data type before converting values to bytes."""
    if isinstance(idx, tuple) and len(idx) == 1:
        idx = idx[0]
    if isinstance(idx, list):
        idx = torch.tensor(idx)
    if isinstance(idx, torch.Tensor) and idx.ndim == 0 and idx.dtype != torch.bool:
        idx = idx.item()
    # A boolean mask accepts one CPU value with a different data type.
    # Use masked_fill_ to keep PyTorch's overflow checks.
    is_masked_scalar = (
        isinstance(idx, torch.Tensor)
        and idx.dtype == torch.bool
        and value.numel() == 1
        and value.device.type == "cpu"
    )
    if _is_scattered_index(idx) and value.dtype != dtype and not is_masked_scalar:
        raise RuntimeError(
            "Index put requires the source and destination dtypes match, "
            f"got {dtype} for the destination and {value.dtype} for the source."
        )
    result_shape = _getitem_result_shape(shape, idx)
    if list(value.shape) == result_shape and value.dtype == dtype:
        return value
    result = torch.empty(result_shape, dtype=dtype, device=value.device)
    if is_masked_scalar:
        result.masked_fill_(torch.tensor(True, device=value.device), value.reshape(()))
    else:
        result[...] = value
    return result


for _name in __all__:
    _obj = globals()[_name]
    if callable(_obj):
        _obj.__module__ = "tensordict.store._store"
