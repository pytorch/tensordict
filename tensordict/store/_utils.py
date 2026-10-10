# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator

import torch
from tensordict._indexing import (
    _ELLIPSIS,
    _getitem_batch_size,
    _INDEX,
    _INT,
    _MASK,
    _read_element,
    _SLICE,
)

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
    "_non_tensor_positions",
    "_non_tensor_write_positions",
    "_normalize_index",
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


def _read_dim0(idx):
    """Read ``idx`` with :func:`~tensordict._indexing._read_element`, if it only selects along dim 0.

    Returns ``(kind, element)`` for an int, a slice, an ``Ellipsis`` (read as
    ``:``), an integer index or a 1-D mask. Returns ``None`` for the other
    indices, which the callers apply locally to the whole tensor: several
    elements, ``None``, a scalar bool or an N-D mask.
    """
    if isinstance(idx, tuple):
        if len(idx) != 1:
            return None
        (idx,) = idx
    kind, _, element = _read_element(idx)
    if kind == _ELLIPSIS:
        return _SLICE, slice(None)
    if kind in (_INT, _SLICE, _INDEX) or (kind == _MASK and element.ndim == 1):
        return kind, element
    return None


def _dim0_positions(kind, element, size: int) -> int | range | list[int]:
    """Return the positions that a :func:`_read_dim0` index selects along a dim of ``size`` elements.

    That is an int for an int, a ``range`` for a slice, and a list for an
    integer index, flattened, or for a mask.
    """
    if kind == _INT:
        return _normalize_index(operator.index(element), size)
    if kind == _SLICE:
        return range(*element.indices(size))
    if kind == _MASK:
        mask = torch.as_tensor(element)
        if mask.shape[0] != size:
            raise IndexError(
                f"The shape of the mask {list(mask.shape)} does not match "
                f"dimension 0 with size {size}"
            )
        return mask.nonzero().squeeze(-1).tolist()
    if not isinstance(element, (list, range)):
        element = element.reshape(-1).tolist()
    return [_normalize_index(int(p), size) for p in element]


def _non_tensor_positions(
    idx, size: int, flatten: bool = False
) -> int | list[int] | None:
    """Return the positions along dim 0 that *idx* selects in a ``json_array``.

    Returns an int for an integer index, which removes the dim, a list of
    positions for an index that keeps the dim, and ``None`` for any other index.
    With ``flatten=True``, an N-D integer index gives its flattened positions.
    """
    read = _read_dim0(idx)
    if read is None:
        return None
    kind, element = read
    if kind == _INDEX and not flatten and getattr(element, "ndim", 1) > 1:
        # an N-D integer index adds dims
        return None
    positions = _dim0_positions(kind, element, size)
    return list(positions) if kind == _SLICE else positions


def _non_tensor_write_positions(idx, size: int) -> int | list[int]:
    """Like :func:`_non_tensor_positions`, but raise for an unsupported index.

    A write only needs the positions, so an N-D integer index is flattened.
    """
    positions = _non_tensor_positions(idx, size, flatten=True)
    if positions is None:
        raise TypeError(
            f"Non-tensor indexed writes support indices along dim 0, got {idx!r}"
        )
    return positions


def _row_size(shape: list[int], dtype: torch.dtype) -> int:
    """The number of bytes of a row of dim 0."""
    row_size = torch.tensor([], dtype=dtype).element_size()
    for s in shape[1:]:
        row_size *= s
    return row_size


def _compute_byte_ranges(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> list[tuple[int, int]] | None:
    """Compute per-row ``(byte_offset, byte_length)`` pairs for the write path."""
    read = _read_dim0(idx)
    if read is None:
        # e.g. a 0-d or N-D mask, which does not select rows of dim 0: the
        # caller reads or writes the whole tensor and indexes it locally
        return None
    kind, element = read
    row_size = _row_size(shape, dtype)
    positions = _dim0_positions(kind, element, shape[0])
    if kind == _INT:
        return [(positions * row_size, row_size)]
    if kind == _SLICE:
        if len(positions) == 0:
            return []
        if positions.step == 1:
            return [(positions[0] * row_size, len(positions) * row_size)]
    return [(p * row_size, row_size) for p in positions]


def _compute_covering_range(
    shape: list[int],
    dtype: torch.dtype,
    idx,
) -> tuple[int, int] | None:
    """Compute a single ``(byte_offset, byte_length)`` for the read path."""
    read = _read_dim0(idx)
    if read is None or read[0] not in (_INT, _SLICE):
        return None
    kind, element = read
    row_size = _row_size(shape, dtype)
    positions = _dim0_positions(kind, element, shape[0])
    if kind == _INT:
        return (positions * row_size, row_size)
    if len(positions) == 0:
        return (0, 0)
    start = positions[0]
    stop = positions[-1] + 1
    return (start * row_size, (stop - start) * row_size)


def _get_local_idx(idx, shape_0: int):
    """Return a local post-index to apply after fetching a covering range."""
    read = _read_dim0(idx)
    if read is None or read[0] != _SLICE:
        return None
    positions = range(*read[1].indices(shape_0))
    if len(positions) == 0 or positions.step == 1:
        return None
    return slice(None, None, positions.step)


def _is_scattered_index(idx) -> bool:
    """Return True when *idx* selects rows of dim 0 with an integer index or a mask."""
    read = _read_dim0(idx)
    return read is not None and read[0] in (_INDEX, _MASK)


def _getitem_result_shape(
    shape: list[int],
    idx,
) -> list[int]:
    """Compute the result shape of ``tensor[idx]`` without creating a tensor."""
    return list(_getitem_batch_size(torch.Size(shape), idx))


def _prepare_indexed_value(
    value: torch.Tensor | float, shape: list[int], dtype: torch.dtype, idx
) -> torch.Tensor:
    """Match the selected shape and data type before converting values to bytes."""
    if not isinstance(value, torch.Tensor):
        # torch writes a Python scalar in the dtype of the entry
        value = torch.as_tensor(value, dtype=dtype)
    read = _read_dim0(idx)
    kind = None if read is None else read[0]
    # A boolean mask accepts one CPU value with a different data type.
    # Use masked_fill_ to keep PyTorch's overflow checks.
    is_masked_scalar = (
        kind == _MASK and value.numel() == 1 and value.device.type == "cpu"
    )
    if kind in (_INDEX, _MASK) and value.dtype != dtype and not is_masked_scalar:
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
