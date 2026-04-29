# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import warnings

from pyvers import get_backend, implement_for, register_backend, set_backend

__all__ = [
    "_decode_key_from_filesystem",
    "_encode_key_for_filesystem",
    "_get_robust_key_setting",
    "_get_robust_key_setting_with_warning",
    "_json_dumps",
    "get_json_backend",
    "json_dumps",
    "set_json_backend",
]


def _encode_key_for_filesystem(key: str, *, robust: bool = True) -> str:
    """Encode a TensorDict key to be safe for filesystem paths."""
    if not robust:
        return key

    unsafe_chars = set('/<>:"|?*\\ \0%')
    unsafe_chars.update(chr(i) for i in range(32))
    unsafe_chars.add(chr(127))

    encoded_parts = []
    for char in key:
        if char in unsafe_chars:
            encoded_parts.append(f"%{ord(char):02X}")
        else:
            encoded_parts.append(char)

    return "".join(encoded_parts)


def _get_robust_key_setting_with_warning(key: str, robust_key) -> bool:
    """Handle the robust_key parameter with smart deprecation warning."""
    if robust_key is not None:
        return robust_key

    robust_encoded = _encode_key_for_filesystem(key, robust=True)
    legacy_encoded = _encode_key_for_filesystem(key, robust=False)

    if robust_encoded != legacy_encoded:
        warnings.warn(
            f"The key '{key}' contains characters that will be handled differently "
            f"in TensorDict v0.12 for better cross-platform support. "
            f"To opt into the new behavior now, use `robust_key=True`. "
            f"To suppress this warning and keep the current behavior, use `robust_key=False`. "
            f"See https://github.com/pytorch/tensordict/issues/1440 for details.",
            FutureWarning,
            stacklevel=3,
        )

    return False


def _get_robust_key_setting(robust_key) -> bool:
    """Handle the robust_key parameter without key-specific logic."""
    if robust_key is None:
        return False
    return robust_key


def _decode_key_from_filesystem(encoded_key: str) -> str:
    """Decode a filesystem-safe key back to the original TensorDict key."""
    decoded_parts = []
    i = 0
    while i < len(encoded_key):
        if encoded_key[i] == "%" and i + 2 < len(encoded_key):
            try:
                hex_str = encoded_key[i + 1 : i + 3]
                char_code = int(hex_str, 16)
                decoded_parts.append(chr(char_code))
                i += 3
            except ValueError:
                decoded_parts.append(encoded_key[i])
                i += 1
        else:
            decoded_parts.append(encoded_key[i])
            i += 1

    return "".join(decoded_parts)


register_backend(group="json", backends={"json": "json", "orjson": "orjson"})


@implement_for("json")
def _json_dumps(data, **kwargs):
    """JSON serialization using standard json module."""
    import json

    return json.dumps(data, **kwargs)


@implement_for("orjson")
def _json_dumps(data, **kwargs):  # noqa: F811
    """JSON serialization using orjson module."""
    import orjson

    if "separators" in kwargs:
        kwargs.pop("separators")
    return orjson.dumps(data, **kwargs)


def json_dumps(data, **kwargs):
    """Unified JSON serialization function that works with both json and orjson backends."""
    return _json_dumps(data, **kwargs)


def set_json_backend(backend):
    """Set the JSON backend to use (either 'json' or 'orjson')."""
    if backend not in ["json", "orjson"]:
        raise ValueError("Backend must be either 'json' or 'orjson'")
    set_backend("json", backend)


def get_json_backend():
    """Get the current JSON backend."""
    return get_backend("json")


if importlib.util.find_spec("orjson") is not None:
    set_json_backend("orjson")
else:
    set_json_backend("json")


for _name in __all__:
    globals()[_name].__module__ = "tensordict.utils"
