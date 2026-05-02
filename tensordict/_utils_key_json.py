# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import importlib.util
import warnings

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

_JSON_BACKENDS = ("json", "orjson")
_JSON_BACKEND = "orjson" if importlib.util.find_spec("orjson") is not None else "json"


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


def _json_dumps(data, **kwargs):
    """JSON serialization using the configured json backend."""
    backend = get_json_backend()

    if _JSON_BACKEND == "orjson" and "separators" in kwargs:
        kwargs.pop("separators")
    return backend.dumps(data, **kwargs)


def json_dumps(data, **kwargs):
    """Unified JSON serialization function that works with both json and orjson backends."""
    return _json_dumps(data, **kwargs)


def set_json_backend(backend):
    """Set the JSON backend to use (either 'json' or 'orjson')."""
    global _JSON_BACKEND

    if backend not in _JSON_BACKENDS:
        raise ValueError("Backend must be either 'json' or 'orjson'")
    importlib.import_module(backend)
    _JSON_BACKEND = backend


def get_json_backend():
    """Get the current JSON backend."""
    return importlib.import_module(_JSON_BACKEND)


for _name in __all__:
    globals()[_name].__module__ = "tensordict.utils"
