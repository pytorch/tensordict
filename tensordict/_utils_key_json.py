# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib
import json

__all__ = [
    "_decode_key_from_filesystem",
    "_encode_key_for_filesystem",
    "_get_robust_key_setting_with_warning",
    "_is_safe_legacy_key",
    "_json_dumps",
    "get_json_backend",
    "json_dumps",
    "set_json_backend",
]

_JSON_BACKENDS = ("json", "orjson")
# orjson writes NaN and infinite floats as null, which load back as None, so
# the standard json module writes the metadata unless set_json_backend("orjson")
# is called (gh-2913)
_JSON_BACKEND = "json"


_FS_UNSAFE_CHARS = set('/<>:"|?*\\ \0%')
_FS_UNSAFE_CHARS.update(chr(i) for i in range(32))
_FS_UNSAFE_CHARS.add(chr(127))
_FS_ENCODE_TABLE = str.maketrans({c: f"%{ord(c):02X}" for c in _FS_UNSAFE_CHARS})


def _encode_key_for_filesystem(key: str, *, robust: bool = True) -> str:
    """Encode a TensorDict key to be safe for filesystem paths."""
    if not robust:
        return key

    # These values do not name a child when used as a path component. A bare
    # "%" cannot be produced by the encoder (literal percent signs become
    # "%25"), so it is an unambiguous representation for the empty key.
    if not key:
        return "%"
    if key in (".", ".."):
        return "%2E" * len(key)

    return key.translate(_FS_ENCODE_TABLE)


def _is_safe_legacy_key(key: str, *, is_collection: bool = False) -> bool:
    """Return whether a raw legacy key stays beneath its memmap prefix."""
    if "/" in key or "\\" in key:
        return False
    if len(key) >= 2 and key[0].isalpha() and key[1] == ":":
        # A drive-relative path such as ``C:payload`` escapes a prefix on
        # Windows even though it has no slash.
        return False
    return not is_collection or key not in ("", ".", "..")


def _get_robust_key_setting_with_warning(key: str, robust_key) -> bool:
    """Handle the robust_key parameter after the robust default migration."""
    if robust_key is None:
        return True
    return robust_key


def _decode_key_from_filesystem(encoded_key: str) -> str:
    """Decode a filesystem-safe key back to the original TensorDict key."""
    if encoded_key == "%":
        return ""
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

    if _JSON_BACKEND == "orjson":
        orjson_kwargs = {k: v for k, v in kwargs.items() if k != "separators"}
        try:
            return backend.dumps(data, **orjson_kwargs)
        except TypeError as err:
            # orjson only takes ints that fit in 64 bits: json writes the others
            if "Integer exceeds 64-bit range" not in str(err):
                raise
            backend = json
    return backend.dumps(data, **kwargs)


def json_dumps(data, **kwargs):
    """Serializes ``data`` to JSON with the backend that :func:`set_json_backend` selects.

    Returns a str with ``json``, and bytes with ``orjson``. The ``separators``
    argument is ignored by ``orjson``.
    """
    return _json_dumps(data, **kwargs)


def set_json_backend(backend):
    """Sets the JSON backend that tensordict uses: ``"json"`` or ``"orjson"``.

    The default is ``"json"``. ``orjson`` writes NaN and infinite floats in
    non-tensor data as ``null``, so they load back as ``None``.
    """
    global _JSON_BACKEND

    if backend not in _JSON_BACKENDS:
        raise ValueError("Backend must be either 'json' or 'orjson'")
    importlib.import_module(backend)
    _JSON_BACKEND = backend


def get_json_backend():
    """Returns the module of the current JSON backend (``json`` or ``orjson``)."""
    return importlib.import_module(_JSON_BACKEND)
