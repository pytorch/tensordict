# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib

import pytest


def test_utils_key_json_import_paths_are_preserved():
    utils_module = importlib.import_module("tensordict.utils")
    helper_module = importlib.import_module("tensordict._utils_key_json")

    for name in helper_module.__all__:
        assert getattr(utils_module, name) is getattr(helper_module, name)
        assert getattr(utils_module, name).__module__ == "tensordict.utils"


def test_filesystem_key_roundtrip_and_warning():
    utils_module = importlib.import_module("tensordict.utils")

    key = "a/b% c"
    encoded = utils_module._encode_key_for_filesystem(key)

    assert encoded == "a%2Fb%25%20c"
    assert utils_module._decode_key_from_filesystem(encoded) == key
    assert utils_module._encode_key_for_filesystem(key, robust=False) == key
    with pytest.warns(FutureWarning):
        assert utils_module._get_robust_key_setting_with_warning(key, None) is False


def test_json_backend_roundtrip():
    utils_module = importlib.import_module("tensordict.utils")
    utils_module.set_json_backend("json")
    assert utils_module.get_json_backend().__name__ == "json"

    data = utils_module.json_dumps({"a": [1, 2]}, separators=(",", ":"))
    if isinstance(data, bytes):
        data = data.decode()
    assert data == '{"a":[1,2]}'
