# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import os
import subprocess
import sys
import warnings

import pytest


def test_utils_options_import_paths_are_preserved():
    utils_module = importlib.import_module("tensordict.utils")
    options_module = importlib.import_module("tensordict._utils_options")

    for name in options_module.__all__:
        assert getattr(utils_module, name) is getattr(options_module, name)
        obj = getattr(utils_module, name)
        if hasattr(obj, "__module__"):
            assert obj.__module__ == "tensordict.utils"


def test_printoptions_share_state():
    utils_module = importlib.import_module("tensordict.utils")
    options_module = importlib.import_module("tensordict._utils_options")

    before = utils_module.get_printoptions()
    with utils_module.set_printoptions(show_device=False):
        assert utils_module.get_printoptions()["show_device"] is False
        assert options_module.get_printoptions()["show_device"] is False
        assert utils_module._REPR_OPTIONS is options_module._REPR_OPTIONS
    assert utils_module.get_printoptions() == before


def test_context_options_share_state():
    utils_module = importlib.import_module("tensordict.utils")
    options_module = importlib.import_module("tensordict._utils_options")

    with pytest.warns(DeprecationWarning, match="legacy lazy mode"):
        mode = utils_module.set_lazy_legacy(True)
    with mode:
        assert utils_module.lazy_legacy()
        assert options_module.lazy_legacy()
    with utils_module.set_capture_non_tensor_stack(True):
        assert utils_module.capture_non_tensor_stack()
        assert options_module.capture_non_tensor_stack()
    with utils_module.set_list_to_stack(False):
        assert not utils_module.list_to_stack()
        assert not options_module.list_to_stack()


def test_set_lazy_legacy_is_deprecated():
    from tensordict import lazy_legacy, set_lazy_legacy

    with pytest.warns(DeprecationWarning, match="removed in TensorDict 0.17"):
        decorator = set_lazy_legacy(True)

    @decorator
    def legacy():
        return lazy_legacy()

    with warnings.catch_warnings():
        # only the decorator warns, not the calls, and turning the mode off
        # does not warn
        warnings.simplefilter("error")
        assert legacy()
        with set_lazy_legacy(False):
            assert not lazy_legacy()


def test_lazy_legacy_env_var_is_deprecated():
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import tensordict\n"
        "print(any('legacy lazy mode' in str(w.message) for w in caught))\n"
    )
    env = {**os.environ, "LAZY_LEGACY_OP": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.stdout.strip() == "True", result.stderr
