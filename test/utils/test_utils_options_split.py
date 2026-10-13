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
    with pytest.warns(DeprecationWarning, match="removed in TensorDict 0.17"):
        mode = utils_module.set_capture_non_tensor_stack(True)
    with mode:
        assert utils_module.capture_non_tensor_stack()
        assert options_module.capture_non_tensor_stack()
    with pytest.warns(DeprecationWarning, match="removed in TensorDict 0.17"):
        mode = utils_module.set_list_to_stack(False)
    with mode:
        assert not utils_module.list_to_stack()
        assert not options_module.list_to_stack()


def test_list_to_stack_returns_bool(monkeypatch):
    from tensordict import _utils_options, list_to_stack, set_list_to_stack

    # CI sets LIST_TO_STACK=1: start from an unset option
    monkeypatch.delenv("LIST_TO_STACK", raising=False)
    monkeypatch.setattr(_utils_options, "_LIST_TO_STACK", None)
    assert list_to_stack() is True
    with set_list_to_stack(True):
        assert list_to_stack() is True
    assert list_to_stack() is True
    assert list_to_stack(allow_none=True) is None


@pytest.mark.parametrize("value", ["1", "0"])
def test_option_env_vars_give_bool(value):
    code = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "from tensordict import capture_non_tensor_stack, lazy_legacy, list_to_stack\n"
        "print(repr((list_to_stack(), capture_non_tensor_stack(), lazy_legacy())))\n"
    )
    env = {
        **os.environ,
        "LIST_TO_STACK": value,
        "CAPTURE_NONTENSOR_STACK": value,
        "LAZY_LEGACY_OP": value,
    }
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.stdout.strip() == repr((value == "1",) * 3), result.stderr


@pytest.mark.parametrize(
    "context,env_var,global_name",
    [
        ("_set_list_to_stack", "LIST_TO_STACK", "_LIST_TO_STACK"),
        (
            "_set_capture_non_tensor_stack",
            "CAPTURE_NONTENSOR_STACK",
            "_CAPTURE_NONTENSOR_STACK",
        ),
        ("set_lazy_legacy", "LAZY_LEGACY_OP", "_LAZY_OP"),
    ],
)
def test_option_block_restores_env(monkeypatch, context, env_var, global_name):
    from tensordict import _utils_options

    monkeypatch.delenv(env_var, raising=False)
    monkeypatch.setattr(_utils_options, global_name, None)
    with getattr(_utils_options, context)(False):
        assert os.environ[env_var] == "False"
    assert env_var not in os.environ

    # an option read from the environment at import keeps its value
    monkeypatch.setenv(env_var, "0")
    monkeypatch.setattr(_utils_options, global_name, "0")
    with getattr(_utils_options, context)(False):
        assert os.environ[env_var] == "False"
    assert os.environ[env_var] == "0"


def test_lazy_legacy_block_keeps_child_process_working(monkeypatch):
    from tensordict import _utils_options, set_lazy_legacy

    monkeypatch.delenv("LAZY_LEGACY_OP", raising=False)
    monkeypatch.setattr(_utils_options, "_LAZY_OP", None)
    with set_lazy_legacy(False):
        pass
    code = (
        "from tensordict import TensorDict, lazy_legacy\n"
        "TensorDict(batch_size=[2, 3]).permute(1, 0)\n"
        "print(lazy_legacy())\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


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


def _import_warnings(env_var, value):
    # Imports tensordict in a subprocess with ``env_var=value`` and returns the
    # messages of the FutureWarnings that the import emits.
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import tensordict\n"
        "for w in caught:\n"
        "    if issubclass(w.category, FutureWarning):\n"
        "        print(w.message)\n"
    )
    env = {**os.environ, env_var: value}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_set_capture_non_tensor_stack_true_is_deprecated():
    from tensordict import capture_non_tensor_stack, set_capture_non_tensor_stack

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        decorator = set_capture_non_tensor_stack(True)
    assert len(record) == 1
    assert record[0].category is DeprecationWarning
    assert str(record[0].message) == (
        "set_capture_non_tensor_stack(True) is deprecated and will be removed in "
        "TensorDict 0.17. Use NonTensorStack.data to get the single value of a "
        "stack of identical values instead."
    )
    assert record[0].filename == __file__

    @decorator
    def capture():
        return capture_non_tensor_stack()

    with warnings.catch_warnings():
        # only the decorator warns, not the calls, and False does not warn
        warnings.simplefilter("error")
        assert capture()
        assert capture()
        with set_capture_non_tensor_stack(False):
            assert not capture_non_tensor_stack()
    assert not capture_non_tensor_stack()


def test_set_list_to_stack_false_is_deprecated():
    from tensordict import list_to_stack, set_list_to_stack

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        decorator = set_list_to_stack(False)
    assert len(record) == 1
    assert record[0].category is DeprecationWarning
    assert str(record[0].message) == (
        "set_list_to_stack(False) is deprecated and will be removed in "
        "TensorDict 0.17. Use td.set_non_tensor(key, value) to store a list as "
        "one value instead."
    )
    assert record[0].filename == __file__

    @decorator
    def no_list_to_stack():
        return list_to_stack()

    with warnings.catch_warnings():
        # only the decorator warns, not the calls, and True does not warn
        warnings.simplefilter("error")
        assert not no_list_to_stack()
        assert not no_list_to_stack()
        with set_list_to_stack(True):
            assert list_to_stack()
        list_to_stack()


@pytest.mark.parametrize("value", ["1", "True"])
def test_capture_non_tensor_stack_env_var_true_is_deprecated(value):
    assert (
        f"CAPTURE_NONTENSOR_STACK={value} is deprecated and will be removed in "
        "TensorDict 0.17." in _import_warnings("CAPTURE_NONTENSOR_STACK", value)
    )


@pytest.mark.parametrize("value", ["0", "False"])
def test_list_to_stack_env_var_false_is_deprecated(value):
    assert (
        f"LIST_TO_STACK={value} is deprecated and will be removed in "
        "TensorDict 0.17." in _import_warnings("LIST_TO_STACK", value)
    )


@pytest.mark.parametrize("value", ["1", "True"])
def test_exclude_td_from_pytree_env_var_true_is_deprecated(value):
    assert (
        f"EXCLUDE_TD_FROM_PYTREE={value} is deprecated and will be removed in "
        "TensorDict 0.17." in _import_warnings("EXCLUDE_TD_FROM_PYTREE", value)
    )


@pytest.mark.parametrize("value,registered", [("1", False), ("0", True)])
def test_exclude_td_from_pytree_env_var_keeps_its_effect(value, registered):
    # Until its removal, EXCLUDE_TD_FROM_PYTREE=1 still removes tensordicts
    # from the pytree registry at import.
    code = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import tensordict\n"
        "from torch.utils._pytree import SUPPORTED_NODES\n"
        "print(tensordict.TensorDict in SUPPORTED_NODES)\n"
    )
    env = {**os.environ, "EXCLUDE_TD_FROM_PYTREE": value}
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(registered)


@pytest.mark.parametrize(
    "env_var,value,message",
    [
        ("TD_GET_DEFAULTS_TO_NONE", "0", "TD_GET_DEFAULTS_TO_NONE=0 is deprecated"),
        ("LIST_TO_STACK", "0", "LIST_TO_STACK=0 is deprecated"),
        ("CAPTURE_NONTENSOR_STACK", "1", "CAPTURE_NONTENSOR_STACK=1 is deprecated"),
        ("LAZY_LEGACY_OP", "1", "The legacy lazy mode"),
        ("EXCLUDE_TD_FROM_PYTREE", "1", "EXCLUDE_TD_FROM_PYTREE=1 is deprecated"),
    ],
)
def test_deprecated_env_vars_warn_with_default_filters(env_var, value, message):
    # Python's default filters hide a DeprecationWarning raised inside
    # tensordict at import time, so these warnings are FutureWarnings.
    env = {
        key: val
        for key, val in os.environ.items()
        if key not in ("PYTHONWARNINGS", "PYTHONDEVMODE")
    }
    env[env_var] = value
    result = subprocess.run(
        [sys.executable, "-c", "import tensordict"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert f"FutureWarning: {message}" in result.stderr


@pytest.mark.parametrize(
    "env_var,value",
    [
        ("CAPTURE_NONTENSOR_STACK", "0"),
        ("LIST_TO_STACK", "1"),
        ("EXCLUDE_TD_FROM_PYTREE", "0"),
    ],
)
def test_default_option_env_vars_do_not_warn(env_var, value):
    assert env_var not in _import_warnings(env_var, value)
