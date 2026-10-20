# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
from pathlib import Path

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import tempfile
import warnings

import pytest
import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    PersistentTensorDict,
    tensorclass,
    TensorDict,
    TypedTensorDict,
)
from tensordict.base import _default_is_leaf, TensorDictBase
from torch import Tensor

_has_mypy = importlib.util.find_spec("mypy") is not None
_has_h5py = importlib.util.find_spec("h5py") is not None

# ---------------------------------------------------------------------------
# Fixture classes
# ---------------------------------------------------------------------------


class PredictorState(TypedTensorDict):
    eta: Tensor
    X: Tensor
    beta: Tensor


class ObservedState(PredictorState):
    y: Tensor
    mu: Tensor
    noise: NotRequired[Tensor]


class PortableOptionalState(PredictorState):
    noise: Tensor | None = None


class SurvivalState(ObservedState):
    event_time: Tensor
    indicator: Tensor
    observed_time: Tensor


class ShadowClass(TypedTensorDict["shadow"]):
    data: Tensor
    clone: Tensor


class FrozenClass(TypedTensorDict["frozen"]):
    x: Tensor


class ShadowFrozenClass(TypedTensorDict["shadow", "frozen"]):
    data: Tensor


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic_construction(self):
        state = PredictorState(
            eta=torch.randn(5, 3),
            X=torch.randn(5, 4),
            beta=torch.randn(5, 1),
            batch_size=[5],
        )
        assert isinstance(state, PredictorState)
        assert isinstance(state, TypedTensorDict)
        assert isinstance(state, TensorDictBase)
        assert state.batch_size == torch.Size([5])

    def test_from_dict_and_instance(self):
        data = {
            "eta": torch.zeros(4, 3),
            "X": torch.ones(4, 3),
            "beta": torch.zeros(4, 3),
        }
        state = PredictorState.from_dict(data, batch_size=[4, 3], names=["n", "t"])
        assert type(state) is PredictorState
        assert state.batch_size == (4, 3)
        assert state.names == ["n", "t"]
        assert (state.X == 1).all()
        other = state.from_dict_instance(
            {**data, "X": torch.full((4, 3), 2.0)}, batch_size=[4]
        )
        assert type(other) is PredictorState
        assert other.batch_size == (4,)
        assert (other.X == 2).all()
        assert (state.X == 1).all()
        # a missing required field warns (it raises from 0.17) and gives an
        # instance without it; check=False gives the same instance silently
        with pytest.warns(FutureWarning, match="misses the required field.*X, beta"):
            partial = PredictorState.from_dict({"eta": torch.zeros(4)}, batch_size=[4])
        assert type(partial) is PredictorState
        assert list(partial.keys()) == ["eta"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            partial = PredictorState.from_dict(
                {"eta": torch.zeros(4)}, batch_size=[4], check=False
            )
            assert list(partial.keys()) == ["eta"]
            # undeclared keys are kept without a warning
            extra = PredictorState.from_dict({**data, "z": torch.zeros(4)})
            assert sorted(extra.keys()) == ["X", "beta", "eta", "z"]
            # a tensor collection that is not a TensorDict is wrapped as is
            stacked = PredictorState.from_dict(lazy_stack([state[0], state[1]]))
            assert type(stacked) is PredictorState
            assert sorted(stacked.keys()) == sorted(data)
            # from_dict_instance does not check the declared fields
            other = state.from_dict_instance({"eta": torch.zeros(4)}, batch_size=[4])
            assert list(other.keys()) == ["eta"]
            # a nested partial write goes through from_dict_instance
            outer = TensorDict(inner=state.clone(), batch_size=[4])
            outer[1:3] = {"inner": {"X": torch.full((2, 3), 3.0)}}
            assert type(outer["inner"]) is PredictorState
            assert (outer["inner", "X"][1:3] == 3).all()
            assert (outer["inner", "X"][::3] == 1).all()
        # the positional constructor warns the same way
        with pytest.warns(FutureWarning, match=r"PredictorState\(dict\).*X, beta"):
            partial = PredictorState({"eta": torch.zeros(4)}, batch_size=[4])
        assert list(partial.keys()) == ["eta"]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert sorted(PredictorState(data, batch_size=[4]).keys()) == sorted(data)

    def test_missing_required_field(self):
        with pytest.raises(TypeError, match="missing required field.*beta"):
            PredictorState(
                eta=torch.randn(5, 3),
                X=torch.randn(5, 4),
                batch_size=[5],
            )

    def test_extra_field(self):
        with pytest.raises(TypeError, match="unexpected field.*z"):
            PredictorState(
                eta=torch.randn(5, 3),
                X=torch.randn(5, 4),
                beta=torch.randn(5, 1),
                z=torch.randn(5),
                batch_size=[5],
            )

    def test_device(self):
        state = PredictorState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            batch_size=[3],
            device="cpu",
        )
        assert state.device == torch.device("cpu")

    def test_names(self):
        state = PredictorState(
            eta=torch.randn(3, 2),
            X=torch.randn(3, 2),
            beta=torch.randn(3, 2),
            batch_size=[3, 2],
            names=["batch", "time"],
        )
        assert state.names == ["batch", "time"]

    def test_multidim_batch(self):
        N, T = 4, 6
        state = PredictorState(
            eta=torch.randn(N, T, 1),
            X=torch.randn(N, T, 3),
            beta=torch.randn(N, T, 1),
            batch_size=[N, T],
        )
        assert state.batch_size == torch.Size([N, T])

    def test_default_field_is_optional(self):
        state = PortableOptionalState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            batch_size=[3],
        )
        assert state.noise is None
        assert "noise" not in state
        assert PortableOptionalState.__optional_keys__ == frozenset({"noise"})

    def test_default_field_can_be_set(self):
        noise = torch.randn(3)
        state = PortableOptionalState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            noise=noise,
            batch_size=[3],
        )
        assert state.noise is noise

    @pytest.mark.skipif(not _has_mypy, reason="mypy is not installed")
    def test_static_field_contract(self, tmp_path):
        from mypy import api

        stubs = tmp_path / "stubs"
        package = stubs / "tensordict"
        package.mkdir(parents=True)
        torch_package = stubs / "torch"
        torch_package.mkdir()
        torch_package.joinpath("__init__.pyi").write_text(
            """class Size: ...
class dtype: ...
"""
        )
        package.joinpath("__init__.pyi").write_text(
            "from .typedtensordict import TypedTensorDict as TypedTensorDict\n"
        )
        package.joinpath("base.pyi").write_text("class TensorDictBase: ...\n")
        package.joinpath("_td.pyi").write_text(
            """from typing import Any

from tensordict.base import TensorDictBase

class TensorDict(TensorDictBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
"""
        )
        package.joinpath("utils.pyi").write_text(
            """from typing import Any, TypeAlias

DeviceType: TypeAlias = Any
"""
        )
        typed_stub = Path(__file__).parents[2] / "tensordict" / "typedtensordict.pyi"
        package.joinpath("typedtensordict.pyi").write_text(typed_stub.read_text())

        config = tmp_path / "mypy.ini"
        config.write_text(
            f"""[mypy]
python_version = 3.10
show_error_codes = True
mypy_path = {stubs}
ignore_missing_imports = True
"""
        )

        valid = tmp_path / "valid.py"
        valid.write_text(
            """from typing_extensions import assert_type
from tensordict import TypedTensorDict
from tensordict._td import TensorDict

class State(TypedTensorDict):
    observation: int
    reward: int
    optional: int | None = None

class ChildState(State):
    done: int

def consume(state: State) -> int:
    return state.observation + state.reward

state = State(observation=1, reward=2, batch_size=[], device="cpu")
child = ChildState(observation=1, reward=2, done=0)
ChildState(
    observation=1, reward=2, done=0, names=["n"], non_blocking=False, lock=True
)
assert_type(state.observation, int)
assert_type(state.optional, int | None)
assert_type(consume(child), int)
assert_type(State.from_tensordict(TensorDict(), check=False), State)
"""
        )
        stdout, stderr, status = api.run(["--config-file", str(config), str(valid)])
        assert status == 0, stdout + stderr

        invalid = tmp_path / "invalid.py"
        invalid.write_text(
            """from tensordict import TypedTensorDict

class State(TypedTensorDict):
    observation: int
    reward: int

State(observation=1)
State(observation="bad", reward=2)
State(observation=1, reward=2, extra=3)
state = State(observation=1, reward=2)
state.missing
"""
        )
        stdout, stderr, status = api.run(["--config-file", str(config), str(invalid)])
        assert status == 1, stdout + stderr
        assert 'Missing named argument "reward"' in stdout
        assert 'incompatible type "str"' in stdout
        assert 'Unexpected keyword argument "extra"' in stdout
        assert 'has no attribute "missing"' in stdout


# ---------------------------------------------------------------------------
# Field access
# ---------------------------------------------------------------------------


class TestFieldAccess:
    def test_attribute_access(self):
        state = PredictorState(
            eta=torch.randn(3, 2),
            X=torch.randn(3, 4),
            beta=torch.randn(3, 1),
            batch_size=[3],
        )
        assert state.eta.shape == (3, 2)
        assert state.X.shape == (3, 4)
        assert state.beta.shape == (3, 1)

    def test_string_key_access(self):
        state = PredictorState(
            eta=torch.randn(3, 2),
            X=torch.randn(3, 4),
            beta=torch.randn(3, 1),
            batch_size=[3],
        )
        assert state["eta"].shape == (3, 2)
        assert state["X"].shape == (3, 4)
        assert state["beta"].shape == (3, 1)

    def test_attribute_set(self):
        state = PredictorState(
            eta=torch.randn(3, 2),
            X=torch.randn(3, 4),
            beta=torch.randn(3, 1),
            batch_size=[3],
        )
        new_eta = torch.ones(3, 2)
        state.eta = new_eta
        assert (state.eta == new_eta).all()

    def test_string_key_set(self):
        state = PredictorState(
            eta=torch.randn(3, 2),
            X=torch.randn(3, 4),
            beta=torch.randn(3, 1),
            batch_size=[3],
        )
        new_eta = torch.ones(3, 2)
        state["eta"] = new_eta
        assert (state["eta"] == new_eta).all()

    def test_attribute_error_undeclared(self):
        state = PredictorState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            batch_size=[3],
        )
        with pytest.raises(AttributeError):
            _ = state.nonexistent

    def test_keys(self):
        state = PredictorState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            batch_size=[3],
        )
        assert set(state.keys()) == {"eta", "X", "beta"}

    @pytest.mark.parametrize("backend", ["dense", "lazy", "memmap"])
    def test_nested_keys_with_leaf_predicate(self, backend, tmp_path):
        source = TensorDict(
            {"eta": torch.ones(3), "X": torch.zeros(3), "beta": torch.ones(3)}, [3]
        )
        if backend == "lazy":
            source = lazy_stack([source, source], 0)
        elif backend == "memmap":
            source = source.memmap(tmp_path)
        state = PredictorState.from_tensordict(source)
        outer = TensorDict({"state": state}, source.batch_size)
        keys = set(outer.keys(True, True, is_leaf=_default_is_leaf))
        assert keys == {("state", "eta"), ("state", "X"), ("state", "beta")}
        assert all(torch.equal(outer.get(key), source.get(key[-1])) for key in keys)


# ---------------------------------------------------------------------------
# Spreading (**state)
# ---------------------------------------------------------------------------


class TestSpreading:
    def test_spread_to_dict(self):
        state = PredictorState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            batch_size=[3],
        )
        d = dict(**state)
        assert set(d.keys()) == {"eta", "X", "beta"}
        for v in d.values():
            assert isinstance(v, torch.Tensor)

    def test_spread_to_child(self):
        state = PredictorState(
            eta=torch.randn(5, 3),
            X=torch.randn(5, 4),
            beta=torch.randn(5, 1),
            batch_size=[5],
        )
        obs = ObservedState(
            **state,
            y=torch.randn(5, 3),
            mu=torch.randn(5, 3),
            batch_size=[5],
        )
        assert isinstance(obs, ObservedState)
        assert isinstance(obs, PredictorState)
        assert set(obs.keys()) == {"eta", "X", "beta", "y", "mu"}

    def test_pipeline_transition(self):
        """Simulate the gaussian() pipeline function from the feature request."""
        state = PredictorState(
            eta=torch.randn(5, 3),
            X=torch.randn(5, 4),
            beta=torch.randn(5, 1),
            batch_size=[5],
        )
        eta = state.eta
        y = eta + torch.randn_like(eta) * 0.1
        obs = ObservedState(**state, y=y, mu=eta, batch_size=state.batch_size)
        assert obs.y.shape == eta.shape
        assert (obs.eta == eta).all()


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_isinstance(self):
        obs = ObservedState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            y=torch.randn(3),
            mu=torch.randn(3),
            batch_size=[3],
        )
        assert isinstance(obs, ObservedState)
        assert isinstance(obs, PredictorState)
        assert isinstance(obs, TypedTensorDict)
        assert isinstance(obs, TensorDictBase)

    def test_field_accumulation(self):
        assert PredictorState.__expected_keys__ == frozenset({"eta", "X", "beta"})
        assert ObservedState.__expected_keys__ == frozenset(
            {"eta", "X", "beta", "y", "mu", "noise"}
        )
        assert SurvivalState.__expected_keys__ == frozenset(
            {
                "eta",
                "X",
                "beta",
                "y",
                "mu",
                "noise",
                "event_time",
                "indicator",
                "observed_time",
            }
        )

    def test_required_vs_optional(self):
        assert PredictorState.__required_keys__ == frozenset({"eta", "X", "beta"})
        assert PredictorState.__optional_keys__ == frozenset()
        assert ObservedState.__required_keys__ == frozenset(
            {"eta", "X", "beta", "y", "mu"}
        )
        assert ObservedState.__optional_keys__ == frozenset({"noise"})

    def test_three_level_inheritance(self):
        surv = SurvivalState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            y=torch.randn(3),
            mu=torch.randn(3),
            event_time=torch.randn(3),
            indicator=torch.randn(3),
            observed_time=torch.randn(3),
            batch_size=[3],
        )
        assert isinstance(surv, SurvivalState)
        assert isinstance(surv, ObservedState)
        assert isinstance(surv, PredictorState)
        assert len(surv.keys()) == 8  # noise not set


# ---------------------------------------------------------------------------
# NotRequired
# ---------------------------------------------------------------------------


class TestNotRequired:
    def test_optional_omitted(self):
        obs = ObservedState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            y=torch.randn(3),
            mu=torch.randn(3),
            batch_size=[3],
        )
        assert "noise" not in obs
        assert set(obs.keys()) == {"eta", "X", "beta", "y", "mu"}

    def test_optional_provided(self):
        obs = ObservedState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            y=torch.randn(3),
            mu=torch.randn(3),
            noise=torch.randn(3),
            batch_size=[3],
        )
        assert "noise" in obs
        assert obs.noise.shape == (3,)

    def test_access_missing_optional(self):
        obs = ObservedState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            y=torch.randn(3),
            mu=torch.randn(3),
            batch_size=[3],
        )
        with pytest.raises(AttributeError, match="declared but not set"):
            _ = obs.noise


# ---------------------------------------------------------------------------
# TensorDict operations
# ---------------------------------------------------------------------------


class TestTensorDictOps:
    @pytest.fixture
    def state(self):
        return PredictorState(
            eta=torch.randn(5, 3),
            X=torch.randn(5, 4),
            beta=torch.randn(5, 1),
            batch_size=[5],
        )

    def test_clone(self, state):
        clone = state.clone()
        assert set(clone.keys()) == set(state.keys())
        assert (clone["eta"] == state["eta"]).all()

    def test_slice(self, state):
        sliced = state[0:3]
        assert sliced.batch_size == torch.Size([3])
        assert sliced["eta"].shape[0] == 3

    def test_index(self, state):
        item = state[0]
        assert item.batch_size == torch.Size([])
        assert item["eta"].shape == (3,)

    def test_to_device(self, state):
        cpu_state = state.to("cpu")
        assert cpu_state.device == torch.device("cpu")

    def test_contiguous(self, state):
        c = state.contiguous()
        assert set(c.keys()) == set(state.keys())

    def test_stack(self):
        states = [
            PredictorState(
                eta=torch.randn(3),
                X=torch.randn(3),
                beta=torch.randn(3),
                batch_size=[3],
            )
            for _ in range(4)
        ]
        stacked = torch.stack(states, dim=0)
        assert stacked.batch_size == torch.Size([4, 3])

    def test_cat(self):
        states = [
            PredictorState(
                eta=torch.randn(3, 2),
                X=torch.randn(3, 2),
                beta=torch.randn(3, 2),
                batch_size=[3],
            )
            for _ in range(4)
        ]
        catted = torch.cat(states, dim=0)
        assert catted.batch_size == torch.Size([12])

    def test_unbind(self, state):
        items = state.unbind(0)
        assert len(items) == 5

    def test_apply(self, state):
        result = state.apply(lambda x: x * 2)
        assert (result["eta"] == state["eta"] * 2).all()

    def test_update(self, state):
        state.update({"eta": torch.ones(5, 3)})
        assert (state["eta"] == 1).all()

    def test_set_returns_self(self, state):
        assert state.set("new", torch.ones(5)) is state
        assert state.set(("nested", "a"), torch.ones(5)) is state
        assert state.set_("eta", torch.ones(5, 3)) is state
        assert state.set_(("nested", "a"), torch.zeros(5)) is state
        assert state.set_at_("eta", torch.full((3,), 2.0), 0) is state
        assert state.set_at_(("nested", "a"), torch.tensor(2.0), 0) is state
        assert (state["eta"][0] == 2).all()
        assert state["nested", "a"][0] == 2

    @pytest.mark.parametrize(
        "op",
        [
            lambda s: s.reshape(5, 1, inplace=True),
            lambda s: s.masked_select(torch.ones(5, dtype=torch.bool), inplace=True),
            lambda s: s.repeat_interleave(2, dim=0, inplace=True),
            lambda s: s.flatten(inplace=True),
            lambda s: s.repeat(2, inplace=True),
        ],
        ids=["reshape", "masked_select", "repeat_interleave", "flatten", "repeat"],
    )
    def test_shape_op_inplace_returns_self(self, state, op):
        # inplace=True keeps the object identity, as on a TensorDict
        expected = op(state.to_tensordict()).batch_size
        assert op(state) is state
        assert state.batch_size == expected
        assert isinstance(state, PredictorState)

    def test_to_dict(self, state):
        d = state.to_dict()
        assert set(d.keys()) == {"eta", "X", "beta"}

    def test_select(self, state):
        sel = state.select("eta", "X")
        assert set(sel.keys()) == {"eta", "X"}

    def test_exclude(self, state):
        exc = state.exclude("beta")
        assert set(exc.keys()) == {"eta", "X"}

    def test_is_locked_setter_erases_cache(self, state):
        # ``is_locked = False`` must drop the results cached while locked,
        # as ``unlock_()`` does.
        state.is_locked = True
        state.detach()
        state.is_locked = False
        assert state._cache is None
        state.set("eta", torch.ones(5, 3))
        state.is_locked = True
        assert (state.detach()["eta"] == 1).all()


# ---------------------------------------------------------------------------
# Class options
# ---------------------------------------------------------------------------


class TestClassOptions:
    def test_shadow_blocks_by_default(self):
        with pytest.raises(AttributeError, match="shadows a TensorDictBase attribute"):

            class Bad(TypedTensorDict):
                clone: Tensor

    def test_shadow_allows_conflicting_names(self):
        s = ShadowClass(
            data=torch.randn(3, 2),
            clone=torch.randn(3, 4),
            batch_size=[3],
        )
        assert s.data.shape == (3, 2)
        assert s.clone.shape == (3, 4)

    def test_shadow_set_via_attr(self):
        s = ShadowClass(
            data=torch.randn(3, 2),
            clone=torch.randn(3, 4),
            batch_size=[3],
        )
        s.data = torch.ones(3, 2)
        assert (s.data == 1).all()

    def test_frozen(self):
        f = FrozenClass(x=torch.randn(3), batch_size=[3])
        assert f.is_locked

    def test_frozen_prevents_modification(self):
        f = FrozenClass(x=torch.randn(3), batch_size=[3])
        with pytest.raises(RuntimeError):
            f["x"] = torch.ones(3)

    def test_combined_options(self):
        sf = ShadowFrozenClass(data=torch.randn(3, 2), batch_size=[3])
        assert sf.data.shape == (3, 2)
        assert sf.is_locked

    def test_option_propagation(self):
        class Child(FrozenClass):
            y: Tensor

        assert Child._frozen is True
        c = Child(x=torch.randn(3), y=torch.randn(3), batch_size=[3])
        assert c.is_locked

    def test_option_propagation_shadow(self):
        class SChild(ShadowClass):
            extra: Tensor

        assert SChild._shadow is True
        sc = SChild(
            data=torch.randn(3),
            clone=torch.randn(3),
            extra=torch.randn(3),
            batch_size=[3],
        )
        assert sc.data.shape == (3,)

    def test_invalid_option(self):
        with pytest.raises(ValueError, match="Unknown TypedTensorDict option"):
            TypedTensorDict["invalid_option"]

    @pytest.mark.parametrize("option", ["autocast", "nocast", "tensor_only"])
    def test_ignored_option_deprecated(self, option):
        match = rf"option '{option}' has no effect.*removed in TensorDict 0\.17"
        with pytest.warns(DeprecationWarning, match=match) as record:
            base = TypedTensorDict[option]
        assert record[0].filename == __file__
        with pytest.warns(DeprecationWarning, match=match) as record:

            class Keyword(TypedTensorDict, **{option: True}):
                x: Tensor

        assert record[0].filename == __file__
        # a subclass inherits the option without a new warning
        with warnings.catch_warnings():
            warnings.simplefilter("error")

            class Child(base):
                x: Tensor

        assert getattr(Child, f"_{option}") is True
        assert getattr(Keyword, f"_{option}") is True

    def test_supported_options_do_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            TypedTensorDict["shadow", "frozen"]

            class Frozen(TypedTensorDict, frozen=True):
                x: Tensor


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_subclass(self):
        class Empty(TypedTensorDict):
            pass

        e = Empty(batch_size=[3])
        assert e.batch_size == torch.Size([3])
        assert len(e.keys()) == 0

    def test_data_requires_shadow(self):
        """'data' shadows a TensorDictBase attribute and requires shadow=True."""
        with pytest.raises(AttributeError, match="shadows a TensorDictBase attribute"):

            class WithData(TypedTensorDict):
                data: Tensor
                x: Tensor

    def test_batch_iter(self):
        """__iter__ on TypedTensorDict iterates over batch dimension."""
        state = PredictorState(
            eta=torch.randn(4, 3),
            X=torch.randn(4, 3),
            beta=torch.randn(4, 3),
            batch_size=[4],
        )
        items = list(state)
        assert len(items) == 4

    def test_contains(self):
        state = PredictorState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            batch_size=[3],
        )
        assert "eta" in state
        assert "nonexistent" not in state

    def test_len(self):
        state = PredictorState(
            eta=torch.randn(4),
            X=torch.randn(4),
            beta=torch.randn(4),
            batch_size=[4],
        )
        assert len(state) == 4

    def test_repr(self):
        state = PredictorState(
            eta=torch.randn(3),
            X=torch.randn(3),
            beta=torch.randn(3),
            batch_size=[3],
        )
        r = repr(state)
        assert "PredictorState" in r or "TensorDict" in r


# ---------------------------------------------------------------------------
# from_tensordict
# ---------------------------------------------------------------------------


class TestFromTensordict:
    def _make_td(self):
        return TensorDict(
            {
                "eta": torch.randn(5, 3),
                "X": torch.randn(5, 4),
                "beta": torch.randn(5, 1),
            },
            batch_size=[5],
        )

    def test_from_tensordict_basic(self):
        td = self._make_td()
        state = PredictorState.from_tensordict(td)
        assert isinstance(state, PredictorState)
        assert isinstance(state, TensorDictBase)
        assert state.batch_size == torch.Size([5])
        assert state.eta.shape == (5, 3)
        assert state._source is td

    def test_from_tensordict_missing_key(self):
        td = TensorDict({"eta": torch.randn(5, 3)}, batch_size=[5])
        with pytest.raises(TypeError, match="missing required field"):
            PredictorState.from_tensordict(td)

    def test_from_tensordict_extra_keys(self):
        td = TensorDict(
            {
                "eta": torch.randn(5, 3),
                "X": torch.randn(5, 4),
                "beta": torch.randn(5, 1),
                "extra": torch.randn(5),
            },
            batch_size=[5],
        )
        state = PredictorState.from_tensordict(td)
        assert isinstance(state, PredictorState)
        assert "extra" in state.keys()

    def test_from_tensordict_lazy_stack(self):
        td1 = TensorDict(
            {"eta": torch.randn(3), "X": torch.randn(4), "beta": torch.randn(1)},
            batch_size=[],
        )
        td2 = TensorDict(
            {"eta": torch.randn(3), "X": torch.randn(4), "beta": torch.randn(1)},
            batch_size=[],
        )
        ls = lazy_stack([td1, td2])
        state = PredictorState.from_tensordict(ls)
        assert isinstance(state, PredictorState)
        assert state.batch_size == torch.Size([2])
        assert state.eta.shape == (2, 3)

    def test_from_tensordict_lazy_stack_repr_does_not_stack(self, monkeypatch):
        # #2800: printing reads the shapes and dtypes from the members, as the
        # lazy stack's own repr does
        tds = [self._make_td() for _ in range(2)]
        tds[1]["X"] = tds[1]["X"].double()
        state = PredictorState.from_tensordict(lazy_stack(tds))
        stack = torch.stack
        calls = []

        def counting_stack(tensors, *args, **kwargs):
            calls.append(len(tensors))
            return stack(tensors, *args, **kwargs)

        monkeypatch.setattr(torch, "stack", counting_stack)
        text = repr(state)
        assert calls == []
        # torch.stack promotes the dtype of X
        assert (
            "X: Tensor(shape=torch.Size([2, 5, 4]), device=cpu, "
            "dtype=torch.float64, is_shared=False)" in text
        )
        assert (
            "eta: Tensor(shape=torch.Size([2, 5, 3]), device=cpu, "
            "dtype=torch.float32, is_shared=False)" in text
        )

    def test_from_tensordict_lazy_stack_repr_ragged(self):
        # the entries whose shapes differ print with -1, as in the lazy stack
        tds = [self._make_td() for _ in range(2)]
        tds[1]["beta"] = torch.zeros(5, 2)
        ls = lazy_stack(tds)
        state = PredictorState.from_tensordict(ls)
        fields = repr(ls).split("    exclusive_fields")[0]
        assert repr(state).startswith(
            fields.replace(type(ls).__name__, "PredictorState")
        )
        assert "beta: Tensor(shape=torch.Size([2, 5, -1])" in fields

    def test_from_tensordict_lazy_stack_inplace(self):
        # #2242: the in-place methods write into the members of the stack
        tds = [self._make_td() for _ in range(2)]
        state = PredictorState.from_tensordict(lazy_stack(tds))
        expected = state.to_tensordict()
        other = expected.apply(lambda x: torch.rand_like(x) + 1)
        for td in (state, expected):
            assert td.abs_() is td
            assert td.add_(other, alpha=2.0) is td
            td *= 2
            assert td.lerp_(other, 0.5) is td
            assert td.update_(other.select("eta") * 3) is td
        assert (state._source[0]["eta"] == tds[0]["eta"]).all()
        torch.testing.assert_close(lazy_stack(tds)["eta"], expected["eta"])
        torch.testing.assert_close(state.to_tensordict(), expected)
        state.copy_(other)
        torch.testing.assert_close(state.to_tensordict(), other)

    def _make_nested_lazy_state(self):
        class NestedState(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(
                a=torch.arange(3.0), n={"x": torch.ones(3, 2) * i}, batch_size=[3]
            )
            for i in range(4)
        ]
        return NestedState.from_tensordict(lazy_stack(tds)), tds

    def test_from_tensordict_lazy_stack_nested_masked_fill_detach_(self):
        # masked_fill_ and detach_ write into the members of a stack with a
        # nested entry
        state, tds = self._make_nested_lazy_state()
        expected = state.to_tensordict()
        mask = torch.tensor([[True, False, True]] * 2 + [[False] * 3] * 2)
        for td in (state, expected):
            assert td.masked_fill_(mask, -1.0) is td
            assert td.detach_() is td
        torch.testing.assert_close(state.to_tensordict(), expected)
        torch.testing.assert_close(lazy_stack(tds).to_tensordict(), expected)

    def test_from_tensordict_lazy_stack_nested_zero_(self):
        state, tds = self._make_nested_lazy_state()
        assert state.zero_() is state
        for td in tds:
            assert (td == 0).all()

    @pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
    def test_from_tensordict_lazy_stack_read_only_inplace(self, tmp_path):
        # A stack of files opened read-only stays unchanged, and warns
        files = []
        for i in range(2):
            self._make_td().to_h5(tmp_path / f"{i}.h5").close()
            files.append(PersistentTensorDict.from_h5(tmp_path / f"{i}.h5"))
        state = PredictorState.from_tensordict(LazyStackedTensorDict(*files))
        before = state.to_tensordict()
        with pytest.warns(FutureWarning, match="leaves the file unchanged"):
            state.add_(1.0)
        with pytest.warns(FutureWarning, match="leaves the file unchanged"):
            state.update_(before + 1)
        torch.testing.assert_close(state.to_tensordict(), before)

    @pytest.mark.parametrize("hetero", [False, True], ids=["same", "hetero"])
    def test_from_tensordict_lazy_stack_nested_inplace(self, hetero):
        # #2519: the in-place methods write into the members of a stack with a
        # nested entry, also when the nested members differ in shape
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(
                a=torch.ones(3), n={"x": torch.ones(3, 2 + hetero * i)}, batch_size=[3]
            )
            for i in range(4)
        ]
        state = Nested.from_tensordict(lazy_stack(tds))
        other = lazy_stack(tds).apply(lambda x: torch.full_like(x, 2.0))
        assert state.zero_() is state
        assert state.add_(1.0) is state
        assert state.add_(other, alpha=2.0) is state
        assert state.lerp_(other, 0.5) is state
        for td in tds:
            assert (td["a"] == 3.5).all()
            assert (td["n", "x"] == 3.5).all()

    def test_from_tensordict_lazy_stack_exclusive_keys_inplace(self):
        # the in-place methods leave the keys that only some members hold
        class Flat(TypedTensorDict):
            a: Tensor

        tds = [
            TensorDict(a=torch.ones(3), batch_size=[3]),
            TensorDict(a=torch.ones(3), c=torch.ones(3), batch_size=[3]),
        ]
        state = Flat.from_tensordict(lazy_stack(tds))
        assert state.add_(1.0) is state
        assert (tds[1]["a"] == 2).all()
        assert (tds[1]["c"] == 1).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_from_tensordict_lazy_stack_update_rejected(self, stack_dim):
        # #2776: a part of the input that a member rejects raises before the
        # members before it are written
        class Flat(TypedTensorDict):
            a: Tensor

        tds = [
            TensorDict(a=torch.ones(3, 2), batch_size=[3]),
            TensorDict(a=torch.ones(3, 3), batch_size=[3]),
        ]
        state = Flat.from_tensordict(lazy_stack(tds, stack_dim))
        source = TensorDict(
            a=torch.zeros(*state.batch_size, 2), batch_size=state.batch_size
        )
        with pytest.raises(RuntimeError, match="must match the size of tensor"):
            state.update_(source)
        assert (tds[0]["a"] == 1).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    def test_from_tensordict_lazy_stack_index_views(self, stack_dim):
        # #2759: indexing returns views of the members, as indexing the stack
        # does, so writes through the result reach them
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(a=torch.ones(3), n={"x": torch.ones(3, 2)}, batch_size=[3])
            for _ in range(4)
        ]
        tds[0]["e"] = torch.ones(3)
        state = Nested.from_tensordict(lazy_stack(tds, stack_dim))
        index = (slice(None),) * stack_dim
        sliced = state[(*index, slice(1, None))]
        assert isinstance(sliced, Nested)
        assert sliced.add_(1) is sliced
        sliced.set_("a", torch.full_like(sliced["a"], 5.0))
        for td in tds[1:]:
            assert (td["a"] == 5).all()
            assert (td["n", "x"] == 2).all()
        assert (tds[0]["a"] == 1).all()
        # out-of-place math gives a new Nested and leaves the members alone
        plus = sliced + 1
        assert isinstance(plus, Nested)
        assert (plus["a"] == 6).all()
        assert (tds[1]["a"] == 5).all()
        sliced.copy_(plus)
        for td in tds[1:]:
            assert (td["a"] == 6).all()
            assert (td["n", "x"] == 3).all()
        # an int on the stack dim gives the member, with its own keys
        first = state[(*index, 0)]
        assert isinstance(first, Nested)
        assert first._source is tds[0]
        assert "e" in first.keys()
        first["a"] = torch.full((3,), 7.0)
        assert (tds[0]["a"] == 7).all()
        # typed members: the result wraps the member's source, not the member
        typed = Nested.from_tensordict(
            lazy_stack([Nested.from_tensordict(td) for td in tds], stack_dim)
        )
        assert typed[(*index, 0)]._source is tds[0]

    @pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
    def test_from_tensordict_lazy_stack_files_index_copies(self, tmp_path):
        # members that read copies (files) index into a copy, as on main, so
        # select, exclude and empty work on the result
        files = [
            PersistentTensorDict.from_dict(
                self._make_td(), filename=tmp_path / f"{i}.h5"
            )
            for i in range(2)
        ]
        state = PredictorState.from_tensordict(lazy_stack(files))
        for sub in (state[0], state[1:]):
            assert isinstance(sub, PredictorState)
            assert not isinstance(sub._source, LazyStackedTensorDict)
            sub.select("eta")
            sub.exclude("eta")
            sub.empty()

    @pytest.mark.parametrize("exclusive", ["e", ("n", "e")], ids=["top", "nested"])
    def test_from_tensordict_lazy_stack_nested_exclusive_keys_inplace(self, exclusive):
        # #2739: with a nested entry, the in-place methods change the shared
        # keys of every member, add no key and leave the exclusive keys
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(a=torch.ones(3), n={"x": torch.ones(3, 2)}, batch_size=[3])
            for _ in range(4)
        ]
        tds[0][exclusive] = torch.ones(3)
        state = Nested.from_tensordict(lazy_stack(tds))
        other = state.to_tensordict().apply(lambda x: torch.full_like(x, 2.0))
        assert state.zero_() is state
        assert state.add_(1.0) is state
        assert state.add_(other, alpha=2.0) is state
        assert state.lerp_(lazy_stack(tds).apply(torch.ones_like), 0.5) is state
        # a typed operand, as in a soft update
        online = Nested.from_tensordict(
            lazy_stack([td.apply(lambda x: torch.full_like(x, 5.0)) for td in tds])
        )
        assert state.lerp_(online, 0.5) is state
        for td in tds:
            assert (td["a"] == 4).all()
            assert (td["n", "x"] == 4).all()
        assert (tds[0][exclusive] == 1).all()
        assert set(tds[0].keys(True, True)) == {"a", ("n", "x"), exclusive}
        for td in tds[1:]:
            assert set(td.keys(True, True)) == {"a", ("n", "x")}
        # a call that raises writes nothing
        with pytest.raises(KeyError):
            state.add_(other.exclude(("n", "x")))
        for td in tds:
            assert (td["a"] == 4).all()

    @pytest.mark.parametrize("method", ["copy_", "update_"])
    @pytest.mark.parametrize("dest_type", ["dense", "typed"])
    @pytest.mark.parametrize("src_type", ["typed", "nested"])
    def test_from_tensordict_lazy_stack_nested_as_source(
        self, method, dest_type, src_type
    ):
        # #2768: copy_ and update_ write the nested entries of a stack with a
        # nested entry, read through a TypedTensorDict or held by a tensordict
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(
                a=torch.full((3,), i), n={"x": torch.full((3, 2), i)}, batch_size=[3]
            )
            for i in (1.0, 2.0, 3.0, 4.0)
        ]
        if src_type == "typed":
            src = Nested.from_tensordict(lazy_stack(tds))
        else:
            src = TensorDict(
                a=lazy_stack(tds)["a"], n=lazy_stack(tds)["n"], batch_size=[4, 3]
            )
        dest = TensorDict(
            a=torch.zeros(4, 3),
            n={"x": torch.zeros(4, 3, 2)},
            b=torch.zeros(4, 3),
            batch_size=[4, 3],
        )
        if dest_type == "typed":
            dest = Nested.from_tensordict(dest)
        assert getattr(dest, method)(src) is dest
        expected = lazy_stack(tds).to_tensordict()
        torch.testing.assert_close(dest.exclude("b").to_tensordict(), expected)
        assert (dest["b"] == 0).all()

    @pytest.mark.parametrize("method", ["copy_", "update_"])
    def test_from_tensordict_lazy_stack_nested_as_source_other_keys(self, method):
        # #2768: a ragged leaf that the destination lacks is not read, and a
        # lazy stack in the destination is still written
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(
                a=torch.full((3,), i),
                n={"x": torch.full((3,), i), "info": torch.ones(3, int(i))},
                batch_size=[3],
            )
            for i in (1.0, 2.0)
        ]
        dest = TensorDict(
            a=torch.zeros(2, 3), n={"x": torch.zeros(2, 3)}, batch_size=[2, 3]
        )
        getattr(dest, method)(Nested.from_tensordict(lazy_stack(tds)))
        torch.testing.assert_close(dest["n", "x"], lazy_stack(tds)["n", "x"])

        def stack(value):
            return lazy_stack(
                [TensorDict(x=torch.full((3,), value), batch_size=[3]) for _ in "ab"]
            )

        dest = TensorDict(n=stack(0.0), m={"x": torch.zeros(2, 3)}, batch_size=[2, 3])
        getattr(dest, method)(TensorDict(n=stack(1.0), m=stack(2.0), batch_size=[2, 3]))
        assert (dest["n", "x"] == 1).all()
        assert (dest["m", "x"] == 2).all()

    @pytest.mark.parametrize("op", ["add", "mul", "sub"])
    @pytest.mark.parametrize("nested", [False, True], ids=["flat", "nested"])
    def test_from_tensordict_lazy_stack_binary_ops(self, nested, op):
        # #2772: the out-of-place binary ops give what they give on the stack,
        # with a scalar or a typed operand
        class Flat(TypedTensorDict):
            a: Tensor

        class Nested(Flat):
            n: TensorDict

        tds = [
            TensorDict(
                a=torch.full((3,), i), n={"x": torch.full((3, 2), i)}, batch_size=[3]
            )
            for i in (1.0, 2.0)
        ]
        cls = Nested
        if not nested:
            tds = [td.exclude("n") for td in tds]
            cls = Flat
        before = lazy_stack(tds).to_tensordict()
        state = cls.from_tensordict(lazy_stack(tds))
        for other in (2.0, state):
            result = getattr(state, op)(other)
            expected = getattr(lazy_stack(tds), op)(
                other._source if isinstance(other, TypedTensorDict) else other
            )
            assert isinstance(result, cls)
            torch.testing.assert_close(result.to_tensordict(), expected.to_tensordict())
        torch.testing.assert_close(lazy_stack(tds).to_tensordict(), before)

    def test_from_tensordict_lazy_stack_data_setter(self):
        # .data pairs the leaves of a stack with a nested entry with those of
        # a dense tensordict, either way (#2772)
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(a=torch.ones(3), n={"x": torch.ones(3, 2)}, batch_size=[3])
            for _ in range(2)
        ]
        state = Nested.from_tensordict(lazy_stack(tds))
        state.data = state.to_tensordict() + 1
        for td in tds:
            assert (td == 2).all()
        dense = state.to_tensordict().zero_()
        dense.data = state
        assert (dense == 2).all()
        # a key that only some members hold raises before anything is set
        tds[0]["e"] = torch.ones(3)
        value = lazy_stack(tds).to_tensordict() + 1
        value["e"] = torch.zeros(2, 3)
        with pytest.raises(KeyError):
            state.data = value
        for td in tds:
            assert (td["a"] == 2).all()

    @staticmethod
    def _make_members(kind, tmp_path):
        def nested(i):
            return TensorDict(
                a=torch.full((3,), i), n={"x": torch.full((3, 2), i)}, batch_size=[3]
            )

        members = [nested(i) for i in (1.0, 2.0)]
        if kind == "locked_members":
            return lazy_stack([td.lock_() for td in members])
        if kind == "locked_stack":
            return lazy_stack(members).lock_()
        if kind == "shared":
            return lazy_stack([td.share_memory_() for td in members])
        if kind == "memmap":
            return lazy_stack([td.memmap_() for td in members])
        if kind == "stack_of_stacks":
            return lazy_stack([lazy_stack(list(td.unbind(0))) for td in members])
        if kind == "sub_td":
            return lazy_stack(
                [td.expand(2, 3).clone()._get_sub_tensordict(0) for td in members]
            )
        if kind == "nested_h5":
            for i, td in enumerate(members):
                td["n"] = PersistentTensorDict.from_dict(
                    td["n"], filename=tmp_path / f"n{i}.h5"
                )
            return lazy_stack(members)
        if kind == "h5":
            return lazy_stack(
                [
                    PersistentTensorDict.from_dict(td, filename=tmp_path / f"{i}.h5")
                    for i, td in enumerate(members)
                ]
            )
        return lazy_stack(members)

    @pytest.mark.parametrize(
        "kind",
        [
            "plain",
            "locked_members",
            "locked_stack",
            "shared",
            "memmap",
            "stack_of_stacks",
            "sub_td",
            pytest.param(
                "h5", marks=pytest.mark.skipif(not _has_h5py, reason="no h5py")
            ),
            pytest.param(
                "nested_h5", marks=pytest.mark.skipif(not _has_h5py, reason="no h5py")
            ),
        ],
    )
    @pytest.mark.parametrize("src_type", ["typed", "dense"])
    @pytest.mark.parametrize("method", ["copy_", "update_"])
    def test_from_tensordict_lazy_stack_nested_as_dest(
        self, method, src_type, kind, tmp_path
    ):
        # #2771, #2773: copy_ and update_ write every entry into the members
        # of a stack with a nested entry, whatever the members are
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        source = self._make_members(kind, tmp_path)
        dest = Nested.from_tensordict(source)
        values = source.to_tensordict().apply(lambda x: x + 10)
        src = values
        if src_type == "typed":
            src = Nested.from_tensordict(lazy_stack(list(values.unbind(0))))
        assert getattr(dest, method)(src) is dest
        # read back from the members, from the files for h5
        torch.testing.assert_close(source.to_tensordict(), values)
        if "h5" in kind:
            # .data cannot set what the files hold: it raises before it sets
            with pytest.raises(RuntimeError, match="Cannot assign .data"):
                dest.data = values + 1
            torch.testing.assert_close(source.to_tensordict(), values)
        assert dest.add_(1) is dest
        assert dest.zero_() is dest
        assert (source.to_tensordict() == 0).all()

    @pytest.mark.parametrize(
        "call",
        [
            lambda td, src: td.copy_(src),
            lambda td, src: td.update_(src),
            lambda td, src: td.update_(src, keys_to_update=[("n", "x")]),
            lambda td, src: td.update_(src, keys_to_update=["n"]),
        ],
        ids=["copy_", "update_", "update_leaf_key", "update_prefix_key"],
    )
    def test_from_tensordict_lazy_stack_held_as_dest(self, call):
        # #2780: copy_ and update_ write the members of a TypedTensorDict over a
        # lazy stack that a TensorDict holds
        class Typed(TypedTensorDict):
            x: Tensor

        members = [TensorDict(x=torch.zeros(3)) for _ in range(2)]
        dest = TensorDict(
            a=torch.zeros(2),
            n=Typed.from_tensordict(lazy_stack(members)),
            batch_size=[2],
        )
        expected = dest.to_tensordict()
        src = expected + 1
        assert call(dest, src) is dest
        call(expected, src)
        torch.testing.assert_close(dest.to_tensordict(), expected)
        for member in members:
            assert (member["x"] == 1).all()

    @pytest.mark.parametrize("method", ["copy_", "update_"])
    def test_from_tensordict_lazy_stack_exclusive_keys_as_dest(self, method):
        # #2771: copy_ and update_ leave the keys that only some members hold,
        # also when the source holds them (#2739)
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(a=torch.ones(3), n={"x": torch.ones(3, 2)}, batch_size=[3])
            for _ in range(2)
        ]
        tds[0]["e"] = torch.ones(3)
        dest = Nested.from_tensordict(lazy_stack(tds))
        src = Nested.from_tensordict(
            lazy_stack([td.apply(lambda x: x + 1) for td in tds])
        )
        assert getattr(dest, method)(src) is dest
        for td in tds:
            assert (td["a"] == 2).all()
            assert (td["n", "x"] == 2).all()
        assert (tds[0]["e"] == 1).all()
        # nested in the source of a dense tensordict, as on main
        outer = TensorDict(t=src, batch_size=[2])
        dense = outer.to_tensordict().exclude(("t", "e")).zero_()
        assert getattr(dense, method)(outer) is dense
        assert (dense == outer.to_tensordict().exclude(("t", "e"))).all()

    def test_from_tensordict_lazy_stack_of_stacks_exclusive_keys_inplace(self):
        # #2759: a key that only some members of an inner stack hold is left
        # unchanged, as in a single stack
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(a=torch.ones(3), n={"x": torch.ones(3, 2)}, batch_size=[3])
            for _ in range(4)
        ]
        tds[0]["e"] = torch.ones(3)
        state = Nested.from_tensordict(
            lazy_stack([lazy_stack(tds[:2]), lazy_stack(tds[2:])])
        )
        assert state.add_(state.to_tensordict()) is state
        assert state.add_(1.0) is state
        for td in tds:
            assert (td["a"] == 3).all()
            assert (td["n", "x"] == 3).all()
        assert (tds[0]["e"] == 1).all()

    @pytest.mark.parametrize("depth", [1, 2])
    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize(
        "op", ["add_", "mul_", "zero_", "lerp_", "clamp_min_", "copy_", "update_"]
    )
    def test_from_tensordict_lazy_stack_held_exclusive_keys_inplace(
        self, op, stack_dim, depth, tmp_path
    ):
        # #2759: a key that only some members of a lazy stack that a member
        # holds is left unchanged, as in a single stack (#2739)
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        batch = [3] if stack_dim else []

        def member(extra, value=1.0):
            inner = [
                TensorDict(x=torch.full((3,), value), batch_size=batch)
                for _ in range(2)
            ]
            if extra:
                inner[0]["e"] = torch.full((3,), value)
            n = lazy_stack(inner, stack_dim)
            if depth == 2:
                n = TensorDict(m=n, batch_size=batch)
            return TensorDict(a=torch.full((3,), value), n=n, batch_size=batch)

        tds = [member(True), member(False)]
        state = Nested.from_tensordict(lazy_stack(tds))
        dense = state.to_tensordict().apply(lambda x: torch.full_like(x, 3.0))
        # an input that holds e too
        held = Nested.from_tensordict(
            lazy_stack([member(True, 3.0), member(False, 3.0)])
        )
        args = {
            "add_": (1.0,),
            "mul_": (2.0,),
            "zero_": (),
            "lerp_": (dense, 0.5),
            "clamp_min_": (3.0,),
            "copy_": (held,),
            "update_": (held,),
        }[op]
        assert getattr(state, op)(*args) is state
        expected = {"zero_": 0.0, "copy_": 3.0, "update_": 3.0, "clamp_min_": 3.0}
        expected = expected.get(op, 2.0)
        inner = ("n", "m") if depth == 2 else ("n",)
        for td in tds:
            assert (td["a"] == expected).all()
            assert (td[inner + ("x",)] == expected).all()
        assert (tds[0][inner].tensordicts[0]["e"] == 1).all()
        if op == "update_":
            # a dense input reaches the leaves of the inner stack too
            assert state.update_(dense.apply(lambda x: x + 1)) is state
            for td in tds:
                assert (td[inner + ("x",)] == 4).all()
            assert (tds[0][inner].tensordicts[0]["e"] == 1).all()
        if op in ("copy_", "update_") and _has_h5py:
            # an h5 file, or a lazy stack of them, which cannot select
            values = dense.apply(lambda x: x + 2)
            file = PersistentTensorDict.from_dict(values, filename=tmp_path / "a.h5")
            files = lazy_stack(
                [
                    PersistentTensorDict.from_dict(
                        value + 1, filename=tmp_path / f"{i}.h5"
                    )
                    for i, value in enumerate(values.unbind(0))
                ]
            )
            for value, source in ((5, file), (6, files)):
                assert getattr(state, op)(source) is state
                for td in tds:
                    assert (td["a"] == value).all()
                    assert (td[inner + ("x",)] == value).all()
                assert (tds[0][inner].tensordicts[0]["e"] == 1).all()
        if op == "add_":
            # a call that raises writes nothing
            with pytest.raises(KeyError):
                state.add_(dense.exclude(inner + ("x",)))
            for td in tds:
                assert (td["a"] == expected).all()

    @pytest.mark.parametrize("op", ["add_", "zero_", "update_", "copy_"])
    def test_from_tensordict_lazy_stack_held_exclusive_keys_batch_sizes_differ(
        self, op
    ):
        # #2759: the members hold tensordicts of other batch sizes, so their
        # shared leaves cannot be read: the source changes them, as on main
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        def members(value):
            inner = [
                TensorDict(x=torch.full((3, 2), value), batch_size=[3])
                for _ in range(2)
            ]
            inner[0]["e"] = torch.full((3, 2), value)
            return [
                TensorDict(
                    a=torch.full((3,), value),
                    n=TensorDict(x=torch.full((3, 2, 2), value), batch_size=[3]),
                    batch_size=[3],
                ),
                TensorDict(
                    a=torch.full((3,), value), n=lazy_stack(inner, 1), batch_size=[3]
                ),
            ]

        tds = members(1.0)
        state = Nested.from_tensordict(lazy_stack(tds))
        args = {
            "add_": (2.0,),
            "zero_": (),
            "update_": (lazy_stack(members(3.0)),),
            "copy_": (lazy_stack(members(3.0)),),
        }[op]
        assert getattr(state, op)(*args) is state
        expected = 0.0 if op == "zero_" else 3.0
        for td in tds:
            assert (td["a"] == expected).all()
            assert (td["n", "x"] == expected).all()

    @pytest.mark.parametrize("op", ["add_", "zero_", "mul_", "copy_"])
    def test_from_tensordict_lazy_stack_uneven_exclusive_keys_inplace(self, op):
        # #2759: the shared leaves differ in shape, so they cannot be read as
        # one tensor: the method runs on the shared leaves of each member and
        # leaves the exclusive keys unchanged (#2739)
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(
                a=torch.full((3, k + 1), 2.0),
                n={"x": torch.full((3, 2), 2.0)},
                batch_size=[3],
            )
            for k in range(2)
        ]
        tds[0]["e"] = torch.full((3,), 2.0)
        state = Nested.from_tensordict(lazy_stack(tds))
        args = {"add_": (1.0,), "zero_": (), "mul_": (state,), "copy_": (state + 1,)}
        assert getattr(state, op)(*args[op]) is state
        expected = {"add_": 3.0, "zero_": 0.0, "mul_": 4.0, "copy_": 3.0}[op]
        for td in tds:
            assert (td["a"] == expected).all()
            assert (td["n", "x"] == expected).all()
        assert (tds[0]["e"] == 2).all()

    @pytest.mark.parametrize("receiver", ["dense", "lazy", "typed"])
    @pytest.mark.parametrize("wrap", ["typed", "tensorclass"])
    def test_dense_inplace_typed_lazy_stack_operand(self, wrap, receiver):
        # #2759: a dense receiver pairs its leaves with those of a typed
        # operand over a lazy stack, also behind a tensorclass; a lazy
        # receiver, also behind a TypedTensorDict, still does
        class Nested(TypedTensorDict):
            a: Tensor
            n: TensorDict

        @tensorclass
        class Wrapper:
            a: Tensor
            n: TensorDict

        tds = [
            TensorDict(a=torch.ones(3), n={"x": torch.ones(3, 2)}, batch_size=[3])
            for _ in range(4)
        ]
        tds[0]["e"] = torch.ones(3)
        other = Nested.from_tensordict(lazy_stack(tds))
        if wrap == "tensorclass":
            other = Wrapper.from_tensordict(other)
        dense = TensorDict(
            a=torch.zeros(4, 3), n={"x": torch.zeros(4, 3, 2)}, batch_size=[4, 3]
        )
        if receiver == "lazy":
            dense = lazy_stack(dense.unbind(0))
        elif receiver == "typed":
            dense = Nested.from_tensordict(lazy_stack(dense.unbind(0)))
        assert dense.lerp_(other, 0.5) is dense
        assert dense.maximum_(other.apply(lambda x: x * 0.75)) is dense
        assert dense.add_(other) is dense
        assert (dense == 1.75).all()

    def test_from_tensordict_memmap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            td = self._make_td()
            td_mm = td.memmap_(tmpdir)
            state = PredictorState.from_tensordict(td_mm)
            assert isinstance(state, PredictorState)
            assert state._is_memmap
            assert state.eta.shape == (5, 3)

    def test_from_tensordict_live_link(self):
        """Mutations through the TypedTensorDict are reflected in the source."""
        td = self._make_td()
        state = PredictorState.from_tensordict(td)
        new_eta = torch.ones(5, 3)
        state.eta = new_eta
        assert (td["eta"] == 1).all()

    def test_from_tensordict_operations(self):
        td = self._make_td()
        state = PredictorState.from_tensordict(td)

        cloned = state.clone()
        assert isinstance(cloned, PredictorState)

        indexed = state[0]
        assert isinstance(indexed, PredictorState)
        assert indexed.batch_size == torch.Size([])

        expanded = state.expand(2, 5)
        assert isinstance(expanded, PredictorState)
        assert expanded.batch_size == torch.Size([2, 5])

    def test_from_tensordict_stack(self):
        td = self._make_td()
        s1 = PredictorState.from_tensordict(td)
        s2 = PredictorState.from_tensordict(self._make_td())
        stacked = torch.stack([s1, s2])
        assert isinstance(stacked, PredictorState)
        assert stacked.batch_size == torch.Size([2, 5])

    def test_from_tensordict_check_false_empty(self):
        """Wrapping an empty TD with check=False succeeds."""
        td = TensorDict({}, batch_size=[5])
        state = PredictorState.from_tensordict(td, check=False)
        assert isinstance(state, PredictorState)
        assert state.batch_size == torch.Size([5])
        assert len(list(state.keys())) == 0

    def test_from_tensordict_check_false_partial(self):
        """Wrapping a partially-filled TD with check=False succeeds."""
        td = TensorDict({"eta": torch.randn(5, 3)}, batch_size=[5])
        state = PredictorState.from_tensordict(td, check=False)
        assert isinstance(state, PredictorState)
        assert state.eta.shape == (5, 3)
        with pytest.raises(AttributeError, match="declared but not set"):
            _ = state.X

    def test_from_tensordict_check_false_then_fill(self):
        """Fill fields iteratively after wrapping with check=False."""
        td = TensorDict({}, batch_size=[5])
        state = PredictorState.from_tensordict(td, check=False)
        state["eta"] = torch.randn(5, 3)
        state["X"] = torch.randn(5, 4)
        state["beta"] = torch.randn(5, 1)
        assert state.eta.shape == (5, 3)
        assert set(state.keys()) == {"eta", "X", "beta"}

    def test_from_tensordict_check_true_rejects_empty(self):
        """Default check=True rejects empty TD."""
        td = TensorDict({}, batch_size=[5])
        with pytest.raises(TypeError, match="missing required field"):
            PredictorState.from_tensordict(td)


# ---------------------------------------------------------------------------
# Operations over each backend
# ---------------------------------------------------------------------------


class _BackendState(TypedTensorDict):
    a: Tensor
    b: Tensor
    n: TensorDict


def _dense_source():
    return TensorDict(
        a=torch.randn(4, 3),
        b=torch.arange(12).reshape(4, 3).clone(),
        n=TensorDict(c=torch.randn(4, 3, 2), batch_size=[4, 3]),
        batch_size=[4, 3],
    )


def _lazy_source():
    return lazy_stack(
        [
            TensorDict(
                a=torch.randn(3),
                b=torch.arange(3) + i,
                n=TensorDict(c=torch.randn(3, 2), batch_size=[3]),
                batch_size=[3],
            )
            for i in range(4)
        ]
    )


def _h5_source(tmp_path):
    return PersistentTensorDict.from_dict(
        _dense_source(), filename=tmp_path / "source.h5"
    )


def _subtd_source():
    big = TensorDict(
        a=torch.randn(8, 3),
        b=torch.arange(24).reshape(8, 3),
        n=TensorDict(c=torch.randn(8, 3, 2), batch_size=[8, 3]),
        batch_size=[8, 3],
    )
    return big._get_sub_tensordict(slice(2, 6))


_BACKEND_MASK = torch.tensor([[True, False, True]] * 2 + [[False] * 3] * 2)

# Operations, run on a typed tensordict over each source and over a dense copy
# of that source. Each returns what the test compares.
_BACKEND_OPS = {
    "where": lambda s: s.where(_BACKEND_MASK, 0.0),
    "where_out": lambda s: (lambda out: (s.where(_BACKEND_MASK, 0.0, out=out), out))(
        s.to_tensordict().zero_()
    ),
    "where_pad": lambda s: s.where(
        _BACKEND_MASK, s.to_tensordict().exclude("b"), pad=0
    ),
    "eq": lambda s: s == 0,
    "ne": lambda s: s != s.to_tensordict(),
    "lt": lambda s: s < 0.5,
    "all": lambda s: (s == s).all(),
    "any0": lambda s: (s > 0).any(0),
    "masked_fill": lambda s: s.masked_fill(_BACKEND_MASK, -1.0),
    "masked_fill_": lambda s: s.masked_fill_(_BACKEND_MASK, -1.0),
    "split": lambda s: s.split([1, 3], 0),
    "chunk": lambda s: s.chunk(2, 0),
    "expand": lambda s: s.expand(2, 4, 3),
    "reshape": lambda s: s.reshape(-1),
    "repeat_interleave": lambda s: s.repeat_interleave(2, dim=1),
    "detach_": lambda s: s.detach_(),
    "share_memory_": lambda s: s.share_memory_(),
    "popitem": lambda s: s.popitem(),
    "masked_select": lambda s: s.masked_select(_BACKEND_MASK[:, 0]),
    "rename_key_": lambda s: s.rename_key_("a", "z"),
    "del_": lambda s: s.del_("a"),
    "setitem_key": lambda s: s.__setitem__("a", torch.zeros(4, 3)),
    "setitem_index": lambda s: s.__setitem__(0, s[1].to_tensordict()),
}

# What raises on main, with the type of the error
_BACKEND_RAISES = {
    ("h5", "share_memory_"): NotImplementedError,
    ("subtd", "detach_"): RuntimeError,
    ("subtd", "share_memory_"): RuntimeError,
}

# The class of the result of the operations that do not return a typed
# tensordict, on a lazy source and on the other sources
_BACKEND_RESULT_CLASS = {
    "eq": ("LazyStackedTensorDict", "TensorDict"),
    "ne": ("LazyStackedTensorDict", "TensorDict"),
    "lt": ("LazyStackedTensorDict", "TensorDict"),
    "any0": ("TensorDict", "TensorDict"),
}


def _backend_values(result):
    if isinstance(result, TensorDictBase):
        return result.to_tensordict()
    if isinstance(result, tuple):
        return tuple(_backend_values(r) for r in result)
    return result


class TestBackends:
    """The operations of a typed tensordict over each kind of source."""

    @pytest.fixture(params=["dense", "lazy", "h5", "subtd"])
    def source(self, request, tmp_path):
        if request.param == "h5":
            if not _has_h5py:
                pytest.skip("h5py not found.")
            return request.param, _h5_source(tmp_path)
        return request.param, {
            "dense": _dense_source,
            "lazy": _lazy_source,
            "subtd": _subtd_source,
        }[request.param]()

    @pytest.mark.parametrize("op", list(_BACKEND_OPS))
    def test_ops_lazy_source(self, source, op):
        kind, td = source
        fn = _BACKEND_OPS[op]
        state = _BackendState.from_tensordict(td)
        dense = _BackendState.from_tensordict(td.to_tensordict())
        error = _BACKEND_RAISES.get((kind, op))
        if error is not None:
            with pytest.raises(error):
                fn(state)
            return
        expected = fn(dense)
        result = fn(state)
        # the state that each call leaves
        torch.testing.assert_close(state.to_tensordict(), dense.to_tensordict())
        if op in _BACKEND_RESULT_CLASS:
            lazy_cls, other_cls = _BACKEND_RESULT_CLASS[op]
            assert type(result).__name__ == (lazy_cls if kind == "lazy" else other_cls)
        elif op in ("where_out",):
            assert result[0] is result[1]
        elif op in ("all",):
            assert result is True
        elif op == "popitem":
            # the source pops: a lazy stack and a file give their own class
            assert result[0] == "n"
            assert type(result[1]) is {
                "lazy": LazyStackedTensorDict,
                "h5": PersistentTensorDict,
            }.get(kind, TensorDict)
            result, expected = result[1], expected[1]
        elif op in ("split", "chunk"):
            assert all(type(r) is _BackendState for r in result)
        elif op in ("setitem_key", "setitem_index"):
            result = expected = None
        elif op in (
            "masked_fill_",
            "detach_",
            "share_memory_",
            "rename_key_",
            "del_",
        ):
            assert result is state
        else:
            assert type(result) is _BackendState
        if op == "share_memory_":
            assert td.is_shared()
        if op == "where_out":
            result, expected = result[1], expected[1]
        torch.testing.assert_close(_backend_values(result), _backend_values(expected))

    def test_masked_fill_source_class(self, source):
        # masked_fill fills the source's own copy: a file or a view gives a
        # dense TensorDict, which the typed result wraps
        kind, td = source
        state = _BackendState.from_tensordict(td)
        result = state.masked_fill(_BACKEND_MASK, -1.0)
        assert type(result) is _BackendState
        expected = {
            "dense": "TensorDict",
            "lazy": "LazyStackedTensorDict",
            "h5": "TensorDict",
            "subtd": "TensorDict",
        }[kind]
        assert type(result._source).__name__ == expected

    @pytest.mark.parametrize(
        "op",
        [
            lambda s: (s.expand(2, 4, 3),),
            lambda s: (s.reshape(2, 2, 3),),
            lambda s: (s.repeat_interleave(2, dim=1),),
            lambda s: s.split([1, 3], 0),
            lambda s: s.chunk(2, 0),
            lambda s: (s.where(_BACKEND_MASK, 0.0),),
            lambda s: (s.masked_select(_BACKEND_MASK[:, 0]),),
            lambda s: (s.repeat(2, 1),),
        ],
        ids=[
            "expand",
            "reshape",
            "repeat_interleave",
            "split",
            "chunk",
            "where",
            "masked_select",
            "repeat",
        ],
    )
    def test_shape_ops_keep_source_class(self, source, op):
        # the typed result wraps what the source's own operation returns
        kind, td = source
        state = _BackendState.from_tensordict(td)
        expected = [type(r) for r in op(td)]
        result = op(state)
        assert all(type(r) is _BackendState for r in result)
        assert [type(r._source) for r in result] == expected

    def test_contiguous_inplace(self, source):
        kind, td = source
        state = _BackendState.from_tensordict(td)
        if kind == "dense":
            assert state.contiguous(inplace=True) is state
        else:
            with pytest.raises(NotImplementedError, match="contiguous"):
                state.contiguous(inplace=True)

    def test_new_unsafe_base_keywords(self):
        # the keywords that the base code passes to _new_unsafe
        result = _BackendState._new_unsafe(
            source={"a": torch.zeros(4)},
            batch_size=torch.Size([4]),
            device=None,
            names=None,
            lock=True,
            is_shared=False,
            is_memmap=False,
        )
        assert type(result) is _BackendState
        assert type(result._source) is TensorDict
        assert result.batch_size == (4,)
        assert result.is_locked

    @pytest.mark.parametrize(
        "op",
        [
            lambda s: s.reshape(-1, inplace=True),
            lambda s: s.repeat_interleave(2, dim=1, inplace=True),
            lambda s: s.masked_select(_BACKEND_MASK[:, 0], inplace=True),
            lambda s: s.flatten(0, 1, inplace=True),
            lambda s: s.unflatten(0, (2, 2), inplace=True),
        ],
        ids=["reshape", "repeat_interleave", "masked_select", "flatten", "unflatten"],
    )
    def test_inplace_shape_ops(self, source, op):
        # a dense source changes shape; the others raise before writing
        kind, td = source
        state = _BackendState.from_tensordict(td)
        expected = op(_BackendState.from_tensordict(td.to_tensordict()))
        before = td.to_tensordict()
        if kind == "dense":
            assert op(state) is state
            torch.testing.assert_close(state.to_tensordict(), expected.to_tensordict())
            return
        with pytest.raises(NotImplementedError, match="inplace=True|in-place"):
            op(state)
        assert td.batch_size == (4, 3)
        torch.testing.assert_close(td.to_tensordict(), before)

    @staticmethod
    def _flat_lazy_state():
        class Flat(TypedTensorDict):
            a: Tensor
            b: Tensor

        tds = [
            TensorDict(
                a=torch.arange(3.0) + 10 * i, b=torch.arange(3) + 10 * i, batch_size=[3]
            )
            for i in range(4)
        ]
        return Flat.from_tensordict(lazy_stack(tds)), tds

    # (op, outcome): "raise" before any write, "self" without a warning,
    # "warn" (returns self with a FutureWarning, until 0.17), or "error" (the
    # error of torch, with no warning)
    _FLAT_LAZY_INPLACE_OPS = {
        "flatten": (lambda s: s.flatten(0, 1, inplace=True), "raise"),
        "unflatten": (lambda s: s.unflatten(1, (3, 1), inplace=True), "raise"),
        "repeat": (lambda s: s.repeat(1, 2, inplace=True), "raise"),
        "gather_new_size": (
            lambda s: s.gather(1, torch.zeros(4, 2, dtype=torch.long), inplace=True),
            "raise",
        ),
        "reshape": (lambda s: s.reshape(12, inplace=True), "raise"),
        "repeat_interleave": (
            lambda s: s.repeat_interleave(2, dim=1, inplace=True),
            "raise",
        ),
        "masked_select": (
            lambda s: s.masked_select(torch.ones(4, 3, dtype=torch.bool), inplace=True),
            "raise",
        ),
        "pad": (lambda s: s.pad([0, 0, 0, 1], inplace=True), "raise"),
        "pad_stack_dim": (lambda s: s.pad([1, 0], inplace=True), "raise"),
        "gather_out_of_bounds": (
            lambda s: s.gather(1, torch.full((4, 3), 5), inplace=True),
            "error",
        ),
        "flatten_same": (lambda s: s.flatten(1, 1, inplace=True), "self"),
        "unflatten_same": (lambda s: s.unflatten(1, (3,), inplace=True), "self"),
        "repeat_ones": (lambda s: s.repeat(1, 1, inplace=True), "self"),
        "gather_same_shape": (
            lambda s: s.gather(1, torch.tensor([[2, 0, 1]] * 4), inplace=True),
            "warn",
        ),
    }

    @pytest.mark.parametrize("parent", [False, True], ids=["direct", "parent"])
    @pytest.mark.parametrize("op", list(_FLAT_LAZY_INPLACE_OPS))
    def test_inplace_shape_ops_flat_lazy_source(self, op, parent):
        # A call that changes the batch size raises before it writes anything,
        # also through a tensordict that holds the typed tensordict, whose
        # leaves before it stay unchanged. The other calls give the result of
        # a dense copy and keep the members' shapes.
        fn, outcome = self._FLAT_LAZY_INPLACE_OPS[op]
        state, tds = self._flat_lazy_state()
        if parent:
            obj = TensorDict(
                x=torch.arange(12.0).reshape(4, 3), t=state, batch_size=[4, 3]
            )
        else:
            obj = state
        before = obj.to_tensordict()
        if outcome == "error":
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                with pytest.raises(RuntimeError, match="out of bounds"):
                    fn(obj)
        elif outcome == "raise":
            with pytest.raises(NotImplementedError, match="inplace=True|in-place"):
                fn(obj)
            assert obj.batch_size == (4, 3)
            torch.testing.assert_close(obj.to_tensordict(), before)
        else:
            expected = fn(before.clone())
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                assert fn(obj) is obj
            future = [w for w in record if issubclass(w.category, FutureWarning)]
            if outcome == "warn":
                assert len(future) == 1
                assert "0.17" in str(future[0].message)
            else:
                assert not future
            torch.testing.assert_close(obj.to_tensordict(), expected)
        assert isinstance(state._source, LazyStackedTensorDict)
        for td in tds:
            assert td.batch_size == (3,)
            assert td["a"].shape == (3,)

    @pytest.mark.parametrize(
        "op",
        [
            lambda s: s.zero_(),
            lambda s: s.add_(1),
            lambda s: s.mul_(2),
            lambda s: s.clamp_min_(1.5),
            lambda s: s.sqrt_(),
            lambda s: s.lerp_(s.to_tensordict(), 0.5),
        ],
        ids=["zero_", "add_", "mul_", "clamp_min_", "sqrt_", "lerp_"],
    )
    def test_pointwise_lazy_exclusive_keys(self, op):
        # #2739: a stack with a nested entry and keys that only some members
        # hold changes the shared keys as a dense copy does, and adds no key
        tds = [
            TensorDict(
                a=torch.ones(3),
                n=TensorDict(x=torch.ones(3, 2), batch_size=[3]),
                batch_size=[3],
            )
            for _ in range(4)
        ]
        tds[0]["e"] = torch.zeros(3)
        state = _BackendState.from_tensordict(lazy_stack(tds), check=False)
        expected = op(state.to_tensordict())
        assert op(state) is state
        torch.testing.assert_close(state.to_tensordict(), expected)
        assert (tds[0]["e"] == 0).all()
        assert set(tds[0].keys(True, True)) == {"a", ("n", "x"), "e"}
        for td in tds[1:]:
            assert set(td.keys(True, True)) == {"a", ("n", "x")}

    @pytest.mark.parametrize(
        "key",
        ["a", ("n", "c"), "missing", (), ("a", "x"), 0],
        ids=["top", "nested", "missing", "empty", "through_tensor", "int"],
    )
    def test_del_(self, source, key):
        # del_ deletes in the source, which checks the key, as on a dense copy
        kind, td = source
        state = _BackendState.from_tensordict(td, check=False)
        dense = _BackendState.from_tensordict(td.to_tensordict(), check=False)
        try:
            dense.del_(key)
        except Exception as err:
            error = type(err)
        else:
            error = None
        if kind == "h5" and key in ((), 0):
            # a file raises ValueError for an empty key until 0.17
            error = ValueError
        elif kind == "h5" and key == ("a", "x"):
            # and h5py's KeyError for a path through a tensor
            error = KeyError
        if error is not None:
            with pytest.raises(error):
                state.del_(key)
            torch.testing.assert_close(td.to_tensordict(), dense.to_tensordict())
            return
        assert state.del_(key) is state
        assert sorted(td.keys(True, True), key=str) == sorted(
            dense.keys(True, True), key=str
        )

    _SETITEM_OPS = {
        "key": lambda s: s.__setitem__("a", torch.zeros(4, 3)),
        "new_key": lambda s: s.__setitem__("z", torch.zeros(4, 3)),
        "nested_key": lambda s: s.__setitem__(("n", "c"), torch.zeros(4, 3, 2)),
        "new_nested_key": lambda s: s.__setitem__(("n", "z"), torch.zeros(4, 3)),
        "int": lambda s: s.__setitem__(0, s[1].to_tensordict()),
        "slice_scalar": lambda s: s.__setitem__(slice(1, 3), 0),
        "mask_scalar": lambda s: s.__setitem__(_BACKEND_MASK, -1),
        "ellipsis": lambda s: s.__setitem__((Ellipsis, 0), s[:, 1].to_tensordict()),
        "dict": lambda s: s.__setitem__(
            1, {"a": torch.ones(3), "b": torch.ones(3, dtype=torch.long)}
        ),
        "tensor_index": lambda s: s.__setitem__(
            torch.tensor([0, 2]), s[torch.tensor([1, 3])].to_tensordict()
        ),
    }

    @pytest.mark.parametrize("op", list(_SETITEM_OPS))
    def test_setitem(self, source, op):
        # the source writes, as a dense copy does; a sub-tensordict and a file
        # write into an existing key and add a new one
        kind, td = source
        fn = self._SETITEM_OPS[op]
        state = _BackendState.from_tensordict(td, check=False)
        dense = _BackendState.from_tensordict(td.to_tensordict(), check=False)
        try:
            fn(dense)
        except Exception as err:
            error = type(err)
        else:
            error = None
        if kind == "subtd" and op in ("new_key", "new_nested_key"):
            # a sub-tensordict cannot add a key for its slice only
            error = None if op == "new_key" else error
        if error is not None:
            with pytest.raises(error):
                fn(state)
            return
        assert fn(state) is None
        if kind == "subtd" and op == "new_key":
            assert "z" in td.keys()
            return
        torch.testing.assert_close(state.to_tensordict(), dense.to_tensordict())

    @pytest.mark.parametrize("parent", [False, True], ids=["direct", "parent"])
    def test_pad_inplace(self, source, parent):
        # a dense source pads in place; the others cannot change their batch
        # size and raise before anything is padded, also through a parent
        kind, td = source
        state = _BackendState.from_tensordict(td, check=False)
        obj = state
        if parent:
            obj = TensorDict(
                x=torch.arange(12.0).reshape(4, 3), t=state, batch_size=[4, 3]
            )
        before = obj.to_tensordict()
        if kind == "dense":
            assert obj.pad([0, 0, 0, 1], inplace=True) is obj
            torch.testing.assert_close(obj.to_tensordict(), before.pad([0, 0, 0, 1]))
            return
        error = RuntimeError if kind == "subtd" and not parent else None
        with pytest.raises(error or NotImplementedError):
            obj.pad([0, 0, 0, 1], inplace=True)
        assert obj.batch_size == (4, 3)
        torch.testing.assert_close(obj.to_tensordict(), before)

    def test_setitem_locked_source(self, source):
        kind, td = source
        if kind == "subtd":
            td._source.lock_()
        else:
            td.lock_()
        state = _BackendState.from_tensordict(td, check=False)
        with pytest.raises(RuntimeError, match="locked"):
            state["z"] = torch.zeros(4, 3)
        assert "z" not in td.keys()
        if kind != "lazy":
            # an existing key and an index write in place
            state[0] = 0
            assert (td["a"][0] == 0).all()

    def test_del_locked_source(self, source):
        kind, td = source
        if kind == "subtd":
            td._source.lock_()
        else:
            td.lock_()
        state = _BackendState.from_tensordict(td, check=False)
        with pytest.raises(RuntimeError, match="locked"):
            state.del_("a")
        assert "a" in td.keys()

    @pytest.mark.skipif(not _has_h5py, reason="h5py not found.")
    def test_rename_key_locked_h5_source(self, tmp_path):
        # a locked file renames, with a FutureWarning, as the file does
        td = _h5_source(tmp_path)
        td.lock_()
        state = _BackendState.from_tensordict(td, check=False)
        with pytest.warns(FutureWarning, match="renames keys of a locked"):
            assert state.rename_key_("a", "z") is state
        assert "z" in td.keys()
        assert "a" not in td.keys()
        dense = _BackendState.from_tensordict(_dense_source().lock_())
        with pytest.raises(RuntimeError, match="locked"):
            dense.rename_key_("a", "z")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
