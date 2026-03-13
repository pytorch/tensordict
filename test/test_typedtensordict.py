# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

try:
    from typing import NotRequired
except ImportError:
    from typing_extensions import NotRequired

import pytest
import torch
from tensordict import TensorDict, TypedTensorDict
from torch import Tensor


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
        assert isinstance(state, TensorDict)
        assert state.batch_size == torch.Size([5])

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
        assert isinstance(obs, TensorDict)

    def test_field_accumulation(self):
        assert PredictorState.__expected_keys__ == frozenset({"eta", "X", "beta"})
        assert ObservedState.__expected_keys__ == frozenset(
            {"eta", "X", "beta", "y", "mu", "noise"}
        )
        assert SurvivalState.__expected_keys__ == frozenset(
            {
                "eta", "X", "beta", "y", "mu", "noise",
                "event_time", "indicator", "observed_time",
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

    def test_to_dict(self, state):
        d = state.to_dict()
        assert set(d.keys()) == {"eta", "X", "beta"}

    def test_select(self, state):
        sel = state.select("eta", "X")
        assert set(sel.keys()) == {"eta", "X"}

    def test_exclude(self, state):
        exc = state.exclude("beta")
        assert set(exc.keys()) == {"eta", "X"}


# ---------------------------------------------------------------------------
# Class options
# ---------------------------------------------------------------------------

class TestClassOptions:
    def test_shadow_blocks_by_default(self):
        with pytest.raises(AttributeError, match="shadows a TensorDict attribute"):
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

    def test_safe_shadow_data_without_option(self):
        """'data' is in _SAFE_SHADOW_NAMES so it works without shadow=True."""
        class WithData(TypedTensorDict):
            data: Tensor
            x: Tensor

        wd = WithData(data=torch.randn(3, 2), x=torch.randn(3, 4), batch_size=[3])
        assert wd.data.shape == (3, 2)
        assert wd.x.shape == (3, 4)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
