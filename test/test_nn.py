# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch

from tensordict import TensorDict
from tensordict.nn import (
    dispatch,
    probabilistic as nn_probabilistic,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from tensordict.nn.distributions import NormalParamExtractor, NormalParamWrapper
from tensordict.nn.functional_modules import make_functional
from tensordict.nn.probabilistic import set_interaction_mode
from torch import nn
from torch.distributions import Normal

try:
    import functorch  # noqa
    from functorch import (
        make_functional_with_buffers as make_functional_functorch,
        vmap,
    )

    _has_functorch = True
    FUNCTORCH_ERR = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERR = str(err)


class TestTDModule:
    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful(self, lazy):
        torch.manual_seed(0)
        param_multiplier = 1
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        tensordict_module = TensorDictModule(
            module=net, in_keys=["in"], out_keys=["out"]
        )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.parametrize("out_keys", [["loc", "scale"], ["loc_1", "scale_1"]])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("interaction_mode", ["mode", "random", None])
    def test_stateful_probabilistic_deprec(self, lazy, interaction_mode, out_keys):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        in_keys = ["in"]
        net = TensorDictModule(
            module=NormalParamWrapper(net), in_keys=in_keys, out_keys=out_keys
        )

        kwargs = {"distribution_class": Normal}
        if out_keys == ["loc", "scale"]:
            dist_in_keys = ["loc", "scale"]
        elif out_keys == ["loc_1", "scale_1"]:
            dist_in_keys = {"loc": "loc_1", "scale": "scale_1"}
        else:
            raise NotImplementedError

        prob_module = ProbabilisticTensorDictModule(
            in_keys=dist_in_keys, out_keys=["out"], **kwargs
        )

        tensordict_module = ProbabilisticTensorDictSequential(net, prob_module)

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_interaction_mode(interaction_mode):
            tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.parametrize("out_keys", [["low"], ["low1"], [("stuff", "low1")]])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("max_dist", [1.0, 2.0])
    @pytest.mark.parametrize("interaction_mode", ["mode", "random", None])
    def test_stateful_probabilistic_kwargs(
        self, lazy, interaction_mode, out_keys, max_dist
    ):
        torch.manual_seed(0)
        if lazy:
            net = nn.LazyLinear(4)
        else:
            net = nn.Linear(3, 4)

        in_keys = ["in"]
        net = TensorDictModule(module=net, in_keys=in_keys, out_keys=out_keys)

        kwargs = {
            "distribution_class": torch.distributions.Uniform,
            "distribution_kwargs": {"high": max_dist},
        }
        if out_keys == ["low"]:
            dist_in_keys = ["low"]
        else:
            dist_in_keys = {"low": out_keys[0]}

        prob_module = ProbabilisticTensorDictModule(
            in_keys=dist_in_keys, out_keys=["out"], **kwargs
        )

        tensordict_module = ProbabilisticTensorDictSequential(net, prob_module)

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_interaction_mode(interaction_mode):
            tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.parametrize(
        "out_keys",
        [
            ["loc", "scale"],
            ["loc_1", "scale_1"],
            [("params_td", "loc_1"), ("scale_1",)],
        ],
    )
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("interaction_mode", ["mode", "random", None])
    def test_stateful_probabilistic(self, lazy, interaction_mode, out_keys):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        in_keys = ["in"]
        net = TensorDictModule(module=net, in_keys=in_keys, out_keys=["params"])
        normal_params = TensorDictModule(
            NormalParamExtractor(), in_keys=["params"], out_keys=out_keys
        )

        kwargs = {"distribution_class": Normal}
        if out_keys == ["loc", "scale"]:
            dist_in_keys = ["loc", "scale"]
        else:
            dist_in_keys = {"loc": out_keys[0], "scale": out_keys[1]}

        prob_module = ProbabilisticTensorDictModule(
            in_keys=dist_in_keys, out_keys=["out"], **kwargs
        )

        tensordict_module = ProbabilisticTensorDictSequential(
            net, normal_params, prob_module
        )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_interaction_mode(interaction_mode):
            tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_before(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.Linear(3, 4 * param_multiplier)

        params = make_functional(net)

        tensordict_module = TensorDictModule(
            module=net, in_keys=["in"], out_keys=["out"]
        )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.Linear(3, 4 * param_multiplier)

        tensordict_module = TensorDictModule(
            module=net, in_keys=["in"], out_keys=["out"]
        )

        params = make_functional(tensordict_module)

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_functorch(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.Linear(3, 4 * param_multiplier)

        tensordict_module = TensorDictModule(
            module=net, in_keys=["in"], out_keys=["out"]
        )

        tensordict_module, params, buffers = make_functional_functorch(
            tensordict_module
        )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(params, buffers, td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_probabilistic_deprec(self):
        torch.manual_seed(0)
        param_multiplier = 2

        tdnet = TensorDictModule(
            module=NormalParamWrapper(nn.Linear(3, 4 * param_multiplier)),
            in_keys=["in"],
            out_keys=["loc", "scale"],
        )

        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"], out_keys=["out"], **kwargs
        )

        tensordict_module = ProbabilisticTensorDictSequential(tdnet, prob_module)
        params = make_functional(tensordict_module)

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        tdnet = TensorDictModule(
            module=nn.Linear(3, 4 * param_multiplier),
            in_keys=["in"],
            out_keys=["params"],
        )
        normal_params = TensorDictModule(
            NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"], out_keys=["out"], **kwargs
        )

        tensordict_module = ProbabilisticTensorDictSequential(
            tdnet, normal_params, prob_module
        )
        params = make_functional(tensordict_module)

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_with_buffer(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.BatchNorm1d(32 * param_multiplier)
        params = make_functional(net)

        tdmodule = TensorDictModule(module=net, in_keys=["in"], out_keys=["out"])

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_with_buffer_probabilistic_deprec(self):
        torch.manual_seed(0)
        param_multiplier = 2

        tdnet = TensorDictModule(
            module=NormalParamWrapper(nn.BatchNorm1d(32 * param_multiplier)),
            in_keys=["in"],
            out_keys=["loc", "scale"],
        )

        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"], out_keys=["out"], **kwargs
        )

        tdmodule = ProbabilisticTensorDictSequential(tdnet, prob_module)
        params = make_functional(tdmodule)

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_with_buffer_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        tdnet = TensorDictModule(
            module=nn.BatchNorm1d(32 * param_multiplier),
            in_keys=["in"],
            out_keys=["params"],
        )
        normal_params = TensorDictModule(
            NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"], out_keys=["out"], **kwargs
        )

        tdmodule = ProbabilisticTensorDictSequential(tdnet, normal_params, prob_module)
        params = make_functional(tdmodule)

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_vmap(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.Linear(3, 4 * param_multiplier)
        tdmodule = TensorDictModule(module=net, in_keys=["in"], out_keys=["out"])

        params = make_functional(tdmodule)

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
        assert td_out is not td_repeat
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_vmap_probabilistic_deprec(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net = NormalParamWrapper(nn.Linear(3, 4 * param_multiplier))

        tdnet = TensorDictModule(module=net, in_keys=["in"], out_keys=["loc", "scale"])

        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"], out_keys=["out"], **kwargs
        )

        tdmodule = ProbabilisticTensorDictSequential(tdnet, prob_module)
        params = make_functional(tdmodule)

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
        assert td_out is not td_repeat
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_vmap_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)

        tdnet = TensorDictModule(module=net, in_keys=["in"], out_keys=["params"])
        normal_params = TensorDictModule(
            NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"], out_keys=["out"], **kwargs
        )

        tdmodule = ProbabilisticTensorDictSequential(tdnet, normal_params, prob_module)
        params = make_functional(tdmodule)

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
        assert td_out is not td_repeat
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    def test_dispatch(self):
        tdm = TensorDictModule(nn.Linear(1, 1), ["a"], ["b"])
        td = TensorDict({"a": torch.zeros(1, 1)}, 1)
        tdm(td)
        out = tdm(a=torch.zeros(1, 1))
        assert (out == td["b"]).all()

    def test_dispatch_nested(self):
        tdm = TensorDictModule(nn.Linear(1, 1), [("a", "c")], [("b", "d")])
        td = TensorDict({("a", "c"): torch.zeros(1, 1)}, [1])
        tdm(td)
        out = tdm(a_c=torch.zeros(1, 1))
        assert (out == td["b", "d"]).all()

    def test_dispatch_nested_confusing(self):
        tdm = TensorDictModule(nn.Linear(1, 1), [("a_1", "c")], [("b_2", "d")])
        td = TensorDict({("a_1", "c"): torch.zeros(1, 1)}, [1])
        tdm(td)
        out = tdm(a_1_c=torch.zeros(1, 1))
        assert (out == td["b_2", "d"]).all()

    def test_dispatch_nested_args(self):
        class MyModuleNest(nn.Module):
            in_keys = [("a", "c"), "d"]
            out_keys = ["b"]

            @dispatch(separator="_")
            def forward(self, tensordict):
                tensordict["b"] = tensordict["a", "c"] + tensordict["d"]
                return tensordict

        module = MyModuleNest()
        (b,) = module(torch.zeros(1, 2), d=torch.ones(1, 2))
        assert (b == 1).all()
        with pytest.raises(RuntimeError, match="Duplicated argument"):
            module(torch.zeros(1, 2), a_c=torch.ones(1, 2))

    def test_dispatch_nested_extra_args(self):
        class MyModuleNest(nn.Module):
            in_keys = [("a", "c"), "d"]
            out_keys = ["b"]

            @dispatch(separator="_")
            def forward(self, tensordict, other):
                tensordict["b"] = tensordict["a", "c"] + tensordict["d"] + other
                return tensordict

        module = MyModuleNest()
        other = 1
        (b,) = module(torch.zeros(1, 2), torch.ones(1, 2), other)
        assert (b == 2).all()

    def test_dispatch_nested_sep(self):
        class MyModuleNest(nn.Module):
            in_keys = [("a", "c")]
            out_keys = ["b"]

            @dispatch(separator="sep")
            def forward(self, tensordict):
                tensordict["b"] = tensordict["a", "c"] + 1
                return tensordict

        module = MyModuleNest()
        (b,) = module(asepc=torch.zeros(1, 2))
        assert (b == 1).all()

    @pytest.mark.parametrize("source", ["keys_in", [("a", "c")]])
    def test_dispatch_nested_source(self, source):
        class MyModuleNest(nn.Module):
            keys_in = [("a", "c")]
            out_keys = ["b"]

            @dispatch(separator="sep", source=source)
            def forward(self, tensordict):
                tensordict["b"] = tensordict["a", "c"] + 1
                return tensordict

        module = MyModuleNest()
        (b,) = module(asepc=torch.zeros(1, 2))
        assert (b == 1).all()

    @pytest.mark.parametrize("dest", ["other", ["b"]])
    def test_dispatch_nested_dest(self, dest):
        class MyModuleNest(nn.Module):
            in_keys = [("a", "c")]
            other = ["b"]

            @dispatch(separator="sep", dest=dest)
            def forward(self, tensordict):
                tensordict["b"] = tensordict["a", "c"] + 1
                return tensordict

        module = MyModuleNest()
        (b,) = module(asepc=torch.zeros(1, 2))
        assert (b == 1).all()

    def test_dispatch_multi(self):
        tdm = TensorDictSequential(
            TensorDictModule(nn.Linear(1, 1), [("a", "c")], [("b", "d")]),
            TensorDictModule(nn.Linear(1, 1), [("a", "c")], ["e"]),
        )
        td = TensorDict({("a", "c"): torch.zeros(1, 1)}, [1])
        tdm(td)
        out1, out2 = tdm(a_c=torch.zeros(1, 1))
        assert (out1 == td["b", "d"]).all()
        assert (out2 == td["e"]).all()

    def test_dispatch_module_with_additional_parameters(self):
        class MyModule(nn.Identity):
            def forward(self, input, c):
                return input

        m = MyModule()
        tdm = TensorDictModule(m, ["a"], ["b"])
        tdm(a=torch.zeros(1, 1), c=1)


class TestTDSequence:
    def test_key_exclusion(self):
        module1 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        module2 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["key1", "key3"], out_keys=["key1"]
        )
        module3 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["foo1", "key3"], out_keys=["key2"]
        )
        seq = TensorDictSequential(module1, module2, module3)
        assert set(seq.in_keys) == {"key1", "key2", "key3"}
        assert set(seq.out_keys) == {"foo1", "key1", "key2"}

    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful(self, lazy):
        torch.manual_seed(0)
        param_multiplier = 1
        if lazy:
            net1 = nn.LazyLinear(4)
            dummy_net = nn.LazyLinear(4)
            net2 = nn.LazyLinear(4 * param_multiplier)
        else:
            net1 = nn.Linear(3, 4)
            dummy_net = nn.Linear(4, 4)
            net2 = nn.Linear(4, 4 * param_multiplier)

        kwargs = {}
        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(
            module=net2,
            in_keys=["hidden"],
            out_keys=["out"],
            **kwargs,
        )
        tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful_probabilistic_deprec(self, lazy):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net1 = nn.LazyLinear(4)
            dummy_net = nn.LazyLinear(4)
            net2 = nn.LazyLinear(4 * param_multiplier)
        else:
            net1 = nn.Linear(3, 4)
            dummy_net = nn.Linear(4, 4)
            net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)

        kwargs = {"distribution_class": Normal}
        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(
            net2, in_keys=["hidden"], out_keys=["loc", "scale"]
        )

        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"], out_keys=["out"], **kwargs
        )
        tdmodule = ProbabilisticTensorDictSequential(
            tdmodule1, dummy_tdmodule, tdmodule2, prob_module
        )

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 4
        tdmodule[1] = tdmodule2
        tdmodule[2] = prob_module
        assert len(tdmodule) == 4

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 4
        del tdmodule[3]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is prob_module

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist = tdmodule.get_dist(td)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful_probabilistic(self, lazy):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net1 = nn.LazyLinear(4)
            dummy_net = nn.LazyLinear(4)
            net2 = nn.LazyLinear(4 * param_multiplier)
        else:
            net1 = nn.Linear(3, 4)
            dummy_net = nn.Linear(4, 4)
            net2 = nn.Linear(4, 4 * param_multiplier)

        kwargs = {"distribution_class": Normal}
        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["params"])

        normal_params = TensorDictModule(
            NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        )
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"], out_keys=["out"], **kwargs
        )
        tdmodule = ProbabilisticTensorDictSequential(
            tdmodule1, dummy_tdmodule, tdmodule2, normal_params, prob_module
        )

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 5
        tdmodule[1] = tdmodule2
        tdmodule[2] = normal_params
        tdmodule[3] = prob_module
        assert len(tdmodule) == 5

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 5
        del tdmodule[4]
        assert len(tdmodule) == 4

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is normal_params
        assert tdmodule[3] is prob_module

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist = tdmodule.get_dist(td)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["out"])
        tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        params = make_functional(tdmodule)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        params["module", "1"] = params["module", "2"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        del params["module", "2"]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td, params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_functorch(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["out"])
        tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        ftdmodule, params, buffers = make_functional_functorch(tdmodule)

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        ftdmodule(params, buffers, td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_probabilistic_deprec(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(
            net2, in_keys=["hidden"], out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            out_keys=["out"],
            in_keys=["loc", "scale"],
            **kwargs,
        )
        tdmodule = ProbabilisticTensorDictSequential(
            tdmodule1, dummy_tdmodule, tdmodule2, prob_module
        )

        params = make_functional(tdmodule, funs_to_decorate=["forward", "get_dist"])

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 4
        tdmodule[1] = tdmodule2
        tdmodule[2] = prob_module
        params["module", "1"] = params["module", "2"]
        params["module", "2"] = params["module", "3"]
        assert len(tdmodule) == 4

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 4
        del tdmodule[3]
        del params["module", "3"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule.module, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is prob_module

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist = tdmodule.get_dist(td, params=params)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["params"])

        normal_params = TensorDictModule(
            NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        )
        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            out_keys=["out"],
            in_keys=["loc", "scale"],
            **kwargs,
        )
        tdmodule = ProbabilisticTensorDictSequential(
            tdmodule1, dummy_tdmodule, tdmodule2, normal_params, prob_module
        )

        params = make_functional(tdmodule, funs_to_decorate=["forward", "get_dist"])

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 5
        tdmodule[1] = tdmodule2
        tdmodule[2] = normal_params
        tdmodule[3] = prob_module
        params["module", "1"] = params["module", "2"]
        params["module", "2"] = params["module", "3"]
        params["module", "3"] = params["module", "4"]
        assert len(tdmodule) == 5

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 5
        del tdmodule[4]
        del params["module", "4"]
        assert len(tdmodule) == 4

        assert hasattr(tdmodule.module, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is normal_params
        assert tdmodule[3] is prob_module

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist = tdmodule.get_dist(td, params=params)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_with_buffer(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["out"])
        tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        params = make_functional(tdmodule)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        params["module", "1"] = params["module", "2"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        del params["module", "2"]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params)

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_with_buffer_probabilistic_deprec(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )
        net2 = NormalParamWrapper(net2)

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(
            net2, in_keys=["hidden"], out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["out"],
            **kwargs,
        )

        tdmodule = ProbabilisticTensorDictSequential(
            tdmodule1, dummy_tdmodule, tdmodule2, prob_module
        )

        params = make_functional(tdmodule, ["forward", "get_dist"])

        assert hasattr(tdmodule.module, "__setitem__")
        assert len(tdmodule.module) == 4
        tdmodule[1] = tdmodule2
        tdmodule[2] = prob_module
        params["module", "1"] = params["module", "2"]
        params["module", "2"] = params["module", "3"]
        assert len(tdmodule) == 4

        assert hasattr(tdmodule.module, "__delitem__")
        assert len(tdmodule.module) == 4
        del tdmodule.module[3]
        del params["module", "3"]
        assert len(tdmodule.module) == 3

        assert hasattr(tdmodule.module, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is prob_module

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params)

        dist = tdmodule.get_dist(td, params=params)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_functional_with_buffer_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["params"])

        normal_params = TensorDictModule(
            NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        )
        kwargs = {"distribution_class": Normal}
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["out"],
            **kwargs,
        )

        tdmodule = ProbabilisticTensorDictSequential(
            tdmodule1, dummy_tdmodule, tdmodule2, normal_params, prob_module
        )

        params = make_functional(tdmodule, ["forward", "get_dist"])

        assert hasattr(tdmodule.module, "__setitem__")
        assert len(tdmodule.module) == 5
        tdmodule[1] = tdmodule2
        tdmodule[2] = normal_params
        tdmodule[3] = prob_module
        params["module", "1"] = params["module", "2"]
        params["module", "2"] = params["module", "3"]
        params["module", "3"] = params["module", "4"]
        assert len(tdmodule) == 5

        assert hasattr(tdmodule.module, "__delitem__")
        assert len(tdmodule.module) == 5
        del tdmodule.module[4]
        del params["module", "4"]
        assert len(tdmodule.module) == 4

        assert hasattr(tdmodule.module, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2
        assert tdmodule[2] is normal_params
        assert tdmodule[3] is prob_module

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params)

        dist = tdmodule.get_dist(td, params=params)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_vmap(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["out"])
        tdmodule = TensorDictSequential(tdmodule1, dummy_tdmodule, tdmodule2)

        params = make_functional(tdmodule)

        assert hasattr(tdmodule, "__setitem__")
        assert len(tdmodule) == 3
        tdmodule[1] = tdmodule2
        params["module", "1"] = params["module", "2"]
        assert len(tdmodule) == 3

        assert hasattr(tdmodule, "__delitem__")
        assert len(tdmodule) == 3
        del tdmodule[2]
        del params["module", "2"]
        assert len(tdmodule) == 2

        assert hasattr(tdmodule, "__getitem__")
        assert tdmodule[0] is tdmodule1
        assert tdmodule[1] is tdmodule2

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
        assert td_out is not td_repeat
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_vmap_probabilistic_deprec(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)

        kwargs = {"distribution_class": Normal}
        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        tdmodule2 = TensorDictModule(
            net2, in_keys=["hidden"], out_keys=["loc", "scale"]
        )
        tdmodule = ProbabilisticTensorDictSequential(
            tdmodule1,
            tdmodule2,
            ProbabilisticTensorDictModule(
                out_keys=["out"],
                in_keys=["loc", "scale"],
                **kwargs,
            ),
        )

        params = make_functional(tdmodule)

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
        assert td_out is not td_repeat
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    def test_vmap_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        kwargs = {"distribution_class": Normal}
        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        tdmodule2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["params"])
        normal_params = TensorDictModule(
            NormalParamExtractor(), in_keys=["params"], out_keys=["loc", "scale"]
        )
        tdmodule = ProbabilisticTensorDictSequential(
            tdmodule1,
            tdmodule2,
            normal_params,
            ProbabilisticTensorDictModule(
                out_keys=["out"], in_keys=["loc", "scale"], **kwargs
            ),
        )

        params = make_functional(tdmodule)

        # vmap = True
        params = params.expand(10)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = vmap(tdmodule, (None, 0))(td, params)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_repeat = td.expand(10, *td.batch_size)
        td_out = vmap(tdmodule, (0, 0))(td_repeat, params)
        assert td_out is not td_repeat
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("functional", [True, False])
    def test_submodule_sequence(self, functional):
        td_module_1 = TensorDictModule(
            nn.Linear(3, 2), in_keys=["in"], out_keys=["hidden"]
        )
        td_module_2 = TensorDictModule(
            nn.Linear(2, 4), in_keys=["hidden"], out_keys=["out"]
        )
        td_module = TensorDictSequential(td_module_1, td_module_2)

        if functional:
            td_1 = TensorDict({"in": torch.randn(5, 3)}, [5])
            sub_seq_1 = td_module.select_subsequence(out_keys=["hidden"])
            params = make_functional(sub_seq_1)
            sub_seq_1(td_1, params=params)
            assert "hidden" in td_1.keys()
            assert "out" not in td_1.keys()
            td_2 = TensorDict({"hidden": torch.randn(5, 2)}, [5])
            sub_seq_2 = td_module.select_subsequence(in_keys=["hidden"])
            params = make_functional(sub_seq_2)
            sub_seq_2(td_2, params=params)
            assert "out" in td_2.keys()
            assert td_2.get("out").shape == torch.Size([5, 4])
        else:
            td_1 = TensorDict({"in": torch.randn(5, 3)}, [5])
            sub_seq_1 = td_module.select_subsequence(out_keys=["hidden"])
            sub_seq_1(td_1)
            assert "hidden" in td_1.keys()
            assert "out" not in td_1.keys()
            td_2 = TensorDict({"hidden": torch.randn(5, 2)}, [5])
            sub_seq_2 = td_module.select_subsequence(in_keys=["hidden"])
            sub_seq_2(td_2)
            assert "out" in td_2.keys()
            assert td_2.get("out").shape == torch.Size([5, 4])

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("stack", [True, False])
    @pytest.mark.parametrize("functional", [True, False])
    def test_sequential_partial_deprec(self, stack, functional):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)
        net2 = TensorDictModule(net2, in_keys=["b"], out_keys=["loc", "scale"])

        net3 = nn.Linear(4, 4 * param_multiplier)
        net3 = NormalParamWrapper(net3)
        net3 = TensorDictModule(net3, in_keys=["c"], out_keys=["loc", "scale"])

        kwargs = {"distribution_class": Normal}

        tdmodule1 = TensorDictModule(net1, in_keys=["a"], out_keys=["hidden"])
        tdmodule2 = ProbabilisticTensorDictSequential(
            net2,
            ProbabilisticTensorDictModule(
                out_keys=["out"], in_keys=["loc", "scale"], **kwargs
            ),
        )
        tdmodule3 = ProbabilisticTensorDictSequential(
            net3,
            ProbabilisticTensorDictModule(
                out_keys=["out"], in_keys=["loc", "scale"], **kwargs
            ),
        )
        tdmodule = TensorDictSequential(
            tdmodule1, tdmodule2, tdmodule3, partial_tolerant=True
        )

        if functional:
            params = make_functional(tdmodule)
        else:
            params = None

        if stack:
            td = torch.stack(
                [
                    TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, []),
                    TensorDict({"a": torch.randn(3), "c": torch.randn(4)}, []),
                ],
                0,
            )
            if functional:
                tdmodule(td, params=params)
            else:
                tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert td["out"].shape[0] == 2
            assert td["loc"].shape[0] == 2
            assert td["scale"].shape[0] == 2
            assert "b" not in td.keys()
            assert "b" in td[0].keys()
        else:
            td = TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, [])
            if functional:
                tdmodule(td, params=params)
            else:
                tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert "b" in td.keys()

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found: err={FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("stack", [True, False])
    @pytest.mark.parametrize("functional", [True, False])
    def test_sequential_partial(self, stack, functional):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = TensorDictModule(net2, in_keys=["b"], out_keys=["params2"])

        net3 = nn.Linear(4, 4 * param_multiplier)
        net3 = TensorDictModule(net3, in_keys=["c"], out_keys=["params3"])

        kwargs = {"distribution_class": Normal}

        tdmodule1 = TensorDictModule(net1, in_keys=["a"], out_keys=["hidden"])
        tdmodule2 = ProbabilisticTensorDictSequential(
            net2,
            TensorDictModule(
                NormalParamExtractor(), in_keys=["params2"], out_keys=["loc", "scale"]
            ),
            ProbabilisticTensorDictModule(
                out_keys=["out"], in_keys=["loc", "scale"], **kwargs
            ),
        )
        tdmodule3 = ProbabilisticTensorDictSequential(
            net3,
            TensorDictModule(
                NormalParamExtractor(), in_keys=["params3"], out_keys=["loc", "scale"]
            ),
            ProbabilisticTensorDictModule(
                out_keys=["out"], in_keys=["loc", "scale"], **kwargs
            ),
        )
        tdmodule = TensorDictSequential(
            tdmodule1, tdmodule2, tdmodule3, partial_tolerant=True
        )

        if functional:
            params = make_functional(tdmodule)
        else:
            params = None

        if stack:
            td = torch.stack(
                [
                    TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, []),
                    TensorDict({"a": torch.randn(3), "c": torch.randn(4)}, []),
                ],
                0,
            )
            if functional:
                tdmodule(td, params=params)
            else:
                tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert td["out"].shape[0] == 2
            assert td["loc"].shape[0] == 2
            assert td["scale"].shape[0] == 2
            assert "b" not in td.keys()
            assert "b" in td[0].keys()
        else:
            td = TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, [])
            if functional:
                tdmodule(td, params=params)
            else:
                tdmodule(td)
            assert "loc" in td.keys()
            assert "scale" in td.keys()
            assert "out" in td.keys()
            assert "b" in td.keys()

    def test_subsequence_weight_update(self):
        td_module_1 = TensorDictModule(
            nn.Linear(3, 2), in_keys=["in"], out_keys=["hidden"]
        )
        td_module_2 = TensorDictModule(
            nn.Linear(2, 4), in_keys=["hidden"], out_keys=["out"]
        )
        td_module = TensorDictSequential(td_module_1, td_module_2)

        td_1 = TensorDict({"in": torch.randn(5, 3)}, [5])
        sub_seq_1 = td_module.select_subsequence(out_keys=["hidden"])
        copy = sub_seq_1[0].module.weight.clone()

        opt = torch.optim.SGD(td_module.parameters(), lr=0.1)
        opt.zero_grad()
        td_1 = td_module(td_1)
        td_1["out"].mean().backward()
        opt.step()

        assert not torch.allclose(copy, sub_seq_1[0].module.weight)
        assert torch.allclose(td_module[0].module.weight, sub_seq_1[0].module.weight)


@pytest.mark.parametrize("mode", ["random", "mode"])
class TestSIM:
    def test_cm(self, mode):
        with set_interaction_mode(mode):
            assert nn_probabilistic._INTERACTION_MODE == mode

    def test_dec(self, mode):
        @set_interaction_mode(mode)
        def dummy():
            assert nn_probabilistic._INTERACTION_MODE == mode

        dummy()


def test_probabilistic_sequential_type_checks():
    td_module_1 = TensorDictModule(nn.Linear(3, 2), in_keys=["in"], out_keys=["hidden"])
    td_module_2 = TensorDictModule(
        nn.Linear(2, 4), in_keys=["hidden"], out_keys=["out"]
    )
    with pytest.raises(
        TypeError,
        match="The final module passed to ProbabilisticTensorDictSequential",
    ):
        ProbabilisticTensorDictSequential(td_module_1, td_module_2)


def test_keyerr_msg():
    module = TensorDictModule(nn.Linear(2, 3), in_keys=["a"], out_keys=["b"])
    with pytest.raises(
        KeyError, match="Some tensors that are necessary for the module call"
    ):
        module(TensorDict({"c": torch.randn(())}, []))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
