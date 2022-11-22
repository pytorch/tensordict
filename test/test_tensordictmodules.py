# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
from functorch import make_functional, make_functional_with_buffers
from tensordict import TensorDict
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    TensorDictModule,
    TensorDictSequential,
)
from tensordict.nn.distributions import NormalParamWrapper
from tensordict.nn.probabilistic import set_interaction_mode
from torch import nn
from torch.distributions import Normal


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
    def test_stateful_probabilistic(self, lazy, interaction_mode, out_keys):
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

        tensordict_module = ProbabilisticTensorDictModule(
            module=net,
            dist_in_keys=dist_in_keys,
            sample_out_key=["out"],
            **kwargs,
        )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_interaction_mode(interaction_mode):
            tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    def test_functional(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.Linear(3, 4 * param_multiplier)

        fnet, params = make_functional(net)

        tensordict_module = TensorDictModule(
            module=fnet, in_keys=["in"], out_keys=["out"]
        )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    def test_functional_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)
        in_keys = ["in"]
        net = NormalParamWrapper(net)
        fnet, params = make_functional(net)
        tdnet = TensorDictModule(
            module=fnet, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}
        tensordict_module = ProbabilisticTensorDictModule(
            module=tdnet,
            dist_in_keys=["loc", "scale"],
            sample_out_key=["out"],
            **kwargs,
        )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    def test_functional_probabilistic_laterconstruct(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)
        in_keys = ["in"]
        net = NormalParamWrapper(net)
        tdnet = TensorDictModule(module=net, in_keys=in_keys, out_keys=["loc", "scale"])

        kwargs = {"distribution_class": Normal}
        tensordict_module = ProbabilisticTensorDictModule(
            module=tdnet,
            dist_in_keys=["loc", "scale"],
            sample_out_key=["out"],
            **kwargs,
        )
        tensordict_module, (
            params,
            buffers,
        ) = tensordict_module.make_functional_with_buffers()

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td = tensordict_module(td, params=params, buffers=buffers)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    def test_functional_with_buffer(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.BatchNorm1d(32 * param_multiplier)

        fnet, params, buffers = make_functional_with_buffers(net)

        tdmodule = TensorDictModule(module=fnet, in_keys=["in"], out_keys=["out"])

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params, buffers=buffers)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

    def test_functional_with_buffer_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.BatchNorm1d(32 * param_multiplier)
        in_keys = ["in"]
        net = NormalParamWrapper(net)
        fnet, params, buffers = make_functional_with_buffers(net)
        tdnet = TensorDictModule(
            module=fnet, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}

        tdmodule = ProbabilisticTensorDictModule(
            module=tdnet,
            dist_in_keys=["loc", "scale"],
            sample_out_key=["out"],
            **kwargs,
        )

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params, buffers=buffers)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

    def test_functional_with_buffer_probabilistic_laterconstruct(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.BatchNorm1d(32 * param_multiplier)
        in_keys = ["in"]
        net = NormalParamWrapper(net)
        tdnet = TensorDictModule(module=net, in_keys=in_keys, out_keys=["loc", "scale"])

        kwargs = {"distribution_class": Normal}
        tdmodule = ProbabilisticTensorDictModule(
            module=tdnet,
            dist_in_keys=["loc", "scale"],
            sample_out_key=["out"],
            **kwargs,
        )
        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers()

        td = TensorDict({"in": torch.randn(3, 32 * param_multiplier)}, [3])
        tdmodule(td, params=params, buffers=buffers)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 32])

    def test_vmap(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net = nn.Linear(3, 4 * param_multiplier)

        fnet, params = make_functional(net)

        tdmodule = TensorDictModule(module=fnet, in_keys=["in"], out_keys=["out"])

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, None)
        td_out = tdmodule(td, params=params, vmap=(0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    def test_vmap_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)
        net = NormalParamWrapper(net)
        in_keys = ["in"]
        fnet, params = make_functional(net)
        tdnet = TensorDictModule(
            module=fnet, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}

        tdmodule = ProbabilisticTensorDictModule(
            module=tdnet,
            dist_in_keys=["loc", "scale"],
            sample_out_key=["out"],
            **kwargs,
        )

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, None)
        td_out = tdmodule(td, params=params, vmap=(0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    def test_vmap_probabilistic_laterconstruct(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net = nn.Linear(3, 4 * param_multiplier)
        net = NormalParamWrapper(net)
        in_keys = ["in"]
        tdnet = TensorDictModule(module=net, in_keys=in_keys, out_keys=["loc", "scale"])

        kwargs = {"distribution_class": Normal}
        tdmodule = ProbabilisticTensorDictModule(
            module=tdnet,
            dist_in_keys=["loc", "scale"],
            sample_out_key=["out"],
            **kwargs,
        )
        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers()

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, buffers=buffers, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0, None)
        td_out = tdmodule(td, params=params, buffers=buffers, vmap=(0, 0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, buffers=buffers, vmap=(0, 0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    def test_nested_keys(self):
        class Net(nn.Module):
            def __init__(self, input_size=100, hidden=10):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden)
                self.fc2 = nn.Linear(hidden, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                logits = self.fc2(x)
                return torch.sigmoid(logits), logits

        module = TensorDictModule(
            Net(),
            in_keys=[("inputs", "x")],
            out_keys=[("outputs", "probabilities"), ("outputs", "logits")],
        )

        x = torch.rand(5, 100)
        tensordict = TensorDict({"inputs": TensorDict({"x": x}, [5])}, [5])

        tensordict = module(tensordict)

        assert tensordict["inputs", "x"] is x
        assert ("outputs", "probabilities") in tensordict.keys(include_nested=True)
        assert ("outputs", "logits") in tensordict.keys(include_nested=True)


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

        with pytest.raises(RuntimeError, match="Cannot call get_dist on a sequence"):
            dist, *_ = tdmodule.get_dist(td)

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
        net2 = NormalParamWrapper(net2)
        net2 = TensorDictModule(
            module=net2, in_keys=["hidden"], out_keys=["loc", "scale"]
        )

        kwargs = {"distribution_class": Normal}
        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            dummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = ProbabilisticTensorDictModule(
            module=net2,
            dist_in_keys=["loc", "scale"],
            sample_out_key=["out"],
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

        dist, *_ = tdmodule.get_dist(td)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

    def test_functional(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        fnet1, params1 = make_functional(net1)
        fdummy_net, _ = make_functional(dummy_net)
        fnet2, params2 = make_functional(net2)
        params = list(params1) + list(params2)

        tdmodule1 = TensorDictModule(fnet1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            fdummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(fnet2, in_keys=["hidden"], out_keys=["out"])
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
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        with pytest.raises(RuntimeError, match="Cannot call get_dist on a sequence"):
            dist, *_ = tdmodule.get_dist(td, params=params)

    def test_functional_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)

        fnet1, params1 = make_functional(net1)
        fdummy_net, _ = make_functional(dummy_net)
        fnet2, params2 = make_functional(net2)
        fnet2 = TensorDictModule(
            module=fnet2, in_keys=["hidden"], out_keys=["loc", "scale"]
        )
        params = list(params1) + list(params2)

        kwargs = {"distribution_class": Normal}

        tdmodule1 = TensorDictModule(fnet1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            fdummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = ProbabilisticTensorDictModule(
            fnet2, dist_in_keys=["loc", "scale"], sample_out_key=["out"], **kwargs
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
        tdmodule(td, params=params)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

        dist, *_ = tdmodule.get_dist(td, params=params)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

    def test_functional_with_buffer(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )

        fnet1, params1, buffers1 = make_functional_with_buffers(net1)
        fdummy_net, _, _ = make_functional_with_buffers(dummy_net)
        fnet2, params2, buffers2 = make_functional_with_buffers(net2)

        params = list(params1) + list(params2)
        buffers = list(buffers1) + list(buffers2)

        tdmodule1 = TensorDictModule(fnet1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            fdummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(fnet2, in_keys=["hidden"], out_keys=["out"])
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

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params, buffers=buffers)

        with pytest.raises(RuntimeError, match="Cannot call get_dist on a sequence"):
            dist, *_ = tdmodule.get_dist(td, params=params, buffers=buffers)

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

    def test_functional_with_buffer_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        dummy_net = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )
        net2 = NormalParamWrapper(net2)

        fnet1, params1, buffers1 = make_functional_with_buffers(net1)
        fdummy_net, _, _ = make_functional_with_buffers(dummy_net)
        # fnet2, params2, buffers2 = make_functional_with_buffers(net2)
        # fnet2 = TensorDictModule(fnet2, in_keys=["hidden"], out_keys=["loc", "scale"])
        net2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["loc", "scale"])
        fnet2, (params2, buffers2) = net2.make_functional_with_buffers()

        params = list(params1) + list(params2)
        buffers = list(buffers1) + list(buffers2)

        kwargs = {"distribution_class": Normal}
        tdmodule1 = TensorDictModule(fnet1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            fdummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = ProbabilisticTensorDictModule(
            fnet2, dist_in_keys=["loc", "scale"], sample_out_key=["out"], **kwargs
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

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params, buffers=buffers)

        dist, *_ = tdmodule.get_dist(td, params=params, buffers=buffers)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

    def test_functional_with_buffer_probabilistic_laterconstruct(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Sequential(nn.Linear(7, 7), nn.BatchNorm1d(7))
        net2 = nn.Sequential(
            nn.Linear(7, 7 * param_multiplier), nn.BatchNorm1d(7 * param_multiplier)
        )
        net2 = NormalParamWrapper(net2)
        net2 = TensorDictModule(net2, in_keys=["hidden"], out_keys=["loc", "scale"])

        kwargs = {"distribution_class": Normal}

        tdmodule1 = TensorDictModule(net1, in_keys=["in"], out_keys=["hidden"])
        tdmodule2 = ProbabilisticTensorDictModule(
            net2,
            dist_in_keys=["loc", "scale"],
            sample_out_key=["out"],
            **kwargs,
        )
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)

        tdmodule, (params, buffers) = tdmodule.make_functional_with_buffers()

        td = TensorDict({"in": torch.randn(3, 7)}, [3])
        tdmodule(td, params=params, buffers=buffers)

        dist, *_ = tdmodule.get_dist(td, params=params, buffers=buffers)
        assert dist.rsample().shape[: td.ndimension()] == td.shape

        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 7])

    def test_vmap(self):
        torch.manual_seed(0)
        param_multiplier = 1

        net1 = nn.Linear(3, 4)
        dummy_net = nn.Linear(4, 4)
        net2 = nn.Linear(4, 4 * param_multiplier)

        fnet1, params1 = make_functional(net1)
        fdummy_net, _ = make_functional(dummy_net)
        fnet2, params2 = make_functional(net2)
        params = params1 + params2

        tdmodule1 = TensorDictModule(fnet1, in_keys=["in"], out_keys=["hidden"])
        dummy_tdmodule = TensorDictModule(
            fdummy_net, in_keys=["hidden"], out_keys=["hidden"]
        )
        tdmodule2 = TensorDictModule(fnet2, in_keys=["hidden"], out_keys=["out"])
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

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, None)
        td_out = tdmodule(td, params=params, vmap=(0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

    def test_vmap_probabilistic(self):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        fnet1, params1 = make_functional(net1)

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)
        fnet2, params2 = make_functional(net2)
        fnet2 = TensorDictModule(fnet2, in_keys=["hidden"], out_keys=["loc", "scale"])

        params = params1 + params2

        kwargs = {"distribution_class": Normal}
        tdmodule1 = TensorDictModule(fnet1, in_keys=["in"], out_keys=["hidden"])
        tdmodule2 = ProbabilisticTensorDictModule(
            fnet2, sample_out_key=["out"], dist_in_keys=["loc", "scale"], **kwargs
        )
        tdmodule = TensorDictSequential(tdmodule1, tdmodule2)

        # vmap = True
        params = [p.repeat(10, *[1 for _ in p.shape]) for p in params]
        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        td_out = tdmodule(td, params=params, vmap=True)
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, None)
        td_out = tdmodule(td, params=params, vmap=(0, None))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

        # vmap = (0, 0)
        td_repeat = td.expand(10, *td.batch_size).clone()
        td_out = tdmodule(td_repeat, params=params, vmap=(0, 0))
        assert td_out is not td
        assert td_out.shape == torch.Size([10, 3])
        assert td_out.get("out").shape == torch.Size([10, 3, 4])

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
            sub_seq_1, (params, buffers) = sub_seq_1.make_functional_with_buffers()
            sub_seq_1(td_1, params=params, buffers=buffers)
            assert "hidden" in td_1.keys()
            assert "out" not in td_1.keys()
            td_2 = TensorDict({"hidden": torch.randn(5, 2)}, [5])
            sub_seq_2 = td_module.select_subsequence(in_keys=["hidden"])
            sub_seq_2, (params, buffers) = sub_seq_2.make_functional_with_buffers()
            sub_seq_2(td_2, params=params, buffers=buffers)
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

    @pytest.mark.parametrize("stack", [True, False])
    @pytest.mark.parametrize("functional", [True, False])
    def test_sequential_partial(self, stack, functional):
        torch.manual_seed(0)
        param_multiplier = 2

        net1 = nn.Linear(3, 4)
        if functional:
            fnet1, params1 = make_functional(net1)
        else:
            params1 = None
            fnet1 = net1

        net2 = nn.Linear(4, 4 * param_multiplier)
        net2 = NormalParamWrapper(net2)
        if functional:
            fnet2, params2 = make_functional(net2)
        else:
            fnet2 = net2
            params2 = None
        fnet2 = TensorDictModule(fnet2, in_keys=["b"], out_keys=["loc", "scale"])

        net3 = nn.Linear(4, 4 * param_multiplier)
        net3 = NormalParamWrapper(net3)
        if functional:
            fnet3, params3 = make_functional(net3)
        else:
            fnet3 = net3
            params3 = None
        fnet3 = TensorDictModule(fnet3, in_keys=["c"], out_keys=["loc", "scale"])

        kwargs = {"distribution_class": Normal}

        tdmodule1 = TensorDictModule(fnet1, in_keys=["a"], out_keys=["hidden"])
        tdmodule2 = ProbabilisticTensorDictModule(
            fnet2, sample_out_key=["out"], dist_in_keys=["loc", "scale"], **kwargs
        )
        tdmodule3 = ProbabilisticTensorDictModule(
            fnet3, sample_out_key=["out"], dist_in_keys=["loc", "scale"], **kwargs
        )
        tdmodule = TensorDictSequential(
            tdmodule1, tdmodule2, tdmodule3, partial_tolerant=True
        )

        if stack:
            td = torch.stack(
                [
                    TensorDict({"a": torch.randn(3), "b": torch.randn(4)}, []),
                    TensorDict({"a": torch.randn(3), "c": torch.randn(4)}, []),
                ],
                0,
            )
            if functional:
                tdmodule(td, params=params1 + params2 + params3)
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
                tdmodule(td, params=params1 + params2 + params3)
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

    def test_nested_keys(self):
        class Net(nn.Module):
            def __init__(self, input_size=100, hidden_size=50, output_size=10):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        class Masker(nn.Module):
            def forward(self, x, mask):
                return torch.softmax(x * mask, dim=1)

        net = TensorDictModule(
            Net(), in_keys=[("input", "x")], out_keys=[("intermediate", "x")]
        )
        masker = TensorDictModule(
            Masker(),
            in_keys=[("intermediate", "x"), ("input", "mask")],
            out_keys=[("output", "probabilities")],
        )
        module = TensorDictSequential(net, masker)

        x = torch.rand(32, 100)
        mask = torch.randint(low=0, high=2, size=(32, 10), dtype=torch.uint8)
        tensordict = TensorDict(
            {"input": TensorDict({"x": x, "mask": mask}, batch_size=[32])},
            batch_size=[32],
        )
        tensordict = module(tensordict)

        assert tensordict["input", "x"] is x
        assert set(tensordict.keys(include_nested=True)) == {
            "input",
            ("input", "x"),
            ("input", "mask"),
            "intermediate",
            ("intermediate", "x"),
            "output",
            ("output", "probabilities"),
        }


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
