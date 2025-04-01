# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import copy
import functools
import os
import pickle
import unittest
import weakref
from collections import OrderedDict
from collections.abc import MutableSequence

import pytest
import torch

from tensordict import (
    is_tensor_collection,
    NonTensorData,
    NonTensorStack,
    set_list_to_stack,
    tensorclass,
    TensorDict,
)
from tensordict._C import unravel_key_list
from tensordict.nn import (
    as_tensordict_module,
    dispatch,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModuleBase,
    TensorDictParams,
    TensorDictSequential,
)
from tensordict.nn.common import TensorDictModule, TensorDictModuleWrapper
from tensordict.nn.distributions import (
    AddStateIndependentNormalScale,
    Delta,
    NormalParamExtractor,
)
from tensordict.nn.distributions.composite import CompositeDistribution
from tensordict.nn.ensemble import EnsembleModule
from tensordict.nn.probabilistic import (
    interaction_type,
    InteractionType,
    set_interaction_type,
)
from tensordict.nn.utils import (
    _set_dispatch_td_nn_modules,
    composite_lp_aggregate,
    set_composite_lp_aggregate,
    set_skip_existing,
    skip_existing,
)

from torch import distributions, nn
from torch.distributions import Categorical, Normal
from torch.utils._pytree import tree_map

try:
    import functorch  # noqa
    from functorch import make_functional_with_buffers as make_functional_functorch

    try:
        from torch import vmap
    except ImportError:
        from functorch import vmap  # noqa: TOR103

    _has_functorch = True
    FUNCTORCH_ERR = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERR = str(err)
try:
    from torch.nn.parameter import Buffer
except ImportError:
    from tensordict.utils import Buffer


# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:You are using `torch.load` with `weights_only=False`"
    ),
    pytest.mark.filterwarnings("ignore:enable_nested_tensor is True"),
    pytest.mark.filterwarnings(
        "ignore:`include_sum` wasn't set when building the `CompositeDistribution`"
    ),
    pytest.mark.filterwarnings(
        "ignore:`inplace` wasn't set when building the `CompositeDistribution`"
    ),
]

PYTORCH_TEST_FBCODE = os.getenv("PYTORCH_TEST_FBCODE")
if PYTORCH_TEST_FBCODE:
    pytestmark.append(
        pytest.mark.filterwarnings("ignore:aggregate_probabilities"),
    )
    pytestmark.append(
        pytest.mark.filterwarnings("ignore:include_sum"),
    )
    pytestmark.append(
        pytest.mark.filterwarnings("ignore:inplace"),
    )


class TestInteractionType:

    @pytest.mark.parametrize(
        "str_and_expected_type",
        [
            ("mode", InteractionType.MODE),
            ("deterministic", InteractionType.DETERMINISTIC),
            ("MEDIAN", InteractionType.MEDIAN),
            ("Mean", InteractionType.MEAN),
            ("RanDom", InteractionType.RANDOM),
        ],
    )
    def test_from_str_correct_conversion(self, str_and_expected_type):
        type_str, expected_type = str_and_expected_type
        assert InteractionType.from_str(type_str) == expected_type

    @pytest.mark.parametrize("unsupported_type_str", ["foo"])
    def test_from_str_correct_raise(self, unsupported_type_str):
        with pytest.raises(ValueError, match=" is not a valid InteractionType"):
            InteractionType.from_str(unsupported_type_str)

    dist_partials = {
        "Bernoulli": functools.partial(
            distributions.Bernoulli, probs=torch.tensor([0.5, 0.5])
        ),
        "Beta": functools.partial(
            distributions.Beta, concentration1=1.0, concentration0=1.0
        ),
        "Binomial": functools.partial(
            distributions.Binomial, total_count=1, probs=torch.tensor([0.5, 0.5])
        ),
        "Categorical": functools.partial(
            distributions.Categorical, probs=torch.tensor([0.5, 0.5])
        ),
        "Cauchy": functools.partial(distributions.Cauchy, loc=0.0, scale=1.0),
        "Chi2": functools.partial(distributions.Chi2, df=1),
        "ContinuousBernoulli": functools.partial(
            distributions.ContinuousBernoulli, probs=torch.tensor([0.5, 0.5])
        ),
        "Dirichlet": functools.partial(
            distributions.Dirichlet, concentration=torch.tensor([0.5, 0.5])
        ),
        "Exponential": functools.partial(distributions.Exponential, rate=1.0),
        "FisherSnedecor": functools.partial(distributions.FisherSnedecor, df1=1, df2=1),
        "Gamma": functools.partial(distributions.Gamma, concentration=1.0, rate=1.0),
        "Geometric": functools.partial(
            distributions.Geometric, probs=torch.tensor([0.5, 0.5])
        ),
        "Gumbel": functools.partial(distributions.Gumbel, loc=0.0, scale=1.0),
        "HalfCauchy": functools.partial(distributions.HalfCauchy, scale=1.0),
        "HalfNormal": functools.partial(distributions.HalfNormal, scale=1.0),
        "InverseGamma": functools.partial(
            distributions.InverseGamma, concentration=1.0, rate=1.0
        ),
        "Kumaraswamy": functools.partial(
            distributions.Kumaraswamy, concentration1=1.0, concentration0=1.0
        ),
        "LKJCholesky": functools.partial(distributions.LKJCholesky, 3, 5),
        "Laplace": functools.partial(distributions.Laplace, loc=0.0, scale=1.0),
        "LogNormal": functools.partial(distributions.LogNormal, loc=0.0, scale=1.0),
        "LogisticNormal": functools.partial(
            distributions.LogisticNormal, loc=0.0, scale=1.0
        ),
        "LowRankMultivariateNormal": functools.partial(
            distributions.LowRankMultivariateNormal,
            loc=torch.zeros(2),
            cov_factor=torch.tensor([[1.0], [0.0]]),
            cov_diag=torch.ones(2),
        ),
        "MixtureSameFamily": functools.partial(
            distributions.MixtureSameFamily,
            mixture_distribution=distributions.Categorical(
                torch.ones(
                    5,
                )
            ),
            component_distribution=distributions.Normal(
                torch.randn(
                    5,
                ),
                torch.rand(
                    5,
                ),
            ),
        ),
        "Multinomial": functools.partial(
            distributions.Multinomial, total_count=1, probs=torch.tensor([0.5, 0.5])
        ),
        "MultivariateNormal": functools.partial(
            distributions.MultivariateNormal,
            loc=torch.ones(3),
            covariance_matrix=torch.eye(3),
        ),
        "NegativeBinomial": functools.partial(
            distributions.NegativeBinomial,
            total_count=1,
            probs=torch.tensor([0.5, 0.5]),
        ),
        "Normal": functools.partial(distributions.Normal, loc=0.0, scale=1.0),
        "OneHotCategorical": functools.partial(
            distributions.OneHotCategorical, probs=torch.tensor([0.5, 0.5])
        ),
        "OneHotCategoricalStraightThrough": functools.partial(
            distributions.OneHotCategoricalStraightThrough,
            probs=torch.tensor([0.5, 0.5]),
        ),
        "Pareto": functools.partial(distributions.Pareto, scale=1.0, alpha=1.0),
        "Poisson": functools.partial(distributions.Poisson, rate=1.0),
        "RelaxedBernoulli": functools.partial(
            distributions.RelaxedBernoulli,
            temperature=1,
            probs=torch.tensor([0.5, 0.5]),
        ),
        "RelaxedOneHotCategorical": functools.partial(
            distributions.RelaxedOneHotCategorical,
            temperature=1,
            probs=torch.tensor([0.5, 0.5]),
        ),
        "StudentT": functools.partial(distributions.StudentT, df=1, loc=0.0, scale=1.0),
        "Uniform": functools.partial(distributions.Uniform, low=0.0, high=1.0),
        "VonMises": functools.partial(
            distributions.VonMises, loc=0.0, concentration=1.0
        ),
        "Weibull": functools.partial(
            distributions.Weibull, scale=1.0, concentration=1.0
        ),
        "Wishart": functools.partial(
            distributions.Wishart, df=torch.tensor([2]), covariance_matrix=torch.eye(2)
        ),
    }

    @pytest.mark.parametrize("partial", dist_partials)
    def test_deterministic_sample(self, partial):
        with (
            pytest.raises(RuntimeError, match="DETERMINISTIC, MEAN and MODE")
            if partial in ("LKJCholesky",)
            else contextlib.nullcontext()
        ):
            partial = self.dist_partials[partial]
            dist_mod = ProbabilisticTensorDictModule(
                in_keys=[], out_keys=["out"], distribution_class=partial
            )
            with set_interaction_type("DETERMINISTIC"):
                td = dist_mod(TensorDict())
                assert td["out"] is not None


class TestTDModule:
    class MyMutableSequence(MutableSequence):
        def __init__(self, initial_data=None):
            self._data = [] if initial_data is None else list(initial_data)

        def __getitem__(self, index):
            return self._data[index]

        def __setitem__(self, index, value):
            self._data[index] = value

        def __delitem__(self, index):
            del self._data[index]

        def __len__(self):
            return len(self._data)

        def insert(self, index, value):
            self._data.insert(index, value)

    def test_module_method_and_kwargs(self):

        class MyNet(nn.Module):
            def my_func(self, tensor: torch.Tensor, *, an_integer: int):
                return tensor + an_integer

        s = TensorDictSequential(
            {
                "a": lambda td: td + 1,
                "b": lambda td: td * 2,
                "c": TensorDictModule(
                    MyNet(),
                    in_keys=["a"],
                    out_keys=["b"],
                    method="my_func",
                    method_kwargs={"an_integer": 2},
                ),
            }
        )
        td = s(TensorDict(a=0))

        assert td["b"] == 4

    def test_mutable_sequence(self):
        in_keys = self.MyMutableSequence(["a", "b", "c"])
        out_keys = self.MyMutableSequence(["d", "e", "f"])
        mod = TensorDictModule(lambda *x: x, in_keys=in_keys, out_keys=out_keys)
        td = mod(TensorDict(a=0, b=0, c=0))
        assert "d" in td
        assert "e" in td
        assert "f" in td

    def test_auto_unravel(self):
        tdm = TensorDictModule(
            lambda x: x,
            in_keys=["a", ("b",), ("c", ("d",))],
            out_keys=["e", ("f",), ("g", ("h",))],
        )

        assert tdm.in_keys == ["a", "b", ("c", "d")]
        assert tdm.out_keys == ["e", "f", ("g", "h")]

        class MyClass(TensorDictModuleBase):
            def __init__(self):
                self.in_keys = ["a", ("b",), ("c", ("d",))]
                self.out_keys = ["e", ("f",), ("g", ("h",))]
                super().__init__()

        c = MyClass()
        assert c.in_keys == ["a", "b", ("c", "d")]
        c.in_keys = ["a1", ("b1",), ("c1", ("d1",))]
        assert c.in_keys == ["a1", "b1", ("c1", "d1")]
        assert c.out_keys == ["e", "f", ("g", "h")]
        c.out_keys = [("e1",), ("f1",), ("g1", "h1")]
        assert c.out_keys == ["e1", "f1", ("g1", "h1")]

        class MyClass2(TensorDictModuleBase):
            in_keys = ["a", ("b",), ("c", ("d",))]
            out_keys = ["e", ("f",), ("g", ("h",))]

        c = MyClass2()
        assert c.in_keys == ["a", "b", ("c", "d")]
        c.in_keys = ["a1", ("b1",), ("c1", ("d1",))]
        assert c.in_keys == ["a1", "b1", ("c1", "d1")]

        assert c.out_keys == ["e", "f", ("g", "h")]
        c.out_keys = [("e1",), ("f1",), ("g1", "h1")]
        assert c.out_keys == ["e1", "f1", ("g1", "h1")]

    @pytest.mark.parametrize("args", [True, False])
    def test_input_keys(self, args):
        if args:
            args = ["1"]
            kwargs = {}
        else:
            args = []
            kwargs = {"1": "a", ("2", "smth"): "b", ("3", ("other", ("thing",))): "c"}

        def fn(a, b=None, *, c=None):
            if "c" in kwargs.values():
                assert c is not None
            if "b" in kwargs.values():
                assert b is not None
            return a + 1

        if kwargs:
            module = TensorDictModule(
                fn, in_keys=kwargs, out_keys=["a"], out_to_in_map=False
            )
            td = TensorDict(
                {
                    "1": torch.ones(1),
                    ("2", "smth"): torch.ones(2),
                    ("3", ("other", ("thing",))): torch.ones(3),
                },
                [],
            )
        else:
            module = TensorDictModule(fn, in_keys=args, out_keys=["a"])
            td = TensorDict({"1": torch.ones(1)}, [])
        assert (module(td)["a"] == 2).all()

    def test_unused_out_to_in_map(self):
        def fn(x, y):
            return x + y

        with pytest.warns(
            match="out_to_in_map is not None but is only used when in_key is a dictionary."
        ):
            _ = TensorDictModule(fn, in_keys=["x"], out_keys=["a"], out_to_in_map=False)

    def test_input_keys_dict_reversed(self):
        in_keys = {"x": "1", "y": "2"}

        def fn(x, y):
            return x + y

        module = TensorDictModule(
            fn, in_keys=in_keys, out_keys=["a"], out_to_in_map=True
        )

        td = TensorDict({"1": torch.ones(1), "2": torch.ones(1) * 3}, [])
        assert (module(td)["a"] == 4).all()

    def test_input_keys_match_reversed(self):
        in_keys = {"1": "x", "2": "y"}
        reversed_in_keys = {v: k for k, v in in_keys.items()}

        def fn(x, y):
            return y - x

        module = TensorDictModule(
            fn, in_keys=in_keys, out_keys=["a"], out_to_in_map=False
        )
        reversed_module = TensorDictModule(
            fn, in_keys=reversed_in_keys, out_keys=["a"], out_to_in_map=True
        )

        td = TensorDict({"1": torch.ones(1), "2": torch.ones(1) * 3}, [])

        assert module(td)["a"] == reversed_module(td)["a"] == torch.Tensor([2])

    @pytest.mark.parametrize("out_to_in_map", [True, False])
    def test_input_keys_wrong_mapping(self, out_to_in_map):
        in_keys = {"1": "x", "2": "y"}
        if not out_to_in_map:
            in_keys = {v: k for k, v in in_keys.items()}

        def fn(x, y):
            return x + y

        module = TensorDictModule(
            fn, in_keys=in_keys, out_keys=["a"], out_to_in_map=out_to_in_map
        )

        td = TensorDict({"1": torch.ones(1), "2": torch.ones(1) * 3}, [])

        with pytest.raises(TypeError, match="got an unexpected keyword argument '1'"):
            module(td)

    def test_input_keys_dict_deprecated_warning(self):
        in_keys = {"1": "x", "2": "y"}

        def fn(x, y):
            return x + y

        with pytest.warns(
            DeprecationWarning,
            match="Using a dictionary in_keys without specifying out_to_in_map is deprecated.",
        ):
            _ = TensorDictModule(fn, in_keys=in_keys, out_keys=["a"])

    def test_reset(self):
        torch.manual_seed(0)
        net = nn.ModuleList([nn.Sequential(nn.Linear(1, 1), nn.ReLU())])
        old_param = net[0][0].weight.data.clone()
        module = TensorDictModule(net, in_keys=["in"], out_keys=["out"])
        another_module = TensorDictModule(
            nn.Conv2d(1, 1, 1, 1), in_keys=["in"], out_keys=["out"]
        )
        seq = TensorDictSequential(module, another_module)

        seq.reset_parameters_recursive()
        assert torch.all(old_param != net[0][0].weight.data)

    def test_reset_warning(self):
        torch.manual_seed(0)
        net = nn.ModuleList([nn.Tanh(), nn.ReLU()])
        module = TensorDictModule(net, in_keys=["in"], out_keys=["out"])
        with pytest.warns(
            UserWarning,
            match="reset_parameters_recursive was called without the parameters argument and did not find any parameters to reset",
        ):
            module.reset_parameters_recursive()

    @pytest.mark.parametrize(
        "net",
        [
            nn.ModuleList([nn.Sequential(nn.Linear(1, 1), nn.ReLU())]),
            nn.Linear(2, 1),
            nn.Sequential(nn.Tanh(), nn.Linear(1, 1), nn.Linear(2, 1)),
        ],
    )
    def test_reset_functional(self, net):
        torch.manual_seed(0)
        module = TensorDictModule(net, in_keys=["in"], out_keys=["out"])
        another_module = TensorDictModule(
            nn.Conv2d(1, 1, 1, 1), in_keys=["in"], out_keys=["out"]
        )
        seq = TensorDictSequential(module, another_module)

        params = TensorDict.from_module(seq)
        old_params = params.clone(recurse=True)
        new_params = params.clone(recurse=True)
        returned_params = seq.reset_parameters_recursive(new_params)

        weights_changed = new_params != old_params
        for w in weights_changed.values(include_nested=True, leaves_only=True):
            assert w.all(), f"Weights should have changed but did not for {w}"

        module_params = TensorDict.from_module(seq)
        overwrote_stateful_params = module_params != old_params
        for p in overwrote_stateful_params.values(
            include_nested=True, leaves_only=True
        ):
            assert not p.any(), f"Overwrote stateful weights from the module {p}"

        returned_params_eq_inplace_updated_params = returned_params == new_params
        for p in returned_params_eq_inplace_updated_params.values(
            include_nested=True, leaves_only=True
        ):
            assert (
                p.all()
            ), f"Discrepancy between returned weights and those in-place updated {p}"

    def test_reset_functional_called_once(self):
        import unittest.mock

        torch.manual_seed(0)
        lin = nn.Linear(1, 1)
        lin.reset_parameters = unittest.mock.Mock(return_value=None)
        net = nn.ModuleList([nn.Sequential(lin, nn.ReLU())])
        module = TensorDictModule(net, in_keys=["in"], out_keys=["out"])
        nested_module = TensorDictModule(module, in_keys=["in"], out_keys=["out"])
        tripled_nested = TensorDictModule(
            nested_module, in_keys=["in"], out_keys=["out"]
        )

        params = TensorDict.from_module(tripled_nested)
        tripled_nested.reset_parameters_recursive(params)
        lin.reset_parameters.assert_called_once()

    def test_reset_extra_dims(self):
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(1, 1), nn.ReLU())
        module = TensorDictModule(net, in_keys=["in"], out_keys=["mid"])
        another_module = TensorDictModule(
            nn.Linear(1, 1), in_keys=["mid"], out_keys=["out"]
        )
        seq = TensorDictSequential(module, another_module)

        params = TensorDict.from_module(seq)
        new_params = params.expand(2, 3, *params.shape).clone()
        # Does not inherit from test case, no assertRaises :(
        with pytest.raises(RuntimeError):
            seq.reset_parameters_recursive(new_params)

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

    @pytest.mark.parametrize("lazy", [True, False])
    def test_stateful_tensorclass(self, lazy):
        @tensorclass
        class Data:
            inputs: torch.Tensor
            outputs: torch.Tensor = None

        torch.manual_seed(0)
        param_multiplier = 1
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        tensordict_module = TensorDictModule(
            module=net, in_keys=["inputs"], out_keys=["outputs"]
        )

        tc = Data(inputs=torch.randn(3, 3), batch_size=[3])
        tensordict_module(tc)
        assert tc.shape == torch.Size([3])
        assert tc.get("outputs").shape == torch.Size([3, 4])

    @pytest.mark.parametrize("out_keys", [["loc", "scale"], ["loc_1", "scale_1"]])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("it", [InteractionType.MODE, InteractionType.RANDOM, None])
    def test_stateful_probabilistic_deprec(self, lazy, it, out_keys):
        torch.manual_seed(0)
        param_multiplier = 2
        if lazy:
            net = nn.LazyLinear(4 * param_multiplier)
        else:
            net = nn.Linear(3, 4 * param_multiplier)

        in_keys = ["in"]
        net = TensorDictModule(
            module=nn.Sequential(net, NormalParamExtractor()),
            in_keys=in_keys,
            out_keys=out_keys,
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
        assert tensordict_module.default_interaction_type is not None

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_interaction_type(it):
            tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.parametrize("out_keys", [["low"], ["low1"], [("stuff", "low1")]])
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("max_dist", [1.0, 2.0])
    @pytest.mark.parametrize("it", [InteractionType.MODE, InteractionType.RANDOM, None])
    def test_stateful_probabilistic_kwargs(self, lazy, it, out_keys, max_dist):
        torch.manual_seed(0)
        if lazy:
            net = nn.LazyLinear(4)
        else:
            net = nn.Linear(3, 4)

        in_keys = ["in"]
        net = TensorDictModule(module=net, in_keys=in_keys, out_keys=out_keys)
        corr = TensorDictModule(
            lambda low: max_dist - low.abs(), in_keys=out_keys, out_keys=out_keys
        )

        kwargs = {
            "distribution_class": distributions.Uniform,
            "distribution_kwargs": {"high": max_dist},
        }
        if out_keys == ["low"]:
            dist_in_keys = ["low"]
        else:
            dist_in_keys = {"low": out_keys[0]}

        prob_module = ProbabilisticTensorDictModule(
            in_keys=dist_in_keys, out_keys=["out"], **kwargs
        )

        tensordict_module = ProbabilisticTensorDictSequential(net, corr, prob_module)
        assert tensordict_module.default_interaction_type is not None

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_interaction_type(it):
            tensordict_module(td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.parametrize("strict", [True, False])
    def test_strict(self, strict):
        def check(a, b):
            assert b is None
            return a

        tdm = TensorDictModule(
            check,
            in_keys=["present", "missing"],
            out_keys=["new_present"],
            strict=strict,
        )
        td = TensorDict(present=0)
        with pytest.raises(KeyError) if strict else contextlib.nullcontext():
            tdout = tdm(td)
            assert tdout["new_present"] is td["present"]

    def test_nontensor(self):
        tdm = TensorDictModule(
            lambda: NonTensorStack(NonTensorData(1), NonTensorData(2)),
            in_keys=[],
            out_keys=["out"],
        )
        assert tdm(TensorDict())["out"] == [1, 2]
        tdm = TensorDictModule(
            lambda: "a string!",
            in_keys=[],
            out_keys=["out"],
        )
        assert tdm(TensorDict())["out"] == "a string!"
        tdm = TensorDictModule(
            lambda a_string: a_string + " is a string!",
            in_keys=["string"],
            out_keys=["another string"],
        )
        assert (
            tdm(TensorDict(string="a string"))["another string"]
            == "a string is a string!"
        )
        tdm = TensorDictModule(
            lambda string: string + " is a string!",
            in_keys={"string": "key"},
            out_keys=["another string"],
            out_to_in_map=True,
        )
        assert (
            tdm(TensorDict(key="a string"))["another string"] == "a string is a string!"
        )

    @pytest.mark.parametrize(
        "out_keys",
        [
            ["loc", "scale"],
            ["loc_1", "scale_1"],
            [("params_td", "loc_1"), ("scale_1",)],
        ],
    )
    @pytest.mark.parametrize("lazy", [True, False])
    @pytest.mark.parametrize("it", [InteractionType.MODE, InteractionType.RANDOM, None])
    def test_stateful_probabilistic(self, lazy, it, out_keys):
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
        assert tensordict_module.default_interaction_type is not None

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        with set_interaction_type(it):
            tensordict_module(td)
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

        with pytest.warns(FutureWarning, match="integrated functorch"):
            tensordict_module, params, buffers = make_functional_functorch(
                tensordict_module
            )

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        tensordict_module(params, buffers, td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    def test_vmap_kwargs(self):
        module = TensorDictModule(
            lambda x, *, y: x + y,
            in_keys={"1": "x", "2": "y"},
            out_keys=["z"],
            out_to_in_map=False,
        )
        td = TensorDict(
            {"1": torch.ones((10,)), "2": torch.ones((10,)) * 2}, batch_size=[10]
        )
        tdout = vmap(module)(td)
        assert tdout is not td
        assert (tdout["z"] == 3).all()

    def test_deepcopy(self):
        class DummyModule(nn.Linear):
            some_attribute = "a"

            def __deepcopy__(self, memodict=None):
                return DummyModule(self.in_features, self.out_features)

        tdmodule = TensorDictModule(DummyModule(1, 1), in_keys=["a"], out_keys=["b"])
        with pytest.raises(AttributeError):
            tdmodule.__deepcopy__
        assert tdmodule.some_attribute == "a"
        assert isinstance(copy.deepcopy(tdmodule), TensorDictModule)

    def test_dispatch_deactivate(self):
        tdm = TensorDictModule(nn.Linear(1, 1), ["a"], ["b"])
        td = TensorDict({"a": torch.zeros(1, 1)}, 1)
        tdm(td)
        with _set_dispatch_td_nn_modules(True):
            out = tdm(a=torch.zeros(1, 1))
            assert (out == td["b"]).all()
        with _set_dispatch_td_nn_modules(False), pytest.raises(
            TypeError, match="missing 1 required positional argument"
        ):
            tdm(a=torch.zeros(1, 1))

        # checks that things are back in place
        tdm = TensorDictModule(nn.Linear(1, 1), ["a"], ["b"])
        tdm(a=torch.zeros(1, 1))

    def test_dispatch(self):
        tdm = TensorDictModule(nn.Linear(1, 1), ["a"], ["b"])
        td = TensorDict({"a": torch.zeros(1, 1)}, 1)
        tdm(td)
        out = tdm(a=torch.zeros(1, 1))
        assert (out == td["b"]).all()

    def test_dispatch_changing_size(self):
        # regression test on non max batch-size for dispatch
        tdm = TensorDictModule(nn.Linear(1, 2), ["a"], ["b"])
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

    @pytest.mark.parametrize("auto_batch_size", [True, False])
    def test_dispatch_auto_batch_size(self, auto_batch_size):
        class MyModuleNest(nn.Module):
            in_keys = [("a", "c"), "d"]
            out_keys = ["b"]

            @dispatch(auto_batch_size=auto_batch_size)
            def forward(self, tensordict):
                if auto_batch_size:
                    assert tensordict.shape == (2, 3)
                else:
                    assert tensordict.shape == ()
                tensordict["b"] = tensordict["a", "c"] + tensordict["d"]
                return tensordict

        module = MyModuleNest()
        b = module(torch.zeros(2, 3), d=torch.ones(2, 3))
        assert (b == 1).all()

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

    @pytest.mark.parametrize("output_type", [dict, TensorDict])
    def test_tdmodule_dict_output(self, output_type):
        class MyModule(nn.Identity):
            def forward(self, input):
                if output_type is dict:
                    return {"b": input}
                else:
                    return TensorDict({"b": input}, [])

        module = TensorDictModule(MyModule(), in_keys=["a"], out_keys=["b"])
        out = module(TensorDict({"a": torch.randn(3)}, []))
        assert (out["b"] == out["a"]).all()

    @set_list_to_stack(True)
    def test_tdmodule_inplace(self):
        tdm = TensorDictModule(
            lambda x: (x, x), in_keys=["x"], out_keys=["y", "z"], inplace=False
        )
        td = TensorDict(x=[0], batch_size=[1], device="cpu")
        td_out = tdm(td)
        assert td_out is not td
        assert "x" not in td_out
        assert "y" in td_out
        assert "z" in td_out
        assert td_out.batch_size == ()
        assert td_out.device is None

        tdm = TensorDictModule(
            lambda x: (x, x), in_keys=["x"], out_keys=["y", "z"], inplace="empty"
        )
        td = TensorDict(x=[0], batch_size=[1], device="cpu")
        td_out = tdm(td)
        assert "x" not in td_out
        assert "y" in td_out
        assert "z" in td_out
        assert td_out.batch_size == (1,)
        assert td_out.device == torch.device("cpu")

        td_out = tdm(td, tensordict_out=TensorDict())
        assert "x" not in td_out
        assert "y" in td_out
        assert "z" in td_out
        assert td_out.batch_size == ()
        assert td_out.device is None


class TestTDSequence:
    @pytest.mark.parametrize("inplace", [True, False, None])
    @pytest.mark.parametrize("module_inplace", [True, False])
    def test_tdseq_inplace(self, inplace, module_inplace):
        model = TensorDictSequential(
            TensorDictModule(
                lambda x: (x + 1, x - 1),
                in_keys=["input"],
                out_keys=[("intermediate", "0"), ("intermediate", "1")],
                inplace=module_inplace,
            ),
            TensorDictModule(
                lambda y0, y1: y0 * y1,
                in_keys=[("intermediate", "0"), ("intermediate", "1")],
                out_keys=["output"],
                inplace=module_inplace,
            ),
            inplace=inplace,
        )
        input = TensorDict(input=torch.zeros(()))
        output = model(input)
        if inplace:
            assert output is input
            assert "input" in output
        else:
            if not module_inplace or inplace is False:
                # In this case, inplace=False and inplace=None have the same behavior
                assert output is not input, (module_inplace, inplace)
                assert "input" not in output, (module_inplace, inplace)
            else:
                # In this case, inplace=False and inplace=None have the same behavior
                assert output is input, (module_inplace, inplace)
                assert "input" in output, (module_inplace, inplace)

        assert "output" in output

    def test_ordered_dict(self):
        linear = nn.Linear(3, 4)
        linear.weight.data.fill_(0)
        linear.bias.data.fill_(1)
        layer0 = TensorDictModule(linear, in_keys=["x"], out_keys=["y"])
        ordered_dict = OrderedDict(
            layer0=layer0,
            layer1=lambda x: x + 1,
        )
        seq = TensorDictSequential(ordered_dict)
        td = seq(TensorDict(x=torch.ones(3)))
        assert (td["x"] == 2).all()
        assert (td["y"] == 2).all()
        assert seq["layer0"] is layer0

    def test_ordered_dict_select_subsequence(self):
        ordered_dict = OrderedDict(
            layer0=TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["y"]),
            layer1=TensorDictModule(lambda x: x - 1, in_keys=["y"], out_keys=["z"]),
            layer2=TensorDictModule(
                lambda x, y: x + y, in_keys=["x", "y"], out_keys=["a"]
            ),
        )
        seq = TensorDictSequential(ordered_dict)
        assert len(seq) == 3
        assert isinstance(seq.module, nn.ModuleDict)
        seq_select = seq.select_subsequence(out_keys=["a"])
        assert len(seq_select) == 2
        assert isinstance(seq_select.module, nn.ModuleDict)
        assert list(seq_select.module) == ["layer0", "layer2"]

    def test_ordered_dict_select_outkeys(self):
        ordered_dict = OrderedDict(
            layer0=TensorDictModule(
                lambda x: x + 1, in_keys=["x"], out_keys=["intermediate"]
            ),
            layer1=TensorDictModule(
                lambda x: x - 1, in_keys=["intermediate"], out_keys=["z"]
            ),
            layer2=TensorDictModule(
                lambda x, y: x + y, in_keys=["x", "z"], out_keys=["a"]
            ),
        )
        seq = TensorDictSequential(ordered_dict)
        assert len(seq) == 3
        assert isinstance(seq.module, nn.ModuleDict)
        seq.select_out_keys("z", "a")
        td = seq(TensorDict(x=0))
        assert "intermediate" not in td
        assert "z" in td
        assert "a" in td

    @pytest.mark.parametrize("args", [True, False])
    def test_input_keys(self, args):
        module0 = TensorDictModule(lambda x: x + 0, in_keys=["input"], out_keys=["1"])
        if args:
            args = ["1"]
            kwargs = {}
        else:
            args = []
            kwargs = {"1": "a", ("2", "smth"): "b", ("3", ("other", ("thing",))): "c"}

        def fn(a, b=None, *, c=None):
            if "c" in kwargs.values():
                assert c is not None
            if "b" in kwargs.values():
                assert b is not None
            return a + 1

        if kwargs:
            module1 = TensorDictModule(
                fn, in_keys=kwargs, out_keys=["a"], out_to_in_map=False
            )
            td = TensorDict(
                {
                    "input": torch.ones(1),
                    ("2", "smth"): torch.ones(2),
                    ("3", ("other", ("thing",))): torch.ones(3),
                },
                [],
            )
        else:
            module1 = TensorDictModule(fn, in_keys=args, out_keys=["a"])
            td = TensorDict({"input": torch.ones(1)}, [])
        module = TensorDictSequential(module0, module1)
        assert (module(td)["a"] == 2).all()

    def test_tdseq_tdoutput(self):
        mod = TensorDictSequential(
            TensorDictModule(lambda x: x + 2, in_keys=["a"], out_keys=["c"]),
            TensorDictModule(lambda x: (x + 2, x), in_keys=["b"], out_keys=["d", "e"]),
        )
        inp = TensorDict({"a": 0, "b": 1})
        inp_clone = inp.clone()
        out = TensorDict()
        out2 = mod(inp, tensordict_out=out)
        assert out is out2
        assert set(out.keys()) == set(mod.out_keys)
        assert set(inp.keys()) == set(inp_clone.keys())
        mod.select_out_keys("d")
        out = TensorDict()
        out2 = mod(inp, tensordict_out=out)
        assert out is out2
        assert set(out.keys()) == set(mod.out_keys) == {"d"}
        assert set(inp.keys()) == set(inp_clone.keys())

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
        assert set(seq.in_keys) == set(unravel_key_list(("key1", "key2", "key3")))
        assert set(seq.out_keys) == set(unravel_key_list(("foo1", "key1", "key2")))

    def test_key_exclusion_constructor(self):
        module1 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        module2 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["key1", "key3"], out_keys=["key1"]
        )
        module3 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["foo1", "key3"], out_keys=["key2"]
        )
        seq = TensorDictSequential(
            module1, module2, module3, selected_out_keys=["key2"]
        )
        assert set(seq.in_keys) == set(unravel_key_list(("key1", "key2", "key3")))
        assert seq.out_keys == ["key2"]

    def test_key_exclusion_constructor_exec(self):
        module1 = TensorDictModule(
            lambda x, y: x + y, in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        module2 = TensorDictModule(
            lambda x, y: x + y, in_keys=["key1", "key3"], out_keys=["key1"]
        )
        module3 = TensorDictModule(
            lambda x, y: x + y, in_keys=["foo1", "key3"], out_keys=["key2"]
        )
        seq = TensorDictSequential(
            module1, module2, module3, selected_out_keys=["key2"]
        )
        assert set(seq.in_keys) == set(unravel_key_list(("key1", "key2", "key3")))
        assert seq.out_keys == ["key2"]
        td = TensorDict(key1=0, key2=0, key3=1)
        out = seq(td)
        assert out is td
        assert "key1" in out
        assert "key2" in out
        assert "key3" in out
        assert "foo1" not in out
        assert out["key2"] == 1

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
        net2 = nn.Sequential(net2, NormalParamExtractor())

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
        assert tdmodule.default_interaction_type is not None

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

    @pytest.mark.parametrize("return_log_prob", [True, False])
    @pytest.mark.parametrize("td_out", [True, False])
    @set_composite_lp_aggregate(False)
    def test_probtdseq(self, return_log_prob, td_out):
        mod = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda x: x + 2, in_keys=["a"], out_keys=["c"]),
            TensorDictModule(lambda x: (x + 2, x), in_keys=["b"], out_keys=["d", "e"]),
            ProbabilisticTensorDictModule(
                in_keys={"loc": "d", "scale": "e"},
                out_keys=["f"],
                distribution_class=Normal,
                return_log_prob=return_log_prob,
                default_interaction_type="random",
            ),
        )
        assert mod.default_interaction_type is not None
        inp = TensorDict({"a": 0.0, "b": 1.0})
        inp_clone = inp.clone()
        if td_out:
            out = TensorDict()
        else:
            out = None
        out2 = mod(inp, tensordict_out=out)
        assert not mod._select_before_return
        if td_out:
            assert out is out2
        else:
            assert out2 is inp
        assert set(out2.keys()) - {"a", "b"} == set(mod.out_keys), (
            td_out,
            return_log_prob,
        )

        inp = inp_clone.clone()
        mod.select_out_keys("f")
        if td_out:
            out = TensorDict()
        else:
            out = None
        out2 = mod(inp, tensordict_out=out)
        assert mod._select_before_return
        if td_out:
            assert out is out2
        else:
            assert out2 is inp
        expected = {"f"}
        if td_out:
            assert set(out2.keys()) == set(mod.out_keys) == expected
        else:
            assert (
                set(out2.keys()) - set(inp_clone.keys())
                == set(mod.out_keys)
                == expected
            )

    @set_composite_lp_aggregate(False)
    def test_probtdseq_multdist(self):

        tdm0 = TensorDictModule(torch.nn.Linear(3, 4), in_keys=["x"], out_keys=["loc"])
        tdm1 = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["y"],
            distribution_class=torch.distributions.Normal,
            distribution_kwargs={"scale": 1},
            default_interaction_type="random",
        )
        tdm2 = TensorDictModule(torch.nn.Linear(4, 5), in_keys=["y"], out_keys=["loc2"])
        tdm3 = ProbabilisticTensorDictModule(
            in_keys={"loc": "loc2"},
            out_keys=["z"],
            distribution_class=torch.distributions.Normal,
            distribution_kwargs={"scale": 1},
            default_interaction_type="random",
        )

        tdm = ProbabilisticTensorDictSequential(
            tdm0,
            tdm1,
            tdm2,
            tdm3,
            return_composite=True,
        )
        assert tdm.default_interaction_type is not None
        dist: CompositeDistribution = tdm.get_dist(TensorDict(x=torch.randn(10, 3)))
        s = dist.sample()
        assert isinstance(dist, CompositeDistribution)
        assert isinstance(dist.log_prob(s), TensorDict)

        v = tdm(TensorDict(x=torch.randn(10, 3)))
        assert set(v.keys()) == {"x", "loc", "y", "loc2", "z"}
        assert isinstance(tdm.log_prob(v), TensorDict)

    @set_composite_lp_aggregate(False)
    def test_probtdseq_intermediate_dist(self):
        tdm0 = TensorDictModule(torch.nn.Linear(3, 4), in_keys=["x"], out_keys=["loc"])
        tdm1 = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["y"],
            distribution_class=torch.distributions.Normal,
            distribution_kwargs={"scale": 1},
            default_interaction_type="random",
        )
        tdm2 = TensorDictModule(torch.nn.Linear(4, 5), in_keys=["y"], out_keys=["loc2"])
        tdm = ProbabilisticTensorDictSequential(
            tdm0,
            tdm1,
            tdm2,
            return_composite=True,
        )
        assert tdm.default_interaction_type is not None
        dist: CompositeDistribution = tdm.get_dist(TensorDict(x=torch.randn(10, 3)))
        assert isinstance(dist, CompositeDistribution)

        s = dist.sample()
        assert isinstance(dist.log_prob(s), TensorDict)

        v = tdm(TensorDict(x=torch.randn(10, 3)))
        assert set(v.keys()) == {"x", "loc", "y", "loc2"}
        assert isinstance(tdm.log_prob(v), TensorDict)

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

        with pytest.warns(FutureWarning):
            ftdmodule, params, buffers = make_functional_functorch(tdmodule)

        td = TensorDict({"in": torch.randn(3, 3)}, [3])
        ftdmodule(params, buffers, td)
        assert td.shape == torch.Size([3])
        assert td.get("out").shape == torch.Size([3, 4])

    @pytest.mark.parametrize(
        "in_keys", [None, ("a",), ("b",), ("d",), ("b", "d"), ("a", "d")]
    )
    @pytest.mark.parametrize(
        "out_keys",
        [None, ("b",), ("c",), ("d",), ("e",), ("b", "c"), ("b", "d"), ("b", "e")],
    )
    def test_submodule_sequence_nested(self, in_keys, out_keys):
        Seq = TensorDictSequential
        Mod = TensorDictModule
        idn = lambda x: x + 1
        module_1 = Seq(
            Mod(idn, in_keys=["a"], out_keys=["b"]),
            Mod(idn, in_keys=["b"], out_keys=["c"]),
            Mod(idn, in_keys=["b"], out_keys=["d"]),
            Mod(idn, in_keys=["d"], out_keys=["e"]),
        )
        module_2 = Seq(
            Seq(
                Mod(idn, in_keys=["a"], out_keys=["b"]),
                Mod(idn, in_keys=["b"], out_keys=["c"]),
            ),
            Seq(
                Mod(idn, in_keys=["b"], out_keys=["d"]),
                Mod(idn, in_keys=["d"], out_keys=["e"]),
            ),
        )
        try:
            sel_module_1 = module_1.select_subsequence(
                in_keys=in_keys, out_keys=out_keys
            )
        except ValueError:
            # incongruent keys
            return
        sel_module_2 = module_2.select_subsequence(in_keys=in_keys, out_keys=out_keys)
        td = TensorDict(
            {
                "a": torch.zeros(()),
                "b": torch.zeros(()),
                "c": torch.zeros(()),
                "d": torch.zeros(()),
                "e": torch.zeros(()),
            },
            [],
        )
        assert (sel_module_1(td.clone()) == sel_module_2(td.clone())).all()

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


@pytest.mark.parametrize("it", [InteractionType.RANDOM, InteractionType.MODE])
class TestSIM:
    def test_cm(self, it):
        with set_interaction_type(it):
            assert interaction_type() == it

    def test_dec(self, it):
        @set_interaction_type(it)
        def dummy():
            assert interaction_type() == it

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
        KeyError,
        match="Some tensors that are necessary for the module call may not have not been found in the input tensordict",
    ):
        module(TensorDict({"c": torch.randn(())}, []))


def test_input():
    class MyModule(nn.Module):
        pass

    def mycallable():
        pass

    # this should work
    for module in [MyModule(), mycallable]:
        TensorDictModule(module, in_keys=["i"], out_keys=["o"])
        TensorDictModule(module, in_keys=["i", "i2"], out_keys=["o"])
        TensorDictModule(module, in_keys=["i"], out_keys=["o", "o2"])
        TensorDictModule(module, in_keys=["i", "i2"], out_keys=["o", "o2"])
        TensorDictModule(module, in_keys=[tuple("i")], out_keys=[tuple("o")])
        TensorDictModule(module, in_keys=[("i", "i2")], out_keys=[tuple("o")])
        TensorDictModule(module, in_keys=[tuple("i")], out_keys=[("o", "o2")])
        TensorDictModule(module, in_keys=[("i", "i2")], out_keys=[("o", "o2")])
        TensorDictModule(
            module, in_keys=[(("i", "i2"), ("i3",))], out_keys=[("o", "o2")]
        )
        TensorDictModule(
            module, in_keys=[("i", "i2")], out_keys=[(("o", "o2"), ("o3",))]
        )
        TensorDictModule(
            module,
            in_keys={"i": "i1", (("i2",),): "i3"},
            out_keys=[("o", "o2")],
            out_to_in_map=False,
        )

        # corner cases that should work
        TensorDictModule(module, in_keys=[("_", "")], out_keys=[("_", "")])
        TensorDictModule(module, in_keys=[("_", "")], out_keys=[("a", "a")])
        TensorDictModule(module, in_keys=[""], out_keys=["_"])
        with pytest.warns(UserWarning, match='key "_"'):
            TensorDictModule(module, in_keys=["_"], out_keys=[""])

    # this should raise
    for wrong_model in (MyModule, int, [123], 1, torch.randn(2)):
        with pytest.raises(ValueError, match=r"Module .* is not callable"):
            TensorDictModule(wrong_model, in_keys=["in"], out_keys=["out"])

    # missing or wrong keys
    for wrong_keys in (None, 123, [123]):
        with pytest.raises(
            ValueError, match="out_keys must be of type list, str or tuples of str"
        ):
            TensorDictModule(MyModule(), in_keys=["in"], out_keys=wrong_keys)

        with pytest.raises(
            ValueError, match="in_keys must be of type list, str or tuples of str"
        ):
            TensorDictModule(MyModule(), in_keys=wrong_keys, out_keys=["out"])


def test_method_forward():
    # ensure calls to custom methods are correctly forwarded to wrapped module
    from unittest.mock import MagicMock

    class MyModule(nn.Module):
        def mycustommethod(self):
            pass

        def overwrittenmethod(self):
            pass

    MyModule.mycustommethod = MagicMock()
    MyModule.overwrittenmethod = MagicMock()

    module = TensorDictModule(MyModule(), in_keys=["in"], out_keys=["out"])
    module.mycustommethod()
    assert MyModule.mycustommethod.called

    module.mycustommethod()
    assert not MyModule.overwrittenmethod.called


class TestSkipExisting:
    @pytest.mark.parametrize("mode", [True, False, None])
    def test_global(self, mode):
        assert skip_existing() is False
        if mode is None:
            with pytest.raises(RuntimeError, match="It seems"):
                with set_skip_existing(mode):
                    pass
            assert skip_existing() is False
            return

        with set_skip_existing(mode):
            assert skip_existing() is mode
        assert skip_existing() is False

    def test_global_with_module(self):
        class MyModule(TensorDictModuleBase):
            in_keys = []
            out_keys = ["out"]

            @set_skip_existing(None)
            def forward(self, tensordict):
                tensordict.set("out", torch.ones(()))
                return tensordict

        module = MyModule()
        td = module(TensorDict({"out": torch.zeros(())}, []))
        assert (td["out"] == 1).all()
        with set_skip_existing(True):
            td = module(TensorDict({"out": torch.zeros(())}, []))  # no print
        assert (td["out"] == 0).all()
        td = module(TensorDict({"out": torch.zeros(())}, []))
        assert (td["out"] == 1).all()

    def test_module(self):
        class MyModule(TensorDictModuleBase):
            in_keys = []
            out_keys = ["out"]

            @set_skip_existing()
            def forward(self, tensordict):
                tensordict.set("out", torch.ones(()))
                return tensordict

        module = MyModule()
        td = module(TensorDict({"out": torch.zeros(())}, []))
        assert (td["out"] == 0).all()
        td = module(TensorDict())  # prints hello
        assert (td["out"] == 1).all()

    def test_tdmodule(self):
        module = TensorDictModule(lambda x: x + 1, in_keys=["in"], out_keys=["out"])
        td = TensorDict({"in": torch.zeros(())}, [])
        module(td)
        assert (td["out"] == 1).all()

        td = TensorDict({"in": torch.zeros(()), "out": torch.zeros(())}, [])
        module(td)
        assert (td["out"] == 1).all()

        td = TensorDict({"in": torch.zeros(()), "out": torch.zeros(())}, [])
        with set_skip_existing(True):
            module(td)
        assert (td["out"] == 0).all()

        td = TensorDict({"in": torch.zeros(()), "out": torch.zeros(())}, [])
        with set_skip_existing(False):
            module(td)
        assert (td["out"] == 1).all()

    def test_tdseq(self):
        class MyModule(TensorDictModuleBase):
            in_keys = ["in"]
            out_keys = ["out"]

            def forward(self, tensordict):
                tensordict["out"] = tensordict["in"] + 1
                return tensordict

        module = TensorDictSequential(MyModule())

        td = TensorDict({"in": torch.zeros(())}, [])
        module(td)
        assert (td["out"] == 1).all()

        td = TensorDict({"in": torch.zeros(()), "out": torch.zeros(())}, [])
        module(td)
        assert (td["out"] == 1).all()

        td = TensorDict({"in": torch.zeros(()), "out": torch.zeros(())}, [])
        with set_skip_existing(True):
            module(td)
        assert (td["out"] == 0).all()

        td = TensorDict({"in": torch.zeros(()), "out": torch.zeros(())}, [])
        with set_skip_existing(False):
            module(td)
        assert (td["out"] == 1).all()


@pytest.mark.parametrize("out_d_key", [("d", "e"), ["d"], ["d", "e"]])
@pytest.mark.parametrize("unpack", [True, False])
class TestSelectOutKeys:
    def test_tdmodule(self, out_d_key, unpack):
        mod = TensorDictModule(
            lambda x, y: (x + 2, y + 2, x), in_keys=["a", "b"], out_keys=["c", "d", "e"]
        )
        assert mod.out_keys == unravel_key_list(["c", "d", "e"])
        td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
        assert all(key in td.keys() for key in ["a", "b", "c", "d", "e"])
        if unpack:
            mod2 = mod.select_out_keys(*out_d_key)
            assert mod2 is mod
            assert mod.out_keys == unravel_key_list(out_d_key)
            td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            assert "c" not in td.keys()
            assert all(key in td.keys() for key in ["a", "b", "d"])
            mod2 = mod.reset_out_keys()
            assert mod2 is mod
            td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            assert all(key in td.keys() for key in ["a", "b", "c", "d", "e"])
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod2 = mod.select_out_keys(out_d_key)

    def test_tdmodule_dispatch(self, out_d_key, unpack):
        mod = TensorDictModule(
            lambda x, y: (x + 2, y + 2, x), in_keys=["a", "b"], out_keys=["c", "d", "e"]
        )
        exp_res = {"c": 2, "d": 3, "e": 0}
        res = mod(torch.zeros(()), torch.ones(()))
        assert len(res) == 3
        for i, v in enumerate(["c", "d", "e"]):
            assert (res[i] == exp_res[v]).all()
        if unpack:
            mod2 = mod.select_out_keys(*out_d_key)
            assert mod2 is mod
            assert mod.out_keys == unravel_key_list(out_d_key)
            res = mod(torch.zeros(()), torch.ones(()))
            if len(list(out_d_key)) == 1:
                res = [res]
            for i, v in enumerate(list(out_d_key)):
                assert (res[i] == exp_res[v]).all()

            mod2 = mod.reset_out_keys()
            assert mod2 is mod
            assert mod.out_keys == ["c", "d", "e"]

            res = mod(torch.zeros(()), torch.ones(()))
            assert len(res) == 3
            for i, v in enumerate(["c", "d", "e"]):
                assert (res[i] == exp_res[v]).all()
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod2 = mod.select_out_keys(out_d_key)

    def test_tdmodule_dispatch_kwargs(self, out_d_key, unpack):
        mod = TensorDictModule(
            lambda x, y: (x + 2, y + 2, x),
            in_keys=["a", ("b", ("1", ("2", ("3",))))],
            out_keys=["c", "d", "e"],
        )
        exp_res = {"c": 2, "d": 3, "e": 0}
        res = mod(a=torch.zeros(()), b_1_2_3=torch.ones(()))
        assert len(res) == 3
        for i, v in enumerate(["c", "d", "e"]):
            assert (res[i] == exp_res[v]).all()
        if unpack:
            mod2 = mod.select_out_keys(*out_d_key)
            assert mod2 is mod
            assert mod.out_keys == unravel_key_list(out_d_key)
            res = mod(torch.zeros(()), torch.ones(()))
            if len(list(out_d_key)) == 1:
                res = [res]
            for i, v in enumerate(list(out_d_key)):
                assert (res[i] == exp_res[v]).all()
            mod2 = mod.reset_out_keys()
            assert mod2 is mod
            res = mod(torch.zeros(()), torch.ones(()))
            assert len(res) == 3
            for i, v in enumerate(["c", "d", "e"]):
                assert (res[i] == exp_res[v]).all()
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod2 = mod.select_out_keys(out_d_key)

    def test_tdmodule_dispatch_firstcall(self, out_d_key, unpack):
        # calling the dispatch first or not may mess up the init
        mod = TensorDictModule(
            lambda x, y: (x + 2, y + 2, x), in_keys=["a", "b"], out_keys=["c", "d", "e"]
        )
        exp_res = {"c": 2, "d": 3, "e": 0}
        res = mod(torch.zeros(()), torch.ones(()))
        assert len(res) == 3
        for i, v in enumerate(["c", "d", "e"]):
            assert (res[i] == exp_res[v]).all()
        if unpack:
            mod2 = mod.select_out_keys(*out_d_key)
            assert mod2 is mod
            assert mod.out_keys == unravel_key_list(out_d_key)
            # ignore result but make sure we call _init
            mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            res = mod(torch.zeros(()), torch.ones(()))
            if len(list(out_d_key)) == 1:
                res = [res]
            for i, v in enumerate(list(out_d_key)):
                assert (res[i] == exp_res[v]).all()
            mod2 = mod.reset_out_keys()
            assert mod2 is mod
            res = mod(torch.zeros(()), torch.ones(()))
            assert len(res) == 3
            for i, v in enumerate(["c", "d", "e"]):
                assert (res[i] == exp_res[v]).all()
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod2 = mod.select_out_keys(out_d_key)

    def test_tdseq(self, out_d_key, unpack):
        mod = TensorDictSequential(
            TensorDictModule(lambda x: x + 2, in_keys=["a"], out_keys=["c"]),
            TensorDictModule(lambda x: (x + 2, x), in_keys=["b"], out_keys=["d", "e"]),
        )
        td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
        assert all(key in td.keys() for key in ["a", "b", "c", "d"])
        if unpack:
            mod2 = mod.select_out_keys(*out_d_key)
            assert mod2 is mod
            assert mod.out_keys == unravel_key_list(out_d_key)
            td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            assert "c" not in td.keys()
            assert all(key in td.keys() for key in ["a", "b", "d"])
            mod2 = mod.reset_out_keys()
            assert mod2 is mod
            td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            assert all(key in td.keys() for key in ["a", "b", "c", "d"])
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent|All keys in selected_out_keys must be in out_keys",
            ):
                mod2 = mod.select_out_keys(out_d_key)

    def test_tdseq_dispatch(self, out_d_key, unpack):
        mod = TensorDictSequential(
            TensorDictModule(lambda x: x + 2, in_keys=["a"], out_keys=["c"]),
            TensorDictModule(lambda x: (x + 2, x), in_keys=["b"], out_keys=["d", "e"]),
        )

        exp_res = {"c": 2, "d": 3, "e": 1}
        res = mod(torch.zeros(()), torch.ones(()))
        assert len(res) == 3
        for i, v in enumerate(["c", "d", "e"]):
            assert (res[i] == exp_res[v]).all()
        if unpack:
            mod2 = mod.select_out_keys(*out_d_key)
            assert mod2 is mod
            assert mod.out_keys == unravel_key_list(out_d_key)
            res = mod(torch.zeros(()), torch.ones(()))
            if len(list(out_d_key)) == 1:
                res = [res]
            for i, v in enumerate(list(out_d_key)):
                assert (res[i] == exp_res[v]).all()
            mod2 = mod.reset_out_keys()
            assert mod2 is mod
            res = mod(torch.zeros(()), torch.ones(()))
            assert len(res) == 3
            for i, v in enumerate(["c", "d", "e"]):
                assert (res[i] == exp_res[v]).all()
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent|All keys in selected_out_keys must be in out_keys",
            ):
                mod2 = mod.select_out_keys(out_d_key)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_tdbase(self, inplace, out_d_key, unpack):
        class MyModule(TensorDictModuleBase):
            in_keys = ["a", "b"]
            out_keys = ["c", "d", "e"]

            def forward(self, tensordict):
                c = tensordict["a"] + 2
                d = tensordict["b"] + 2
                if not inplace:
                    tensordict = tensordict.select()
                tensordict["c"] = c
                tensordict["d"] = d
                tensordict["e"] = d + 2
                return tensordict

        # since we play with the __new__ and class attributes, let's check that everything's ok for a second instance
        for i in range(2):
            mod = MyModule()
            # some magic happened and in_keys and out_keys are not class attributes anymore!
            assert mod.out_keys is mod._out_keys, i
            td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            if inplace:
                assert set(td.keys()) == {"a", "b", "c", "d", "e"}
            else:
                assert set(td.keys()) == {"c", "d", "e"}
            if unpack:
                mod2 = mod.select_out_keys(*out_d_key)
                assert mod2 is mod
                assert mod.out_keys == unravel_key_list(out_d_key)
                td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
                assert "c" not in td.keys()
                if inplace:
                    assert set(td.keys()) == {"a", "b", *out_d_key}
                else:
                    assert set(td.keys()) == {*out_d_key}
                mod2 = mod.reset_out_keys()
                assert mod2 is mod
                td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
                if inplace:
                    assert set(td.keys()) == {"a", "b", "c", "d", "e"}
                else:
                    assert set(td.keys()) == {"c", "d", "e"}
            else:
                with pytest.raises(
                    (RuntimeError, ValueError),
                    match=r"key should be a |Can't select non existent",
                ):
                    mod2 = mod.select_out_keys(out_d_key)

    def test_tdmodule_wrap(self, out_d_key, unpack):
        mod = TensorDictModuleWrapper(
            TensorDictModule(
                lambda x, y: (x + 2, y + 2, x),
                in_keys=["a", "b"],
                out_keys=["c", "d", "e"],
            )
        )
        td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
        assert all(key in td.keys() for key in ["a", "b", "c", "d", "e"])
        if unpack:
            mod2 = mod.select_out_keys(*out_d_key)
            assert mod2 is mod
            assert mod.out_keys == unravel_key_list(out_d_key)
            td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            assert all(key not in td.keys() for key in {"c", "d", "e"} - {*out_d_key})
            assert all(key in td.keys() for key in ["a", "b", *out_d_key])
            mod2 = mod.reset_out_keys()
            assert mod2 is mod
            td = mod(TensorDict({"a": torch.zeros(()), "b": torch.ones(())}, []))
            assert all(key in td.keys() for key in ["a", "b", "c", "d", "e"])
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod2 = mod.select_out_keys(out_d_key)

    def test_tdmodule_wrap_dispatch(self, out_d_key, unpack):
        mod = TensorDictModuleWrapper(
            TensorDictModule(
                lambda x, y: (x + 2, y + 2, x),
                in_keys=["a", "b"],
                out_keys=["c", "d", "e"],
            )
        )
        exp_res = {"c": 2, "d": 3, "e": 0}
        res = mod(torch.zeros(()), torch.ones(()))
        assert len(res) == 3
        for i, v in enumerate(["c", "d", "e"]):
            assert (res[i] == exp_res[v]).all()
        if unpack:
            mod2 = mod.select_out_keys(*out_d_key)
            assert mod2 is mod
            assert mod.out_keys == unravel_key_list(out_d_key)
            res = mod(torch.zeros(()), torch.ones(()))
            if len(list(out_d_key)) == 1:
                res = [res]
            for i, v in enumerate(list(out_d_key)):
                assert (res[i] == exp_res[v]).all()
            mod2 = mod.reset_out_keys()
            assert mod2 is mod
            res = mod(torch.zeros(()), torch.ones(()))
            assert len(res) == 3
            for i, v in enumerate(["c", "d", "e"]):
                assert (res[i] == exp_res[v]).all()
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod2 = mod.select_out_keys(out_d_key)


@pytest.mark.parametrize("has_lambda", [False, True])
def test_serialization(has_lambda):
    if has_lambda:
        mod = lambda x: x + 1
    else:
        mod = nn.Linear(3, 3)
    mod = TensorDictModule(mod, in_keys=["x"], out_keys=["y"])
    serialized = pickle.dumps(mod)
    mod_unpickle = pickle.loads(serialized)
    x = torch.randn(3)
    assert (mod(x=x) == mod_unpickle(x=x)).all()


def test_module_buffer():
    module = nn.ModuleList([])
    td = TensorDict(
        {
            "a": torch.zeros(3),
            ("b", "c"): torch.ones(3),
        },
        [],
    )
    # should we monkey-patch module.register_buffer to make this possible?
    module._buffers["td"] = td
    assert module.td is td
    # test some functions that call _apply
    module.double()
    assert module.td["b", "c"].dtype is torch.float64
    module.float()
    assert module.td["b", "c"].dtype is torch.float
    module.bfloat16()
    assert module.td["b", "c"].dtype is torch.bfloat16
    if torch.cuda.device_count():
        module.cuda()
        assert module.td.device.type == "cuda"


class TestProbabilisticTensorDictModule:
    @set_composite_lp_aggregate(False)
    @pytest.mark.parametrize("inplace", [True, False, None])
    @pytest.mark.parametrize("module_inplace", [True, False])
    def test_tdprobseq_inplace(self, inplace, module_inplace):
        model = ProbabilisticTensorDictSequential(
            TensorDictModule(
                lambda x: (x + 1, x - 1),
                in_keys=["input"],
                out_keys=[("intermediate", "0"), ("intermediate", "1")],
                inplace=module_inplace,
            ),
            TensorDictModule(
                lambda y0, y1: y0 * y1,
                in_keys=[("intermediate", "0"), ("intermediate", "1")],
                out_keys=["output"],
                inplace=module_inplace,
            ),
            ProbabilisticTensorDictModule(
                in_keys={"logits": "output"},
                out_keys=["sample"],
                return_log_prob=True,
                distribution_class=Categorical,
            ),
            inplace=inplace,
        )
        input = TensorDict(input=torch.zeros((5,)))
        output = model(input)
        assert "sample_log_prob" in output
        assert "sample" in output
        if inplace:
            assert output is input
            assert "input" in output
        else:
            if not module_inplace or inplace is False:
                # In this case, inplace=False and inplace=None have the same behavior
                assert output is not input, (module_inplace, inplace)
                assert "input" not in output, (module_inplace, inplace)
            else:
                # In this case, inplace=False and inplace=None have the same behavior
                assert output is input, (module_inplace, inplace)
                assert "input" in output, (module_inplace, inplace)

        assert "output" in output

    @pytest.mark.parametrize("return_log_prob", [True, False])
    @set_composite_lp_aggregate(False)
    def test_probabilistic_n_samples(self, return_log_prob):
        prob = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1},
            return_log_prob=return_log_prob,
            num_samples=2,
            default_interaction_type="random",
        )
        # alone
        td = TensorDict(loc=torch.randn(3, 4), batch_size=[3])
        td = prob(td)
        assert "sample" in td
        assert td.shape == (2, 3)
        assert td["sample"].shape == (2, 3, 4)
        if return_log_prob:
            assert "sample_log_prob" in td
        assert prob.dist_sample_keys == ["sample"]
        assert prob.dist_params_keys == ["loc"]

    @pytest.mark.parametrize("return_log_prob", [True, False])
    @pytest.mark.parametrize("return_composite", [True, False])
    @set_composite_lp_aggregate(False)
    def test_probabilistic_seq_n_samples(
        self,
        return_log_prob,
        return_composite,
    ):
        aggregate_probabilities = composite_lp_aggregate()
        prob = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1},
            return_log_prob=return_log_prob,
            num_samples=2,
            default_interaction_type="random",
        )
        # in a sequence
        seq = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
            prob,
            return_composite=return_composite,
        )
        td = TensorDict(x=torch.randn(3, 4), batch_size=[3])
        if return_composite:
            assert isinstance(seq.get_dist(td), CompositeDistribution)
        else:
            assert isinstance(seq.get_dist(td), Normal)
        td = seq(td)
        assert "sample" in td
        assert td.shape == (2, 3)
        assert td["sample"].shape == (2, 3, 4)
        if return_log_prob:
            assert "sample_log_prob" in td

        # log-prob from the sequence
        log_prob = seq.log_prob(td)
        if aggregate_probabilities or not return_composite:
            assert isinstance(log_prob, torch.Tensor)
        else:
            assert isinstance(log_prob, TensorDict)
        assert seq.dist_sample_keys == ["sample"]
        assert seq.dist_params_keys == ["loc"]

    @pytest.mark.parametrize("return_log_prob", [True, False])
    @pytest.mark.parametrize("return_composite", [True])
    @set_composite_lp_aggregate(False)
    def test_intermediate_probabilistic_seq_n_samples(
        self,
        return_log_prob,
        return_composite,
    ):
        prob = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1},
            return_log_prob=return_log_prob,
            num_samples=2,
            default_interaction_type="random",
        )
        aggregate_probabilities = composite_lp_aggregate()
        # intermediate in a sequence
        seq = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
            prob,
            TensorDictModule(
                lambda x: x + 1, in_keys=["sample"], out_keys=["new_sample"]
            ),
            return_composite=return_composite,
        )
        td = TensorDict(x=torch.randn(3, 4), batch_size=[3])
        assert isinstance(seq.get_dist(td), CompositeDistribution)
        td = seq(td)
        assert "sample" in td
        assert td.shape == (2, 3)
        assert td["sample"].shape == (2, 3, 4)
        if return_log_prob:
            assert "sample_log_prob" in td

        # log-prob from the sequence
        log_prob = seq.log_prob(td)
        if aggregate_probabilities or not return_composite:
            assert isinstance(log_prob, torch.Tensor)
        else:
            assert isinstance(log_prob, TensorDict)
        assert seq.dist_sample_keys == ["sample"]
        assert seq.dist_params_keys == ["loc"]

    @pytest.mark.parametrize(
        "log_prob_key",
        [
            None,
            "sample_log_prob",
            ("nested", "sample_log_prob"),
            ("data", "sample_log_prob"),
        ],
    )
    @set_composite_lp_aggregate(False)  # not a legacy test
    def test_nested_keys_probabilistic_delta(self, log_prob_key):
        policy_module = TensorDictModule(
            nn.Linear(1, 1), in_keys=[("data", "states")], out_keys=[("data", "param")]
        )
        td = TensorDict(
            {"data": TensorDict({"states": torch.zeros(3, 4, 1)}, [3, 4])}, [3]
        )

        module = ProbabilisticTensorDictModule(
            in_keys=[("data", "param")],
            out_keys=[("data", "action")],
            distribution_class=Delta,
            return_log_prob=True,
            log_prob_key=log_prob_key,
        )
        assert module.dist_sample_keys == [("data", "action")]
        assert module.dist_params_keys == [("data", "param")]
        td_out = module(policy_module(td))
        assert td_out["data", "action"].shape == (3, 4, 1)
        if log_prob_key:
            assert td_out[log_prob_key].shape == (3, 4)
        else:
            assert td_out[module.log_prob_key].shape == (3, 4)

        module = ProbabilisticTensorDictModule(
            in_keys={"param": ("data", "param")},
            out_keys=[("data", "action")],
            distribution_class=Delta,
            return_log_prob=True,
            log_prob_key=log_prob_key,
        )
        assert module.dist_sample_keys == [("data", "action")]
        assert module.dist_params_keys == [("data", "param")]
        td_out = module(policy_module(td))
        assert td_out["data", "action"].shape == (3, 4, 1)
        if log_prob_key:
            assert td_out[log_prob_key].shape == (3, 4)
        else:
            assert td_out[module.log_prob_key].shape == (3, 4)

    @pytest.mark.parametrize(
        "log_prob_key",
        [
            None,
            "sample_log_prob",
            ("nested", "sample_log_prob"),
            ("data", "sample_log_prob"),
        ],
    )
    @set_composite_lp_aggregate(False)
    def test_nested_keys_probabilistic_normal(self, log_prob_key):
        loc_module = TensorDictModule(
            nn.Linear(1, 1),
            in_keys=[("data", "states")],
            out_keys=[("data", "loc")],
        )
        scale_module = TensorDictModule(
            nn.Linear(1, 1),
            in_keys=[("data", "states")],
            out_keys=[("data", "scale")],
        )
        scale_module.module.weight.data.abs_()
        scale_module.module.bias.data.abs_()
        td = TensorDict(
            {"data": TensorDict({"states": torch.zeros(3, 4, 1)}, [3, 4])}, [3]
        )

        module = ProbabilisticTensorDictModule(
            in_keys=[("data", "loc"), ("data", "scale")],
            out_keys=[("data", "action")],
            distribution_class=Normal,
            return_log_prob=True,
            log_prob_key=log_prob_key,
        )
        td_out = module(loc_module(scale_module(td)))
        assert td_out["data", "action"].shape == (3, 4, 1)
        if log_prob_key:
            assert td_out[log_prob_key].shape == (3, 4, 1)
        else:
            assert td_out[module.log_prob_key].shape == (3, 4, 1)

        module = ProbabilisticTensorDictModule(
            in_keys={"loc": ("data", "loc"), "scale": ("data", "scale")},
            out_keys=[("data", "action")],
            distribution_class=Normal,
            return_log_prob=True,
            log_prob_key=log_prob_key,
        )
        td_out = module(loc_module(scale_module(td)))
        assert td_out["data", "action"].shape == (3, 4, 1)
        if log_prob_key:
            assert td_out[log_prob_key].shape == (3, 4, 1)
        else:
            assert td_out[module.log_prob_key].shape == (3, 4, 1)

    def test_index_prob_seq(self):
        m0 = ProbabilisticTensorDictModule(
            in_keys=["loc"], out_keys=["sample"], distribution_class=Normal
        )
        m1 = TensorDictModule(lambda x: x, in_keys=["other"], out_keys=["something"])
        m2 = ProbabilisticTensorDictModule(
            in_keys=["scale"], out_keys=["sample2"], distribution_class=Normal
        )
        seq = ProbabilisticTensorDictSequential(m0, m1, m2)
        assert isinstance(seq[0], ProbabilisticTensorDictModule)
        assert isinstance(seq[:2], TensorDictSequential)
        assert not isinstance(seq[:2], ProbabilisticTensorDictSequential)
        assert isinstance(seq[-2:], ProbabilisticTensorDictSequential)

        seq = ProbabilisticTensorDictSequential(m0, m1, m2, return_composite=True)
        assert isinstance(seq[0], ProbabilisticTensorDictModule)
        assert isinstance(seq[:2], ProbabilisticTensorDictSequential)
        assert isinstance(seq[-2:], ProbabilisticTensorDictSequential)

    def test_no_warning_single_key(self):
        # Check that there is no warning if the number of out keys is 1 and sample log prob is set
        torch.manual_seed(0)
        with set_composite_lp_aggregate(None):
            mod = ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                distribution_class=torch.distributions.Normal,
                out_keys=[("an", "action")],
                log_prob_key="sample_log_prob",
                return_log_prob=True,
            )
            td = TensorDict(loc=torch.randn(()), scale=torch.rand(()))
            mod(td.copy())
            mod.log_prob(mod(td.copy()))
            mod.log_prob_key

            # Don't set the key and trigger the warning
            mod = ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                distribution_class=torch.distributions.Normal,
                out_keys=[("an", "action")],
                return_log_prob=True,
            )
            with pytest.warns(
                DeprecationWarning, match="You are querying the log-probability key"
            ):
                mod(td.copy())
                mod.log_prob(mod(td.copy()))
                mod.log_prob_key

            # add another variable, and trigger the warning
            mod = ProbabilisticTensorDictModule(
                in_keys=["params"],
                distribution_class=CompositeDistribution,
                distribution_kwargs={
                    "distribution_map": {
                        "dirich": torch.distributions.Dirichlet,
                        "categ": torch.distributions.Categorical,
                    }
                },
                out_keys=[("dirich", "categ")],
                return_log_prob=True,
            )
            with pytest.warns(
                DeprecationWarning, match="You are querying the log-probability key"
            ), pytest.warns(
                DeprecationWarning,
                match="Composite log-prob aggregation wasn't defined explicitly",
            ):
                td = TensorDict(
                    params=TensorDict(
                        dirich=TensorDict(
                            concentration=torch.rand(
                                (
                                    10,
                                    11,
                                )
                            )
                        ),
                        categ=TensorDict(logits=torch.rand((5,))),
                    )
                )
                mod(td.copy())
                mod.log_prob(mod(td.copy()))
                mod.log_prob_key


class TestEnsembleModule:
    def test_init(self):
        """Ensure that we correctly initialize copied weights s.t. they are not identical
        to the original weights."""
        torch.manual_seed(0)
        module = TensorDictModule(
            nn.Sequential(
                nn.Linear(2, 3),
                nn.ReLU(),
                nn.Linear(3, 1),
            ),
            in_keys=["a"],
            out_keys=["b"],
        )
        mod = EnsembleModule(module, num_copies=2)
        for param in mod.params_td.values(True, True):
            p0, p1 = param.unbind(0)
            assert not torch.allclose(
                p0, p1
            ), f"Ensemble params were not initialized correctly {p0}, {p1}"

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    @pytest.mark.parametrize(
        "net",
        [
            nn.Linear(1, 1),
            nn.Sequential(nn.Linear(1, 1)),
            nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1)),
        ],
    )
    def test_siso_forward(self, net):
        """Ensure that forward works for a single input and output"""
        module = TensorDictModule(
            net,
            in_keys=["bork"],
            out_keys=["dork"],
        )
        mod = EnsembleModule(module, num_copies=2)
        td = TensorDict({"bork": torch.randn(5, 1)}, batch_size=[5])
        out = mod(td)
        assert "dork" in out.keys(), "Ensemble forward failed to write keys"
        assert out["dork"].shape == torch.Size(
            [2, 5, 1]
        ), "Ensemble forward failed to expand input"
        outs = out["dork"].unbind(0)
        assert not torch.allclose(outs[0], outs[1]), "Outputs should be different"

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    @pytest.mark.parametrize(
        "net",
        [
            nn.Linear(1, 1),
            nn.Sequential(nn.Linear(1, 1)),
            nn.Sequential(nn.Linear(1, 1), nn.ReLU(), nn.Linear(1, 1)),
        ],
    )
    def test_chained_ensembles(self, net):
        """Ensure that the expand_input argument works"""
        module = TensorDictModule(net, in_keys=["bork"], out_keys=["dork"])
        next_module = TensorDictModule(
            copy.deepcopy(net), in_keys=["dork"], out_keys=["spork"]
        )
        e0 = EnsembleModule(module, num_copies=4, expand_input=True)
        e1 = EnsembleModule(next_module, num_copies=4, expand_input=False)
        seq = TensorDictSequential(e0, e1)
        td = TensorDict({"bork": torch.randn(5, 1)}, batch_size=[5])
        out = seq(td)

        for out_key in ["dork", "spork"]:
            assert out_key in out.keys(), f"Ensemble forward failed to write {out_key}"
            assert out[out_key].shape == torch.Size(
                [4, 5, 1]
            ), f"Ensemble forward failed to expand input for {out_key}"
            same_outputs = torch.isclose(
                out[out_key].repeat(4, 1, 1), out[out_key].repeat_interleave(4, dim=0)
            ).reshape(4, 4, 5, 1)
            mask_out_diags = torch.eye(4).logical_not()
            assert not torch.any(
                same_outputs[mask_out_diags]
            ), f"Module ensemble outputs should be different for {out_key}"

    def test_reset_once(self):
        """Ensure we only call reset_parameters() once per ensemble member"""
        lin = nn.Linear(1, 1)
        lin.reset_parameters = unittest.mock.Mock()
        module = TensorDictModule(
            nn.Sequential(lin),
            in_keys=["a"],
            out_keys=["b"],
        )
        EnsembleModule(module, num_copies=2)
        assert (
            lin.reset_parameters.call_count == 2
        ), f"Reset parameters called {lin.reset_parameters.call_count} times should be 2"


class TestTensorDictParams:
    def _get_params(self):
        module = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4))
        params = TensorDict.from_module(module)
        params.lock_()
        return params

    class CustomModule(nn.Module):
        def __init__(self, *params):
            super().__init__()
            if len(params) == 1:
                params = params[0]
                self.params = params
            else:
                for i, p in enumerate(params):
                    setattr(self, f"params{i}", p)

    def test_td_params(self):
        params = self._get_params()
        p = TensorDictParams(params)
        m = self.CustomModule(p)
        assert (
            TensorDict(dict(m.named_parameters()), [])
            == TensorDict({"params": params.flatten_keys(".")}, []).flatten_keys(".")
        ).all()

        assert not m.params.is_locked
        assert m.params._param_td.is_locked

        assert (
            m.params["0", "weight"] is not None
        )  # assess that param can be accessed via nested indexing

        # assert assignment
        m.params["other"] = torch.randn(3)
        assert isinstance(m.params["other"], nn.Parameter)
        assert m.params["other"].requires_grad

        # change that locking is unchanged
        assert not m.params.is_locked
        assert m.params._param_td.is_locked

        assert m.params.other.requires_grad
        del m.params["other"]

        assert m.params["0", "weight"].requires_grad
        assert (m.params == params).all()
        assert (params == m.params).all()

    def test_constructors(self):
        class MyModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_parameter(
                    "param", nn.Parameter(torch.randn(3, requires_grad=True))
                )
                self.register_buffer("buf", torch.randn(3))
                self.register_buffer("buf_int", torch.randint(3, ()))

        td = TensorDict.from_module(MyModule())
        assert not isinstance(td, TensorDictParams)
        td = TensorDictParams(td)
        assert isinstance(td, TensorDictParams)
        assert isinstance(td["param"], nn.Parameter)
        assert isinstance(td["buf"], nn.Parameter)
        assert isinstance(td["buf_int"], Buffer)
        td = TensorDict.from_module(MyModule())
        assert not isinstance(td, TensorDictParams)
        td = TensorDictParams(td, no_convert=True)
        assert isinstance(td, TensorDictParams)
        assert isinstance(td["param"], nn.Parameter)
        assert isinstance(td["buf"], Buffer)
        assert isinstance(td["buf_int"], Buffer)

        td = TensorDict.from_module(MyModule(), as_module=True)
        assert isinstance(td, TensorDictParams)
        assert isinstance(td["param"], nn.Parameter)
        assert isinstance(td["buf"], Buffer)
        assert isinstance(td["buf_int"], Buffer)

        tdparams = TensorDictParams(a=0, b=1.0)
        assert isinstance(tdparams["a"], Buffer)
        assert isinstance(tdparams["b"], nn.Parameter)

        tdparams = TensorDictParams({"a": 0, "b": 1.0})
        assert isinstance(tdparams["a"], Buffer)
        assert isinstance(tdparams["b"], nn.Parameter)
        tdparams_copy = tdparams.copy()

        def assert_is_identical(a, b):
            assert a is b

        tdparams.apply(assert_is_identical, tdparams_copy, filter_empty=True)

    def test_td_params_cast(self):
        params = self._get_params()
        p = TensorDictParams(params)
        m = self.CustomModule(p)
        for dtype in ("half", "double", "float"):
            getattr(m, dtype)()
            for p in params.values(True, True):
                assert p.dtype == getattr(torch, dtype)

    def test_td_params_tying(self):
        params = self._get_params()
        p1 = TensorDictParams(params)
        p2 = TensorDictParams(params)
        m = self.CustomModule(p1, p2)
        for key in dict(m.named_parameters()).keys():
            assert key.startswith("params0")

    def test_td_params_post_hook(self):
        hook = lambda self, x: x.data
        td = TensorDict(
            {
                "a": {
                    "b": {"c": torch.zeros((), requires_grad=True)},
                    "d": torch.zeros((), requires_grad=True),
                },
                "e": torch.zeros((), requires_grad=True),
            },
            [],
        )
        param_td = TensorDictParams(td)
        param_td.register_get_post_hook(hook)
        assert all(p.requires_grad for p in td.values(True, True))
        assert all(not p.requires_grad for p in param_td.values(True, True))
        assert {p.data.data_ptr() for p in param_td.values(True, True)} == {
            p.data.data_ptr() for p in td.values(True, True)
        }
        assert not param_td["e"].requires_grad
        assert not param_td["a", "b", "c"].requires_grad
        assert not param_td.get("e").requires_grad
        assert not param_td.get(("a", "b", "c")).requires_grad

    def test_tdparams_clone(self):
        td = TensorDict(
            {
                "a": {
                    "b": {"c": nn.Parameter(torch.zeros((), requires_grad=True))},
                    "d": Buffer(torch.zeros((), requires_grad=False)),
                },
                "e": nn.Parameter(torch.zeros((), requires_grad=True)),
                "f": Buffer(torch.zeros((), requires_grad=False)),
            },
            [],
        )
        td = TensorDictParams(td, no_convert=True)
        tdclone = td.clone()
        assert type(tdclone) == type(td)  # noqa
        for key, val in tdclone.items(True, True):
            assert type(val) == type(td.get(key))  # noqa
            assert val.requires_grad == td.get(key).requires_grad
            assert val.data_ptr() != td.get(key).data_ptr()
            assert (val == td.get(key)).all()

    def test_tdparams_clone_tying(self):
        c = nn.Parameter(torch.zeros((), requires_grad=True))
        td = TensorDict(
            {
                "a": {
                    "b": {"c": c},
                },
                "c": c,
            },
            [],
        )
        td = TensorDictParams(td, no_convert=True)
        td_clone = td.clone()
        assert td_clone["c"] is td_clone["a", "b", "c"]

    @pytest.mark.parametrize("with_batch", [False, True])
    def test_func_on_tdparams(self, with_batch):
        # tdparams isn't represented in a nested way, so we must check that calling to_module on it works ok
        net = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sequential(
                nn.Linear(2, 2),
                nn.Dropout(),
                nn.BatchNorm1d(2),
                nn.Sequential(
                    nn.Tanh(),
                    nn.Linear(2, 2),
                ),
            ),
        )

        if with_batch:
            params = TensorDict.from_modules(net, net, as_module=True)
            params0 = params[0].expand(3).clone().apply(lambda x: x.data * 0)
        else:
            params = TensorDict.from_module(net, as_module=True)
            params0 = params.apply(lambda x: x.data * 0)

        assert (params0 == 0).all()
        with params0.to_module(params):
            assert (params == 0).all()
        assert not (params == 0).all()

        # Now with a module around it
        class MyModule(nn.Module):
            pass

        m = MyModule()
        m.params = params
        params_m = TensorDict.from_module(m, as_module=True)
        if with_batch:
            params_m0 = params_m.clone()
            params_m0["params"] = params_m0["params"][0].expand(3).clone()
        else:
            params_m0 = params_m
        params_m0 = params_m0.apply(lambda x: x.data * 0)
        assert (params_m0 == 0).all()
        assert not (params_m == 0).all()
        with params_m0.to_module(m):
            assert (params_m == 0).all()
        assert not (params_m == 0).all()

    def test_load_state_dict(self):
        net = nn.Sequential(
            nn.Linear(2, 2),
            nn.Sequential(
                nn.Linear(2, 2),
                nn.Dropout(),
                nn.BatchNorm1d(2),
                nn.Sequential(
                    nn.Tanh(),
                    nn.Linear(2, 2),
                ),
            ),
        )

        params = TensorDict.from_module(net, as_module=True)
        assert any(isinstance(p, nn.Parameter) for p in params.values(True, True))
        weakrefs = {weakref.ref(t) for t in params.values(True, True)}

        # Now with a module around it
        class MyModule(nn.Module):
            pass

        module = MyModule()
        module.model = MyModule()
        module.model.params = params
        sd = module.state_dict()
        sd = {
            key: val * 0 if isinstance(val, torch.Tensor) else val
            for key, val in sd.items()
        }
        module.load_state_dict(sd)
        assert (params == 0).all()
        assert any(isinstance(p, nn.Parameter) for p in params.values(True, True))
        assert weakrefs == {weakref.ref(t) for t in params.values(True, True)}

    def test_inplace_ops(self):
        td = TensorDict(
            {
                "a": {
                    "b": {"c": torch.zeros((), requires_grad=True)},
                    "d": torch.zeros((), requires_grad=True),
                },
                "e": torch.zeros((), requires_grad=True),
            },
            [],
        )
        param_td = TensorDictParams(td)
        param_td.copy_(param_td.data.apply(lambda x: x + 1))
        assert (param_td == 1).all()
        param_td.zero_()
        assert (param_td == 0).all()


class TestCompositeDist:
    def test_const(self):
        params = TensorDict(
            {
                "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
                ("nested", "disc"): {"logits": torch.randn(3, 10)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.Categorical,
            },
        )
        assert dist.batch_shape == params.shape
        assert len(dist.dists) == 2
        assert "cont" in dist.dists
        assert ("nested", "disc") in dist.dists

    def test_sample(self):
        params = TensorDict(
            {
                "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
                ("nested", "disc"): {"logits": torch.randn(3, 10)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.Categorical,
            },
        )
        sample = dist.sample()
        assert sample.shape == params.shape
        sample = dist.sample((4,))
        assert sample.shape == torch.Size((4,) + params.shape)

    def test_set_composite_lp_aggregate(self):
        d = torch.distributions
        params = TensorDict(
            {
                "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
                ("nested", "disc"): {"logits": torch.randn(3, 10)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical},
        )
        sample = dist.sample((4,))
        with set_composite_lp_aggregate(False):
            lp = dist.log_prob(sample)
            assert isinstance(lp, TensorDict)

        with set_composite_lp_aggregate(True):
            lp = dist.log_prob(sample)
            assert isinstance(lp, torch.Tensor)

    @pytest.mark.parametrize("mode", [None, True, False])
    def test_set_composite_lp_aggregate_build_and_get(self, mode):
        d = torch.distributions
        dist_maker = functools.partial(
            CompositeDistribution,
            distribution_map={"cont": d.Normal, ("nested", "disc"): d.Categorical},
        )
        with set_composite_lp_aggregate(mode):
            p = ProbabilisticTensorDictModule(
                in_keys=["params"],
                out_keys=["cont", ("nested", "disc")],
                distribution_class=dist_maker,
                return_log_prob=True,
            )
            if composite_lp_aggregate(nowarn=True):
                with (
                    pytest.warns(DeprecationWarning)
                    if mode is None
                    else contextlib.nullcontext()
                ):
                    assert p.log_prob_key == "sample_log_prob"
            else:
                assert p.log_prob_keys == ["cont_log_prob", ("nested", "disc_log_prob")]

        if mode in (True, None):
            with set_composite_lp_aggregate(True):
                assert p.out_keys == ["cont", ("nested", "disc"), "sample_log_prob"]
                assert p.log_prob_key == "sample_log_prob"
                assert p.log_prob_keys == ["sample_log_prob"]
            with set_composite_lp_aggregate(False):
                with pytest.raises(RuntimeError):
                    p.out_keys
                with pytest.raises(RuntimeError):
                    p.log_prob_key
                with pytest.raises(RuntimeError):
                    p.log_prob_keys
        else:
            with set_composite_lp_aggregate(False):
                assert p.out_keys == [
                    "cont",
                    ("nested", "disc"),
                    "cont_log_prob",
                    ("nested", "disc_log_prob"),
                ]
                with pytest.raises(RuntimeError):
                    p.log_prob_key
                assert p.log_prob_keys == ["cont_log_prob", ("nested", "disc_log_prob")]
            with set_composite_lp_aggregate(True):
                with pytest.raises(RuntimeError):
                    p.out_keys
                with pytest.raises(RuntimeError):
                    p.log_prob_key
                with pytest.raises(RuntimeError):
                    p.log_prob_keys

    @set_composite_lp_aggregate(False)
    def test_from_distributions(self):

        # Values are not used to build the dists
        params = TensorDict(
            {
                ("0", "loc"): None,
                ("1", "nested", "loc"): None,
                ("0", "scale"): None,
                ("1", "nested", "scale"): None,
            }
        )
        d0 = torch.distributions.Normal(0, 1)
        d1 = torch.distributions.Normal(torch.zeros(1, 2), torch.ones(1, 2))

        d = CompositeDistribution.from_distributions(
            params, {"0": d0, ("1", "nested"): d1}
        )
        s = d.sample()
        assert s["0"].shape == ()
        assert s["1", "nested"].shape == (1, 2)
        assert isinstance(s["0"], torch.Tensor)
        assert isinstance(s["1", "nested"], torch.Tensor)

    def test_sample_named(self):
        params = TensorDict(
            {
                "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
                ("nested", "disc"): {"logits": torch.randn(3, 10)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.Categorical,
            },
            name_map={
                "cont": ("sample", "cont"),
                (("nested",), "disc"): ("sample", "disc"),
            },
        )
        sample = dist.sample()
        assert sample.shape == params.shape
        sample = dist.sample((4,))
        assert sample.shape == torch.Size((4,) + params.shape)
        assert sample["sample", "cont"].shape == torch.Size((4, 3, 4))
        assert sample["sample", "disc"].shape == torch.Size((4, 3))

    def test_rsample(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.RelaxedOneHotCategorical,
            },
            extra_kwargs={("nested", "disc"): {"temperature": torch.tensor(1.0)}},
        )
        sample = dist.rsample()
        assert sample.shape == params.shape
        assert sample.requires_grad
        sample = dist.rsample((4,))
        assert sample.shape == torch.Size((4,) + params.shape)
        assert sample.requires_grad

    @set_composite_lp_aggregate(True)
    def test_log_prob_legacy(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )
        with pytest.warns(DeprecationWarning, match="aggregate_probabilities"):
            CompositeDistribution(
                params,
                distribution_map={
                    "cont": distributions.Normal,
                    ("nested", "disc"): distributions.RelaxedOneHotCategorical,
                },
                extra_kwargs={("nested", "disc"): {"temperature": torch.tensor(1.0)}},
                aggregate_probabilities=True,
            )

        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.RelaxedOneHotCategorical,
            },
            extra_kwargs={("nested", "disc"): {"temperature": torch.tensor(1.0)}},
        )

        sample = dist.rsample((4,))
        lp = dist.log_prob(sample)
        assert isinstance(lp, torch.Tensor)
        assert lp.requires_grad

    @set_composite_lp_aggregate(False)
    def test_log_prob(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )
        with pytest.warns(DeprecationWarning, match="aggregate_probabilities"):
            CompositeDistribution(
                params,
                distribution_map={
                    "cont": distributions.Normal,
                    ("nested", "disc"): distributions.RelaxedOneHotCategorical,
                },
                extra_kwargs={("nested", "disc"): {"temperature": torch.tensor(1.0)}},
                aggregate_probabilities=True,
            )

        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.RelaxedOneHotCategorical,
            },
            extra_kwargs={("nested", "disc"): {"temperature": torch.tensor(1.0)}},
        )

        sample = dist.rsample((4,))
        lp = dist.log_prob(sample)
        assert isinstance(lp, TensorDict)
        assert lp.requires_grad

    @set_composite_lp_aggregate(True)
    def test_log_prob_composite_legacy(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.RelaxedOneHotCategorical,
            },
            extra_kwargs={("nested", "disc"): {"temperature": torch.tensor(1.0)}},
        )
        sample = dist.rsample((4,))
        sample_lp = dist.log_prob_composite(sample)
        assert isinstance(sample_lp, TensorDict)

    @set_composite_lp_aggregate(False)
    def test_log_prob_composite(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.RelaxedOneHotCategorical,
            },
            extra_kwargs={("nested", "disc"): {"temperature": torch.tensor(1.0)}},
        )
        sample = dist.rsample((4,))
        sample_lp = dist.log_prob_composite(sample)
        assert sample_lp.get("cont_log_prob").requires_grad
        assert sample_lp.get(("nested", "disc_log_prob")).requires_grad
        assert sample_lp is not sample
        assert "sample_log_prob" not in sample_lp.keys()

    @set_composite_lp_aggregate(True)
    def test_entropy_legacy(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )

        with pytest.warns(DeprecationWarning, match="aggregate_probabilities"):
            CompositeDistribution(
                params,
                distribution_map={
                    "cont": distributions.Normal,
                    ("nested", "disc"): distributions.Categorical,
                },
                aggregate_probabilities=True,
            )

        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.Categorical,
            },
        )
        ent = dist.entropy()
        assert ent.shape == params.shape == dist._batch_shape
        assert isinstance(ent, torch.Tensor)
        assert ent.requires_grad

    @set_composite_lp_aggregate(False)
    def test_entropy(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )

        with pytest.warns(DeprecationWarning, match="aggregate_probabilities"):
            CompositeDistribution(
                params,
                distribution_map={
                    "cont": distributions.Normal,
                    ("nested", "disc"): distributions.Categorical,
                },
                aggregate_probabilities=True,
            )

        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.Categorical,
            },
        )
        ent = dist.entropy()
        assert ent.shape == params.shape == dist._batch_shape
        assert isinstance(ent, TensorDict)
        assert ent.requires_grad

    @set_composite_lp_aggregate(True)
    def test_entropy_composite_legacy(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.Categorical,
            },
        )
        sample = dist.entropy()
        assert isinstance(sample, torch.Tensor)

    @set_composite_lp_aggregate(False)
    def test_entropy_composite(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "disc"): {"logits": torch.randn(3, 10, requires_grad=True)},
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "disc"): distributions.Categorical,
            },
        )
        sample = dist.entropy()
        assert sample.shape == params.shape == dist._batch_shape
        assert sample.get("cont_entropy").requires_grad
        assert sample.get(("nested", "disc_entropy")).requires_grad
        assert "entropy" not in sample.keys()

    def test_cdf(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "cont"): {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "cont"): distributions.Normal,
            },
        )
        sample = dist.rsample((4,))
        sample = dist.cdf(sample)
        assert sample.get("cont_cdf").requires_grad
        assert sample.get(("nested", "cont_cdf")).requires_grad

    def test_icdf(self):
        params = TensorDict(
            {
                "cont": {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
                ("nested", "cont"): {
                    "loc": torch.randn(3, 4, requires_grad=True),
                    "scale": torch.rand(3, 4, requires_grad=True),
                },
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "cont": distributions.Normal,
                ("nested", "cont"): distributions.Normal,
            },
        )
        sample = dist.rsample((4,))
        sample = dist.cdf(sample)
        sample = dist.icdf(sample)
        assert sample.get("cont_icdf").requires_grad
        assert sample.get(("nested", "cont_icdf")).requires_grad
        torch.testing.assert_close(sample.get("cont"), sample.get("cont_icdf"))

    @pytest.mark.parametrize(
        "interaction", [InteractionType.MODE, InteractionType.MEAN]
    )
    @pytest.mark.parametrize("return_log_prob", [True, False])
    @pytest.mark.parametrize("map_names", [True, False])
    @set_composite_lp_aggregate(True)
    def test_prob_module_legacy(self, interaction, return_log_prob, map_names):
        params = TensorDict(
            {
                "params": {
                    "cont": {
                        "loc": torch.randn(3, 4, requires_grad=True),
                        "scale": torch.rand(3, 4, requires_grad=True),
                    },
                    ("nested", "cont"): {
                        "loc": torch.randn(3, 4, requires_grad=True),
                        "scale": torch.rand(3, 4, requires_grad=True),
                    },
                }
            },
            batch_size=(3,),
        )
        in_keys = ["params"]
        out_keys = ["cont", ("nested", "cont")]
        distribution_map = {
            "cont": distributions.Normal,
            ("nested", "cont"): distributions.Normal,
        }
        distribution_kwargs = {"distribution_map": distribution_map}
        if map_names:
            distribution_kwargs.update(
                {
                    "name_map": {
                        "cont": ("sample", "cont"),
                        ("nested", "cont"): ("sample", "nested", "cont"),
                    }
                }
            )
            out_keys = list(distribution_kwargs["name_map"].values())
        module = ProbabilisticTensorDictModule(
            in_keys=in_keys,
            out_keys=None,
            distribution_class=CompositeDistribution,
            distribution_kwargs=distribution_kwargs,
            default_interaction_type=interaction,
            return_log_prob=return_log_prob,
        )
        assert module.dist_sample_keys == out_keys
        assert module.dist_params_keys == in_keys
        if not return_log_prob:
            assert module.out_keys[-2:] == out_keys
        else:
            # loosely checks that the log-prob keys have been added
            assert module.out_keys[-2:] != out_keys

        assert module.log_prob_key == "sample_log_prob"
        assert module.log_prob_keys == ["sample_log_prob"]
        sample = module(params)

        key_logprob0 = ("sample", "cont_log_prob") if map_names else "cont_log_prob"
        key_logprob1 = (
            ("sample", "nested", "cont_log_prob")
            if map_names
            else ("nested", "cont_log_prob")
        )
        if return_log_prob:
            assert key_logprob0 in sample
            assert key_logprob1 in sample
            assert module.log_prob_key in sample, list(sample.keys(True, True))

        assert all(key in sample for key in module.out_keys)
        sample_clone = sample.clone()
        lp = module.log_prob(sample_clone)
        assert isinstance(lp, torch.Tensor)
        assert lp.shape == sample_clone.shape
        if return_log_prob:
            torch.testing.assert_close(
                lp,
                sample.get(key_logprob0).sum(-1) + sample.get(key_logprob1).sum(-1),
            )
        else:
            torch.testing.assert_close(
                lp,
                sample_clone.get(key_logprob0).sum(-1)
                + sample_clone.get(key_logprob1).sum(-1),
            )

    @pytest.mark.parametrize(
        "interaction", [InteractionType.MODE, InteractionType.MEAN]
    )
    @pytest.mark.parametrize("return_log_prob", [True, False])
    @pytest.mark.parametrize("map_names", [True, False])
    @set_composite_lp_aggregate(False)
    def test_prob_module(self, interaction, return_log_prob, map_names):
        params = TensorDict(
            {
                "params": {
                    "cont": {
                        "loc": torch.randn(3, 4, requires_grad=True),
                        "scale": torch.rand(3, 4, requires_grad=True),
                    },
                    ("nested", "cont"): {
                        "loc": torch.randn(3, 4, requires_grad=True),
                        "scale": torch.rand(3, 4, requires_grad=True),
                    },
                }
            },
            batch_size=(3,),
        )
        in_keys = ["params"]
        out_keys = ["cont", ("nested", "cont")]
        distribution_map = {
            "cont": distributions.Normal,
            ("nested", "cont"): distributions.Normal,
        }
        distribution_kwargs = {"distribution_map": distribution_map}
        if map_names:
            distribution_kwargs.update(
                {
                    "name_map": {
                        "cont": ("sample", "cont"),
                        ("nested", "cont"): ("sample", "nested", "cont"),
                    }
                }
            )
            out_keys = list(distribution_kwargs["name_map"].values())
        module = ProbabilisticTensorDictModule(
            in_keys=in_keys,
            out_keys=None,
            distribution_class=CompositeDistribution,
            distribution_kwargs=distribution_kwargs,
            default_interaction_type=interaction,
            return_log_prob=return_log_prob,
        )
        assert module.dist_sample_keys == out_keys
        assert module.dist_params_keys == in_keys
        if not return_log_prob:
            assert module.out_keys[-2:] == out_keys
        else:
            # loosely checks that the log-prob keys have been added
            assert module.out_keys[-2:] != out_keys

        sample = module(params)
        key_logprob0 = ("sample", "cont_log_prob") if map_names else "cont_log_prob"
        key_logprob1 = (
            ("sample", "nested", "cont_log_prob")
            if map_names
            else ("nested", "cont_log_prob")
        )
        assert module.log_prob_keys == [key_logprob0, key_logprob1]
        with pytest.raises(RuntimeError):
            module.log_prob_key
        if return_log_prob:
            assert key_logprob0 in sample
            assert key_logprob1 in sample
            assert "sample_log_prob" not in sample
        assert all(key in sample for key in module.out_keys)
        sample_clone = sample.clone()
        lp = module.log_prob(sample_clone)
        assert is_tensor_collection(lp)
        assert key_logprob0 in lp
        assert key_logprob1 in lp

    @pytest.mark.parametrize(
        "interaction", [InteractionType.MODE, InteractionType.MEAN]
    )
    @pytest.mark.parametrize("map_names", [True, False])
    @set_composite_lp_aggregate(True)
    def test_prob_module_nested_legacy(self, interaction, map_names):
        params = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "params": {
                            "cont": {
                                "loc": torch.randn(3, 4, requires_grad=True),
                                "scale": torch.rand(3, 4, requires_grad=True),
                            },
                            ("nested", "cont"): {
                                "loc": torch.randn(3, 4, requires_grad=True),
                                "scale": torch.rand(3, 4, requires_grad=True),
                            },
                        }
                    },
                    batch_size=3,
                ),
                "done": torch.ones(1),
            }
        )
        in_keys = [("agents", "params")]
        out_keys = ["cont", ("nested", "cont")]
        distribution_map = {
            "cont": distributions.Normal,
            ("nested", "cont"): distributions.Normal,
        }
        distribution_kwargs = {
            "distribution_map": distribution_map,
            "log_prob_key": ("agents", "sample_log_prob"),
        }
        if map_names:
            distribution_kwargs.update(
                {
                    "name_map": {
                        "cont": ("sample", "agents", "cont"),
                        ("nested", "cont"): ("sample", "agents", "nested", "cont"),
                    }
                }
            )
            out_keys = list(distribution_kwargs["name_map"].values())
        module = ProbabilisticTensorDictModule(
            in_keys=in_keys,
            out_keys=None,
            distribution_class=CompositeDistribution,
            distribution_kwargs=distribution_kwargs,
            default_interaction_type=interaction,
            return_log_prob=True,
            log_prob_key=("agents", "sample_log_prob"),
        )
        # loosely checks that the log-prob keys have been added
        assert module.out_keys[-2:] != out_keys
        assert module.dist_sample_keys == out_keys
        assert module.dist_params_keys == in_keys

        sample = module(params)
        key_logprob0 = (
            ("sample", "agents", "cont_log_prob") if map_names else "cont_log_prob"
        )
        key_logprob1 = (
            ("sample", "agents", "nested", "cont_log_prob")
            if map_names
            else ("nested", "cont_log_prob")
        )
        assert key_logprob0 in sample
        assert key_logprob1 in sample
        assert all(key in sample for key in module.out_keys)

        lp = sample.get(module.log_prob_key)
        assert module.log_prob_key == ("agents", "sample_log_prob")
        assert isinstance(lp, torch.Tensor)
        torch.testing.assert_close(
            lp,
            sample.get(key_logprob0).sum(-1) + sample.get(key_logprob1).sum(-1),
        )

    @pytest.mark.parametrize(
        "interaction", [InteractionType.MODE, InteractionType.MEAN]
    )
    @pytest.mark.parametrize("map_names", [True, False])
    @set_composite_lp_aggregate(False)
    def test_prob_module_nested(self, interaction, map_names):
        params = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "params": {
                            "cont": {
                                "loc": torch.randn(3, 4, requires_grad=True),
                                "scale": torch.rand(3, 4, requires_grad=True),
                            },
                            ("nested", "cont"): {
                                "loc": torch.randn(3, 4, requires_grad=True),
                                "scale": torch.rand(3, 4, requires_grad=True),
                            },
                        }
                    },
                    batch_size=3,
                ),
                "done": torch.ones(1),
            }
        )
        in_keys = [("agents", "params")]
        out_keys = ["cont", ("nested", "cont")]
        distribution_map = {
            "cont": distributions.Normal,
            ("nested", "cont"): distributions.Normal,
        }
        distribution_kwargs = {
            "distribution_map": distribution_map,
            "log_prob_key": ("agents", "sample_log_prob"),
        }
        if map_names:
            distribution_kwargs.update(
                {
                    "name_map": {
                        "cont": ("sample", "agents", "cont"),
                        ("nested", "cont"): ("sample", "agents", "nested", "cont"),
                    }
                }
            )
            out_keys = list(distribution_kwargs["name_map"].values())
        module = ProbabilisticTensorDictModule(
            in_keys=in_keys,
            out_keys=None,
            distribution_class=CompositeDistribution,
            distribution_kwargs=distribution_kwargs,
            default_interaction_type=interaction,
            return_log_prob=True,
        )
        # loosely checks that the log-prob keys have been added
        assert module.out_keys[-2:] != out_keys
        assert module.dist_sample_keys == out_keys
        assert module.dist_params_keys == in_keys

        sample = module(params)
        key_logprob0 = (
            ("sample", "agents", "cont_log_prob") if map_names else "cont_log_prob"
        )
        key_logprob1 = (
            ("sample", "agents", "nested", "cont_log_prob")
            if map_names
            else ("nested", "cont_log_prob")
        )
        assert key_logprob0 in sample
        assert key_logprob1 in sample
        assert all(key in sample for key in module.out_keys)

        lp = sample.select(*module.log_prob_keys)
        assert lp[key_logprob0] is not None
        assert lp[key_logprob1] is not None

    @pytest.mark.parametrize(
        "interaction", [InteractionType.MODE, InteractionType.MEAN]
    )
    @pytest.mark.parametrize("return_log_prob", [True, False])
    @pytest.mark.parametrize("ordereddict", [True, False])
    @set_composite_lp_aggregate(True)
    def test_prob_module_seq_legacy(self, interaction, return_log_prob, ordereddict):
        params = TensorDict(
            {
                "params": {
                    "cont": {
                        "loc": torch.randn(3, 4, requires_grad=True),
                        "scale": torch.rand(3, 4, requires_grad=True),
                    },
                    ("nested", "cont"): {
                        "loc": torch.randn(3, 4, requires_grad=True),
                        "scale": torch.rand(3, 4, requires_grad=True),
                    },
                }
            },
            batch_size=(3,),
        )
        in_keys = ["params"]
        out_keys = ["cont", ("nested", "cont")]
        distribution_map = {
            "cont": distributions.Normal,
            ("nested", "cont"): distributions.Normal,
        }
        backbone = TensorDictModule(lambda: None, in_keys=[], out_keys=[])
        args = [
            backbone,
            ProbabilisticTensorDictModule(
                in_keys=in_keys,
                out_keys=out_keys,
                distribution_class=CompositeDistribution,
                distribution_kwargs={"distribution_map": distribution_map},
                default_interaction_type=interaction,
                return_log_prob=return_log_prob,
            ),
        ]
        if ordereddict:
            args = [
                OrderedDict(
                    backbone=args[0],
                    proba=args[1],
                )
            ]
        module = ProbabilisticTensorDictSequential(*args)
        assert module.dist_sample_keys == out_keys
        assert module.dist_params_keys == in_keys
        sample = module(params)
        if return_log_prob:
            assert "cont_log_prob" in sample.keys()
            assert ("nested", "cont_log_prob") in sample.keys(True)
        sample_clone = sample.clone()

        dist = module.get_dist(sample_clone)
        assert isinstance(dist, CompositeDistribution)

        sample_clone = sample.clone()
        lp = module.log_prob(sample_clone)

        if return_log_prob:
            torch.testing.assert_close(
                lp,
                sample.get("cont_log_prob").sum(-1)
                + sample.get(("nested", "cont_log_prob")).sum(-1),
            )
        else:
            torch.testing.assert_close(
                lp,
                sample_clone.get("cont_log_prob").sum(-1)
                + sample_clone.get(("nested", "cont_log_prob")).sum(-1),
            )

    @pytest.mark.parametrize(
        "interaction", [InteractionType.MODE, InteractionType.MEAN]
    )
    @set_composite_lp_aggregate(True)
    def test_prob_module_seq_nested_legacy(self, interaction):
        params = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "params": {
                            "cont": {
                                "loc": torch.randn(3, 4, requires_grad=True),
                                "scale": torch.rand(3, 4, requires_grad=True),
                            },
                            ("nested", "cont"): {
                                "loc": torch.randn(3, 4, requires_grad=True),
                                "scale": torch.rand(3, 4, requires_grad=True),
                            },
                        }
                    },
                    batch_size=3,
                ),
                "done": torch.ones(1),
            }
        )
        in_keys = [("agents", "params")]
        out_keys = ["cont", ("nested", "cont")]
        distribution_map = {
            "cont": distributions.Normal,
            ("nested", "cont"): distributions.Normal,
        }
        log_prob_key = ("agents", "sample_log_prob")
        backbone = TensorDictModule(lambda: None, in_keys=[], out_keys=[])
        module = ProbabilisticTensorDictSequential(
            backbone,
            ProbabilisticTensorDictModule(
                in_keys=in_keys,
                out_keys=out_keys,
                distribution_class=CompositeDistribution,
                distribution_kwargs={"distribution_map": distribution_map},
                default_interaction_type=interaction,
                return_log_prob=True,
                log_prob_key=log_prob_key,
            ),
        )
        assert module.dist_sample_keys == out_keys
        assert module.dist_params_keys == in_keys
        sample = module(params)
        assert "cont_log_prob" in sample.keys()
        assert ("nested", "cont_log_prob") in sample.keys(True)
        lp = sample[log_prob_key]
        torch.testing.assert_close(
            lp,
            sample.get("cont_log_prob").sum(-1)
            + sample.get(("nested", "cont_log_prob")).sum(-1),
        )

    @pytest.mark.parametrize(
        "interaction", [InteractionType.MODE, InteractionType.MEAN]
    )
    @set_composite_lp_aggregate(False)
    def test_prob_module_seq_nested(self, interaction):
        params = TensorDict(
            {
                "agents": TensorDict(
                    {
                        "params": {
                            "cont": {
                                "loc": torch.randn(3, 4, requires_grad=True),
                                "scale": torch.rand(3, 4, requires_grad=True),
                            },
                            ("nested", "cont"): {
                                "loc": torch.randn(3, 4, requires_grad=True),
                                "scale": torch.rand(3, 4, requires_grad=True),
                            },
                        }
                    },
                    batch_size=3,
                ),
                "done": torch.ones(1),
            }
        )
        in_keys = [("agents", "params")]
        out_keys = ["cont", ("nested", "cont")]
        distribution_map = {
            "cont": distributions.Normal,
            ("nested", "cont"): distributions.Normal,
        }
        log_prob_keys = ("cont_log_prob", ("nested", "cont_log_prob"))
        backbone = TensorDictModule(lambda: None, in_keys=[], out_keys=[])
        module = ProbabilisticTensorDictSequential(
            backbone,
            ProbabilisticTensorDictModule(
                in_keys=in_keys,
                out_keys=out_keys,
                distribution_class=CompositeDistribution,
                distribution_kwargs={"distribution_map": distribution_map},
                default_interaction_type=interaction,
                return_log_prob=True,
                log_prob_keys=log_prob_keys,
            ),
        )
        assert module.dist_sample_keys == out_keys
        assert module.dist_params_keys == in_keys
        sample = module(params)
        assert "cont_log_prob" in sample.keys()
        assert ("nested", "cont_log_prob") in sample.keys(True)


class TestAddStateIndependentNormalScale:
    def test_add_scale_basic(self, num_outputs=4):
        module = nn.Linear(3, num_outputs)
        module_normal = AddStateIndependentNormalScale(num_outputs)
        tensor = torch.randn(3)
        loc, scale = module_normal(module(tensor))
        assert loc.shape == (num_outputs,)
        assert scale.shape == (num_outputs,)
        assert (scale > 0).all()

    def test_add_scale_sequence(self, num_outputs=4):
        module = nn.LSTM(3, num_outputs)
        module_normal = AddStateIndependentNormalScale(num_outputs)
        tensor = torch.randn(4, 2, 3)
        loc, scale, others = module_normal(*module(tensor))
        assert loc.shape == (4, 2, num_outputs)
        assert scale.shape == (4, 2, num_outputs)
        assert (scale > 0).all()

    def test_add_scale_init_value(self, num_outputs=4):
        module = nn.Linear(3, num_outputs)
        init_value = 1.0
        module_normal = AddStateIndependentNormalScale(
            num_outputs,
            scale_mapping="relu",
            init_value=init_value,
        )
        tensor = torch.randn(3)
        loc, scale = module_normal(module(tensor))
        assert loc.shape == (num_outputs,)
        assert scale.shape == (num_outputs,)
        assert (scale == init_value).all()


class TestStateDict:
    @pytest.mark.parametrize("detach", [True, False])
    def test_sd_params(self, detach):
        td = TensorDict({"1": 1, "2": 2, "3": {"3": 3}}, [])
        td = TensorDictParams(td)
        if detach:
            sd = td.detach().clone().zero_().state_dict()
        else:
            sd = td.state_dict()
            sd = tree_map(lambda t: t * 0 if isinstance(t, torch.Tensor) else t, sd)
        # do some op to create a graph
        td.apply(lambda x: x + 1)
        # load the data
        td.load_state_dict(sd)
        # check that data has been loaded
        assert (td == 0).all()

    def test_sd_module(self):
        td = TensorDict({"1": 1.0, "2": 2.0, "3": {"3": 3.0}}, [])
        td = TensorDictParams(td)
        module = nn.Linear(3, 4)
        module.td = td

        sd = module.state_dict()
        assert "td.1" in sd
        assert "td.3.3" in sd
        sd = {k: v * 0 if isinstance(v, torch.Tensor) else v for k, v in sd.items()}

        # load the data
        module.load_state_dict(sd)

        # check that data has been loaded
        assert (module.td == 0).all()
        for val in td.values(True, True):
            assert isinstance(val, nn.Parameter)


@pytest.mark.parametrize("as_module", [True, False])
class TestToModule:
    @property
    def _transformer(self):
        # we use transformer because it's deep, has buffers etc.
        return nn.Transformer(d_model=8, dim_feedforward=8).eval()

    @property
    def _module_shared(self):
        # a module with the same layer appearing twice
        l0 = nn.Linear(8, 9)
        l1 = nn.Linear(9, 8)
        return nn.Sequential(
            l0,
            l1,
            nn.Sequential(
                l0,
            ),
        )

    @property
    def _tuple_x(self):
        x = torch.randn(2, 2, 8)
        return (x, x)

    @property
    def _x(self):
        return (torch.randn(2, 2, 8),)

    @pytest.mark.parametrize(
        "module_name,input_name",
        [["_module_shared", "_x"], ["_transformer", "_tuple_x"]],
    )
    @pytest.mark.parametrize("inplace", [True, False])
    def test_static(self, module_name, input_name, as_module, inplace):
        torch.manual_seed(0)
        module = getattr(self, module_name)
        x = getattr(self, input_name)
        params = TensorDict.from_module(module, as_module=as_module)
        if inplace:
            params = params.clone()
        params0 = params.clone().zero_()
        y = module(*x)
        params0.to_module(module, inplace=inplace)
        y0 = module(*x)
        # check identities
        for k, p1, p2 in zip(
            params0.keys(True, True),
            TensorDict.from_module(module).values(True, True),
            params0.values(True, True),
        ):
            if inplace:
                assert p1 is not p2, k
            else:
                assert p1 is p2, k
        params.to_module(module, inplace=inplace)
        # check identities
        for p1, p2 in zip(
            TensorDict.from_module(module).values(True, True),
            params.values(True, True),
        ):
            if inplace:
                assert p1 is not p2
            else:
                assert p1 is p2
        y1 = module(*x)
        torch.testing.assert_close(y, y1)
        assert (y0 == 0).all()
        assert (y0 != y1).all()

    @pytest.mark.parametrize(
        "module_name,input_name",
        [["_module_shared", "_x"], ["_transformer", "_tuple_x"]],
    )
    @pytest.mark.parametrize("inplace", [True, False])
    def test_cm(self, module_name, input_name, as_module, inplace):
        torch.manual_seed(0)
        module = getattr(self, module_name)
        x = getattr(self, input_name)
        params = TensorDict.from_module(module, as_module=as_module)
        params0 = params.clone().zero_()
        y = module(*x)
        with params0.to_module(module, inplace=inplace):
            y0 = module(*x)
            if as_module:
                # if as_module=False, params0 is not made of parameters anymore
                assert (params0 == TensorDict.from_module(module)).all()

                # check identities
                for p1, p2 in zip(
                    TensorDict.from_module(module).values(True, True),
                    params0.values(True, True),
                ):
                    if inplace:
                        assert p1 is not p2
                    else:
                        assert p1 is p2

        y1 = module(*x)
        torch.testing.assert_close(y, y1)
        assert (y0 == 0).all()
        assert (y0 != y1).all()
        assert (TensorDict.from_module(module) == params).all()

    @pytest.mark.parametrize(
        "module_name,input_name",
        [["_module_shared", "_x"], ["_transformer", "_tuple_x"]],
    )
    def test_cm_meta(self, module_name, input_name, as_module):
        torch.manual_seed(0)
        module = getattr(self, module_name)
        x = getattr(self, input_name)
        params = TensorDict.from_module(module, as_module=as_module)
        params_meta = params.detach().to("meta")
        y = module(*x)
        with params_meta.to_module(module):
            module_meta = copy.deepcopy(module)
        y1 = module(*x)
        with params.to_module(module_meta):
            y2 = module_meta(*x)
        torch.testing.assert_close(y, y1)
        torch.testing.assert_close(y, y2)
        assert (TensorDict.from_module(module) == params).all()

    def test_params_detach(self, as_module):
        class MyLinear(nn.Linear):
            def __setattr__(self, key, value):
                return super().__setattr__(key, value)

        linear = MyLinear(3, 4)
        params = TensorDict.from_module(linear, as_module=as_module)
        # this will break if the parameters are not deleted before being set
        with params.detach().to_module(linear):
            linear(torch.randn(3))
        assert len(list(linear.parameters())) == 2
        assert (TensorDict.from_module(linear) == params).all()


class TestAsTDM:
    @pytest.mark.parametrize("in_keys", ["c", ["c"]])
    @pytest.mark.parametrize("out_keys", ["d", ["d"]])
    def test_module(self, in_keys, out_keys):
        class A:
            @as_tensordict_module(in_keys=in_keys, out_keys=out_keys)
            def func(self, c):
                return c + 1

        a = A()
        assert a.func(TensorDict(c=0))["d"] == 1

    @pytest.mark.parametrize("in_keys", ["c", ["c"]])
    @pytest.mark.parametrize("out_keys", ["d", ["d"]])
    def test_free_func(self, in_keys, out_keys):
        @as_tensordict_module(in_keys=in_keys, out_keys=out_keys)
        def func(c):
            return c + 1

        assert func(TensorDict(c=0))["d"] == 1


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
