# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import copy
import functools
import importlib
import inspect
import os
import pathlib
import pickle
import re
import threading
import typing
import unittest
import warnings
import weakref
from collections import OrderedDict
from collections.abc import MutableSequence

import functorch
import pytest
import tensordict.nn.common
import tensordict.nn.distributions
import tensordict.nn.distributions.truncated_normal
import tensordict.nn.params
import tensordict.nn.probabilistic
import tensordict.nn.utils
import torch
from _utils_internal import is_npu_available
from functorch import make_functional_with_buffers as make_functional_functorch
from tensordict import (
    is_leaf_nontensor,
    is_tensor_collection,
    lazy_stack,
    NonTensorData,
    NonTensorStack,
    set_list_to_stack,
    tensorclass,
    TensorDict,
    TensorDictBase,
    unravel_key,
    unravel_key_list,
)
from tensordict.nn import (
    as_tensordict_module,
    dispatch,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorClassModuleBase,
    TensorDictModuleBase,
    TensorDictParams,
    TensorDictSequential,
)
from tensordict.nn.common import TensorDictModule, TensorDictModuleWrapper
from tensordict.nn.distributions import (
    AddStateIndependentNormalScale,
    Delta,
    NormalParamExtractor,
    OneHotCategorical,
    TruncatedNormal,
)
from tensordict.nn.distributions.composite import CompositeDistribution
from tensordict.nn.distributions.truncated_normal import TruncatedStandardNormal
from tensordict.nn.ensemble import EnsembleModule
from tensordict.nn.probabilistic import (
    interaction_type,
    InteractionType,
    set_interaction_type,
)
from tensordict.nn.tensorclass_module import InputClass, OutputClass
from tensordict.nn.utils import (
    _dispatch_td_nn_modules,
    _set_dispatch_td_nn_modules,
    composite_lp_aggregate,
    set_composite_lp_aggregate,
    set_skip_existing,
    skip_existing,
)
from tensordict.tensorclass import TensorClass
from torch import distributions, nn, vmap
from torch.distributions import Categorical, Normal
from torch.nn.parameter import Buffer
from torch.utils._pytree import tree_map

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

IS_FB = os.getenv("PYTORCH_TEST_FBCODE")

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


def get_device():
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif is_npu_available():
        device = torch.device("npu:0")
    elif torch.mps.is_available():
        device = torch.device("mps:0")
    return device


class TestInteractionType:
    def test_public_interaction_type(self):
        assert "interaction_type" in tensordict.nn.__all__
        assert tensordict.nn.interaction_type is interaction_type
        with set_interaction_type("mode"):
            assert tensordict.nn.interaction_type() == InteractionType.MODE

    def test_base(self):
        with set_interaction_type("DETERMINISTIC"):
            assert interaction_type() == InteractionType.DETERMINISTIC

        with set_interaction_type("MODE"):
            assert interaction_type() == InteractionType.MODE

        with set_interaction_type("MEDIAN"):
            assert interaction_type() == InteractionType.MEDIAN

        with set_interaction_type("MEAN"):
            assert interaction_type() == InteractionType.MEAN

        with set_interaction_type("RANDOM"):
            assert interaction_type() == InteractionType.RANDOM

        with set_interaction_type("mode"):
            assert interaction_type() == InteractionType.MODE

        with set_interaction_type("median"):
            assert interaction_type() == InteractionType.MEDIAN

        with set_interaction_type("mean"):
            assert interaction_type() == InteractionType.MEAN

        with set_interaction_type("random"):
            assert interaction_type() == InteractionType.RANDOM

        with set_interaction_type(InteractionType.DETERMINISTIC):
            assert interaction_type() == InteractionType.DETERMINISTIC

        with set_interaction_type(InteractionType.MODE):
            assert interaction_type() == InteractionType.MODE

        with set_interaction_type(InteractionType.MEDIAN):
            assert interaction_type() == InteractionType.MEDIAN

        with set_interaction_type(InteractionType.MEAN):
            assert interaction_type() == InteractionType.MEAN

        with set_interaction_type(InteractionType.RANDOM):
            assert interaction_type() == InteractionType.RANDOM

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

    # Each switch, the function that reads it, and two values with the mode
    # that each gives
    _SWITCHES = [
        (
            set_interaction_type,
            interaction_type,
            ("random", InteractionType.RANDOM),
            ("mean", InteractionType.MEAN),
        ),
        (set_skip_existing, skip_existing, (True, True), (["out"], ["out"])),
        (
            set_composite_lp_aggregate,
            composite_lp_aggregate,
            (True, True),
            (False, False),
        ),
    ]

    @pytest.mark.parametrize("switch, read, first, second", _SWITCHES)
    def test_thread_local(self, switch, read, first, second):
        # A thread_local block changes the mode of its own thread only, and
        # blocks that overlap in two threads restore their own thread's mode.
        base = read()
        in_block, release = threading.Event(), threading.Event()
        seen = {}

        def other_thread():
            with switch(first[0], thread_local=True):
                seen["inside"] = read()
                # a block without the flag stays in this thread too
                with switch(second[0]):
                    seen["nested"] = read()
                    in_block.set()
                    release.wait(10)
            seen["after"] = read()

        thread = threading.Thread(target=other_thread)
        thread.start()
        try:
            assert in_block.wait(10)
            assert read() == base
            with switch(second[0], thread_local=True):
                assert read() == second[1]
                release.set()
                thread.join(10)
                assert read() == second[1]
            assert read() == base
        finally:
            release.set()
            thread.join(10)
        assert seen == {"inside": first[1], "nested": second[1], "after": base}
        assert switch(first[0], thread_local=True).clone().thread_local

    @pytest.mark.parametrize("switch, read, first, second", _SWITCHES)
    def test_process_wide_default(self, switch, read, first, second):
        # Without thread_local, the mode that a thread sets applies to all
        # threads, as before.
        base = read()
        seen = {}

        def other_thread():
            seen["thread"] = read()

        with switch(first[0]):
            thread = threading.Thread(target=other_thread)
            thread.start()
            thread.join(10)
        assert seen == {"thread": first[1]}
        assert read() == base

    def test_thread_local_module(self):
        # The policy of a thread outside the block returns the mean while
        # another thread samples.
        policy = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=Normal,
            default_interaction_type="deterministic",
        )
        obs = TensorDict(loc=torch.zeros(3), scale=torch.ones(3))
        in_block, release = threading.Event(), threading.Event()

        def learner():
            with set_interaction_type("random", thread_local=True):
                in_block.set()
                release.wait(10)

        thread = threading.Thread(target=learner)
        thread.start()
        try:
            assert in_block.wait(10)
            assert (policy(obs.clone())["action"] == 0).all()
        finally:
            release.set()
            thread.join(10)

    def test_str_enum(self):
        # InteractionType derives from enum.StrEnum, or from the backport
        # tensordict.nn.utils._StrEnum before Python 3.11.
        assert InteractionType.MODE == "mode"
        assert InteractionType("MODE") is InteractionType.MODE
        assert pickle.loads(pickle.dumps(InteractionType.MODE)) is InteractionType.MODE

        class Backported(tensordict.nn.utils._StrEnum):
            MODE = "mode"

        assert Backported.MODE == "mode"
        assert isinstance(Backported.MODE, str)
        with pytest.warns(
            DeprecationWarning,
            match=r"^tensordict\.nn\.utils\.StrEnum is deprecated and will be "
            r"removed in TensorDict 0\.17\.$",
        ) as record:
            assert tensordict.nn.utils.StrEnum is tensordict.nn.utils._StrEnum
        assert record[0].filename == __file__

    def test_str_enum_str(self):
        # str() and format() give the value, as with enum.StrEnum, on every
        # Python version.
        class Backported(tensordict.nn.utils._StrEnum):
            MODE = "mode"

        for member in (InteractionType.MODE, Backported.MODE):
            assert str(member) == "mode"
            assert "%s" % member == "mode"
            assert f"{member}" == "mode"
            assert format(member, ">5") == " mode"

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

    @pytest.mark.parametrize("in_keys", [["obs"], {"x": "obs"}])
    def test_get_kwargs(self, in_keys):
        # get_kwargs reach get() whether in_keys is a list or a dict
        td = TensorDict.lazy_stack(
            [TensorDict(obs=torch.ones(2)), TensorDict(obs=torch.ones(3))]
        )
        module = TensorDictModule(
            lambda x: x,
            in_keys=in_keys,
            out_keys=["y"],
            get_kwargs={"as_padded_tensor": True, "padding_value": -1.0},
        )
        torch.testing.assert_close(
            module(td)["y"], torch.tensor([[1.0, 1.0, -1.0], [1.0, 1.0, 1.0]])
        )

    def test_repr(self):
        class MyModule(TensorDictModuleBase):
            in_keys = ["a"]
            out_keys = ["b"]

            def __init__(self):
                super().__init__()
                self.net = nn.Linear(3, 4)

            def forward(self, tensordict):
                tensordict.set("b", self.net(tensordict.get("a")))
                return tensordict

        string = repr(MyModule())
        assert "MyModule(" in string
        assert "net=Linear(in_features=3, out_features=4" in string
        assert "device=" not in string
        assert "in_keys=['a']" in string
        assert "out_keys=['b']" in string

        # an nn.Module payload is a child: listed once, and not through __dict__
        mod = TensorDictModule(nn.Linear(3, 4), in_keys=["a"], out_keys=["b"])
        assert "module" not in mod.__dict__
        string = repr(mod)
        assert string.count("module=") == 1
        assert "module=Linear(in_features=3, out_features=4" in string

        # a callable that is not an nn.Module is not a child: it comes from __dict__
        mod = TensorDictModule(lambda x: x + 1, in_keys=["a"], out_keys=["b"])
        assert dict(mod.named_children()) == {}
        string = repr(mod)
        assert string.count("module=") == 1
        assert "module=<function" in string

    def test_device_deprecation(self):
        mod = TensorDictModule(nn.Linear(3, 4), in_keys=["a"], out_keys=["b"])
        with pytest.warns(DeprecationWarning, match=r"removed in TensorDict 0\.17"):
            assert mod.device == torch.device("cpu")

    def test_is_functional_deprecation(self):
        mod = TensorDictModule(nn.Linear(3, 4), in_keys=["a"], out_keys=["b"])
        with pytest.warns(
            DeprecationWarning,
            match=r"^TensorDictModule\.is_functional is deprecated and will be "
            r"removed in TensorDict 0\.17\.$",
        ) as record:
            assert mod.is_functional is False
        assert record[0].filename == __file__

        with pytest.warns(FutureWarning, match="integrated functorch"):
            fmodule, _, _ = make_functional_functorch(nn.Linear(3, 4))
        mod = TensorDictModule(fmodule, in_keys=["a"], out_keys=["b"])
        with pytest.warns(DeprecationWarning, match="is_functional"):
            assert mod.is_functional is True
        seq = TensorDictSequential(mod)
        with pytest.warns(DeprecationWarning, match="is_functional"):
            assert seq.is_functional is False

    @pytest.mark.parametrize(
        "name", ["FunctionalModule", "FunctionalModuleWithBuffers"]
    )
    def test_functional_module_classes_deprecation(self, name):
        with pytest.warns(
            DeprecationWarning,
            match=rf"^tensordict\.nn\.common\.{name} is deprecated and will be "
            rf"removed in TensorDict 0\.17\. Use functorch\.{name} instead\.$",
        ) as record:
            cls = getattr(tensordict.nn.common, name)
        assert record[0].filename == __file__
        assert cls is getattr(functorch, name)

    def test_dispatch_get_source_deprecation(self):
        class Module(nn.Module):
            in_keys = ["a"]

        with pytest.warns(
            DeprecationWarning,
            match=r"^dispatch\.get_source\(\) is deprecated and will be removed "
            r"in TensorDict 0\.17\.$",
        ) as record:
            assert dispatch().get_source(None, Module()) == ["a"]
        assert record[0].filename == __file__
        with pytest.warns(DeprecationWarning, match="get_source"):
            assert dispatch(source=["b"]).get_source(None, Module()) == ["b"]

    def test_mutable_sequence(self):
        in_keys = self.MyMutableSequence(["a", "b", "c"])
        out_keys = self.MyMutableSequence(["d", "e", "f"])
        mod = TensorDictModule(lambda *x: x, in_keys=in_keys, out_keys=out_keys)
        td = mod(TensorDict(a=0, b=0, c=0))
        assert "d" in td
        assert "e" in td
        assert "f" in td

    @pytest.mark.parametrize("subclass", [False, True])
    def test_wrapper_runs_hooks_once(self, subclass):
        # gh-2942: the wrapper ran the wrapped module's forward hooks a second
        # time, and the copied selection hook raised a TypeError
        class PlusOne(TensorDictModuleWrapper):
            def forward(self, tensordict):
                tensordict = self.td_module(tensordict)
                tensordict["y"] = tensordict["y"] + 1
                return tensordict

        inner = TensorDictModule(lambda x: (x + 1, x * 2), ["x"], ["y", "z"])
        inner.register_forward_hook(
            lambda module, args, out: out.set("y", out["y"] * 10)
        )
        wrapper_cls = PlusOne if subclass else TensorDictModuleWrapper
        offset = 1 if subclass else 0
        td = wrapper_cls(inner)(TensorDict(x=torch.tensor(1.0)))
        assert td["y"] == 20 + offset
        inner.select_out_keys("y")
        td = wrapper_cls(inner)(TensorDict(x=torch.tensor(1.0)))
        assert td["y"] == 20 + offset
        assert "z" not in td

    @pytest.mark.parametrize("wrap", [False, True])
    def test_out_keys_setter(self, wrap):
        net = nn.Linear(3, 4)
        td_module = TensorDictModule(net, in_keys=["in"], out_keys=["out"])
        if wrap:
            td_module = TensorDictModuleWrapper(td_module)

        def split_output(module, args, output):
            return output, output.mean(dim=1)

        net.register_forward_hook(split_output)
        td_module.out_keys = ["out1", "out2"]

        x = torch.randn(3, 3)
        td_out = td_module(TensorDict({"in": x}, [3]))
        assert td_module.out_keys == td_module.out_keys_source == ["out1", "out2"]
        assert "out" not in td_out
        torch.testing.assert_close(td_out["out2"], td_out["out1"].mean(dim=1))

        out1, out2 = td_module(x)
        torch.testing.assert_close(out2, out1.mean(dim=1))

        td_module.select_out_keys("out1")
        td_module.out_keys = ["out1", "out2"]
        assert "out2" in td_module(TensorDict({"in": x}, [3]))

    def test_select_out_keys_property(self):
        class PropertyModule(TensorDictModuleBase):
            @property
            def out_keys(self):
                return self.keys

            @out_keys.setter
            def out_keys(self, value):
                self.keys = value

        module = PropertyModule()
        module.out_keys = ["a", "b"]
        module.select_out_keys("a").select_out_keys("a").reset_out_keys()
        assert module.out_keys == ["a", "b"]
        module.select_out_keys("a")
        module.out_keys = ["a", "c"]
        module.reset_out_keys()
        assert module.out_keys == ["a", "c"]

        module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["sample"],
            distribution_class=Normal,
            return_log_prob=True,
        )

        with pytest.raises(RuntimeError, match="does not support select_out_keys"):
            module.select_out_keys("sample")
        td = module(TensorDict({"loc": torch.zeros(()), "scale": torch.ones(())}, []))
        assert set(module.out_keys).issubset(td.keys())

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
            assert p.all(), (
                f"Discrepancy between returned weights and those in-place updated {p}"
            )

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
        with (
            _set_dispatch_td_nn_modules(False),
            pytest.raises(TypeError, match="missing 1 required positional argument"),
        ):
            tdm(a=torch.zeros(1, 1))

        # checks that things are back in place
        tdm = TensorDictModule(nn.Linear(1, 1), ["a"], ["b"])
        tdm(a=torch.zeros(1, 1))

    def test_dispatch_deactivate_restores_mode(self):
        # Leaving the block restores the previous mode, not the global
        # _ContextManager object (#1948).
        mode = _dispatch_td_nn_modules()
        with _set_dispatch_td_nn_modules(not mode):
            assert _dispatch_td_nn_modules() == (not mode)
        assert _dispatch_td_nn_modules() == mode

        tdm = TensorDictModule(nn.Linear(1, 1), ["a"], ["b"])
        with _set_dispatch_td_nn_modules(True):
            with _set_dispatch_td_nn_modules(False):
                with _set_dispatch_td_nn_modules(True):
                    assert _dispatch_td_nn_modules()
                assert not _dispatch_td_nn_modules()
                with pytest.raises(
                    TypeError, match="missing 1 required positional argument"
                ):
                    tdm(a=torch.zeros(1, 1))

            # A reused instance that changes nothing on entry restores nothing.
            set_dispatch_off = _set_dispatch_td_nn_modules(False)
            with set_dispatch_off:
                assert not _dispatch_td_nn_modules()
            with _set_dispatch_td_nn_modules(False):
                with set_dispatch_off:
                    pass
                assert not _dispatch_td_nn_modules()

            # An instance nested in itself restores the mode of the outer block.
            with set_dispatch_off:
                with set_dispatch_off:
                    assert not _dispatch_td_nn_modules()
                assert not _dispatch_td_nn_modules()
            assert _dispatch_td_nn_modules()

            # The mode is restored when the block raises.
            with pytest.raises(RuntimeError, match="inside"):
                with set_dispatch_off:
                    raise RuntimeError("inside")
            assert _dispatch_td_nn_modules()
        assert _dispatch_td_nn_modules() == mode

    def test_dispatch_deactivate_function(self):
        @dispatch(source=["a"], dest=["b"])
        def func(tensordict):
            tensordict["b"] = tensordict["a"] + 1
            return tensordict

        assert (func(torch.zeros(1, 2)) == 1).all()
        td = TensorDict(a=torch.zeros(1, 2))
        with _set_dispatch_td_nn_modules(False):
            assert func(td) is td
        assert (td["b"] == 1).all()

    @pytest.mark.parametrize("dispatch_mode", [True, False])
    @pytest.mark.parametrize(
        "make_input",
        [
            lambda: TensorDict(a=torch.zeros(2, 1), batch_size=[2]),
            lambda: TensorDict(a=torch.zeros(2, 1), batch_size=[2], names=["n"]),
            lambda: lazy_stack([TensorDict(a=torch.zeros(1)) for _ in range(2)]),
            lambda: TensorDictParams(TensorDict(a=torch.zeros(2, 1))),
            lambda: TensorDict(a=torch.zeros(2, 1), batch_size=[2]).lock_(),
        ],
        ids=["td", "names", "lazy", "params", "locked"],
    )
    def test_dispatch_tensordict_input(self, dispatch_mode, make_input):
        # A tensordict is passed to forward as is, with dispatch on or off
        tdm = TensorDictModule(nn.Linear(1, 1), ["a"], ["b"])
        td = make_input()
        with _set_dispatch_td_nn_modules(dispatch_mode):
            if td.is_locked:
                with pytest.raises(RuntimeError, match="locked"):
                    tdm(td)
                assert "b" not in td.keys()
                return
            out = tdm(td)
        assert out is td
        assert type(td) is type(make_input())
        assert td["b"].shape == (2, 1)
        assert td.names == make_input().names

    @pytest.mark.parametrize("dispatch_mode", [True, False])
    @pytest.mark.parametrize("bound", [False, True])
    def test_dispatch_tensordict_reads_source(self, dispatch_mode, bound):
        # With a tensordict input, the source is read only when dispatch is
        # on: a missing one raises, and a property is resolved once
        class LazyKeys(nn.Module):
            reads = 0

            @property
            def in_keys(self):
                type(self).reads += 1
                return ["a"]

            out_keys = ["b"]

            def inner(self, tensordict):
                return tensordict.set("b", tensordict["a"] + 1)

            forward = dispatch(inner)

        class NoKeys(nn.Module):
            def inner(self, tensordict):
                return tensordict

            forward = dispatch(inner)

        lazy, nokeys = LazyKeys(), NoKeys()
        if bound:
            lazy_call = dispatch(lazy.inner)
            nokeys_call = dispatch(nokeys.inner)
        else:
            lazy_call, nokeys_call = lazy, nokeys
        td = TensorDict(a=torch.zeros(1))
        with _set_dispatch_td_nn_modules(dispatch_mode):
            assert lazy_call(td) is td
            assert LazyKeys.reads == int(dispatch_mode)
            if dispatch_mode:
                with pytest.raises(AttributeError, match="in_keys"):
                    nokeys_call(td)
            else:
                assert nokeys_call(td) is td
        assert (td["b"] == 1).all()

    def test_dispatch_tensordict_unbound_none_self(self):
        # An unbound call with self=None and a tensordict raises the error
        # of a function without in keys, as with any other input
        td = TensorDict(x=torch.randn(2, 3), batch_size=[2])
        with pytest.raises(RuntimeError, match="The in keys must be passed"):
            TensorDictModule.forward(None, td)

    @pytest.mark.parametrize("dispatch_mode", [True, False])
    def test_dispatch_tensordict_keyword(self, dispatch_mode):
        # ``module(tensordict=td)`` behaves like ``module(td)`` (#3199)
        linear = nn.Linear(6, 1)
        tdm = TensorDictModule(linear, in_keys=["x"], out_keys=["q"])
        seq = TensorDictSequential(tdm)
        prob = ProbabilisticTensorDictModule(
            in_keys=["param"], out_keys=["s"], distribution_class=Delta
        )
        prob_seq = ProbabilisticTensorDictSequential(
            TensorDictModule(linear, in_keys=["x"], out_keys=["param"]), prob
        )
        wrapper = TensorDictModuleWrapper(tdm)
        with _set_dispatch_td_nn_modules(dispatch_mode):
            for module, out_key in [
                (tdm, "q"),
                (seq, "q"),
                (prob_seq, "s"),
                (wrapper, "q"),
            ]:
                td = TensorDict(x=torch.randn(5, 6), batch_size=[5])
                assert module(tensordict=td) is td
                assert td[out_key].shape == (5, 1)
            td = TensorDict(param=torch.randn(5, 1), batch_size=[5])
            assert prob(tensordict=td) is td
            assert (td["s"] == td["param"]).all()

            td = TensorDict(x=torch.randn(5, 6), batch_size=[5])
            td_out = TensorDict(batch_size=[5])
            assert tdm(tensordict=td, tensordict_out=td_out) is td_out
            assert "q" in td_out.keys() and "q" not in td.keys()

            td = TensorDict(x=torch.randn(5, 6), q=torch.zeros(5, 1), batch_size=[5])
            with set_skip_existing(True):
                assert tdm(tensordict=td) is td
            assert (td["q"] == 0).all()

            params = dict(tdm.named_parameters())
            td = TensorDict(x=torch.randn(5, 6), batch_size=[5])
            out = torch.func.functional_call(tdm, params, (), {"tensordict": td})
            assert out is td
            torch.testing.assert_close(td["q"], linear(td["x"]))

            # A decorated function, as in CudaGraphModule
            @dispatch(source=["x"], dest=["y"])
            def func(tensordict):
                return tensordict.set("y", tensordict["x"] + 1)

            assert func(tensordict=td) is td
            assert (td["y"] == td["x"] + 1).all()

    def test_dispatch_tensordict_keyword_input_key(self):
        # An input key named "tensordict" keeps taking the keyword argument
        tdm = TensorDictModule(lambda x: x + 1, in_keys=["tensordict"], out_keys=["y"])
        assert (tdm(tensordict=torch.zeros(2)) == 1).all()
        tdm = TensorDictModule(
            lambda x: x.apply(lambda t: t + 1), in_keys=["tensordict"], out_keys=["y"]
        )
        out = tdm(tensordict=TensorDict(a=torch.zeros(2)))
        assert (out["a"] == 1).all()

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

    @pytest.mark.parametrize(
        "source,dest", [("in_keys", "out_keys"), (["a", "b"], ["c"])]
    )
    def test_dispatch_bound_method(self, source, dest):
        # https://github.com/pytorch/tensordict/issues/1459
        class MyModule(nn.Module):
            in_keys = ["a", "b"]
            out_keys = ["c"]

            def inner_forward(self, tensordict):
                tensordict["c"] = tensordict["a"] - tensordict["b"]
                return tensordict

            def forward(self, *args, **kwargs):
                return dispatch(source=source, dest=dest)(self.inner_forward)(
                    *args, **kwargs
                )

            def filler_forward(self, filler, tensordict):
                return self.inner_forward(tensordict)

        module = MyModule()
        a, b = torch.ones(1, 2), torch.zeros(1, 2)
        assert (module(a, b) == 1).all()
        assert (module(a=a, b=b) == 1).all()
        td = TensorDict(a=a, b=b)
        assert module(td) is td
        assert (td["c"] == 1).all()
        dispatched = dispatch(source=source, dest=dest)(module.inner_forward)
        td = TensorDict(a=a, b=b)
        with _set_dispatch_td_nn_modules(False):
            assert dispatched(td) is td
        assert (td["c"] == 1).all()
        # self is bound already: the first argument must be the tensordict
        with pytest.raises(RuntimeError, match="Got filler instead"):
            dispatch(source=source, dest=dest)(module.filler_forward)

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

    def test_dispatch_ignored_out_key(self):
        module = TensorDictModule(
            lambda x: (x + 1, x - 1, x + 2), in_keys=["x"], out_keys=["y", "_", "z"]
        )
        y, z = module(x=torch.zeros(2))
        assert (y == 1).all()
        assert (z == 2).all()
        seq = TensorDictSequential(
            TensorDictModule(
                lambda x: (x + 1, x - 1), in_keys=["x"], out_keys=["y", "_"]
            ),
            TensorDictModule(lambda y: y * 2, in_keys=["y"], out_keys=["w"]),
        )
        y, w = seq(x=torch.zeros(2))
        assert (y == 1).all()
        assert (w == 2).all()

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

    @pytest.mark.parametrize("output_type", [dict, TensorDict])
    @pytest.mark.parametrize("nested", [False, True])
    @pytest.mark.parametrize("selected", [["b"], ["c", "b"], []])
    @pytest.mark.parametrize("call", ["tensordict", "positional", "keyword", "out"])
    def test_tdmodule_dict_output_selected(self, output_type, nested, selected, call):
        out_keys = [("result", key) if nested else key for key in ("b", "c")]
        selected = [("result", key) if nested else key for key in selected]

        def func(x):
            return output_type({out_keys[0]: x + 1, out_keys[1]: x * 10})

        module = TensorDictModule(func, in_keys=["x"], out_keys=out_keys)
        module.select_out_keys(*selected)
        assert module.out_keys_source == out_keys
        x = torch.arange(3.0)
        expected = {out_keys[0]: x + 1, out_keys[1]: x * 10}
        if call in ("tensordict", "out"):
            td = TensorDict(x=x, batch_size=[3])
            if call == "out":
                out = TensorDict(batch_size=[3])
                result = module(td, tensordict_out=out)
                assert result is out
                expected_keys = set(selected)
                assert set(td.keys()) == {"x"}
            else:
                result = module(td)
                assert result is td
                expected_keys = {"x", *selected}
            assert set(result.keys(True, True)) == expected_keys
            for key in selected:
                torch.testing.assert_close(result[key], expected[key])
        else:
            result = module(x) if call == "positional" else module(x=x)
            expected_values = tuple(expected[key] for key in selected)
            if len(expected_values) == 1:
                expected_values = expected_values[0]
            torch.testing.assert_close(result, expected_values)

        module.reset_out_keys()
        result = module(TensorDict(x=x, batch_size=[3]))
        assert module.out_keys == out_keys
        for key in out_keys:
            torch.testing.assert_close(result[key], expected[key])

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

    @pytest.mark.parametrize("inplace", [False, "empty"])
    def test_tdseq_inplace_false_keeps_input(self, inplace):
        # https://github.com/pytorch/tensordict/issues/2257
        model = TensorDictSequential(
            TensorDictModule(
                lambda x: x + 1,
                in_keys=["x"],
                out_keys=[("nested", "y")],
            ),
            TensorDictModule(
                lambda y: y * 2,
                in_keys=[("nested", "y")],
                out_keys=["z"],
            ),
            inplace=inplace,
        )
        input = TensorDict(
            x=torch.zeros(2),
            nested=TensorDict(a=torch.zeros(2), batch_size=[2]),
            batch_size=[2],
        )
        output = model(input)
        assert output is not input
        assert set(input.keys(True, True)) == {"x", ("nested", "a")}
        assert (output["nested", "y"] == 1).all()
        assert (output["z"] == 2).all()

    @pytest.mark.parametrize("stack_dim", [0, 1])
    @pytest.mark.parametrize("module_inplace", [True, False, "empty"])
    def test_partial_tolerant_lazy_stack_keeps_outputs(self, module_inplace, stack_dim):
        # https://github.com/pytorch/tensordict/issues/2757
        stack = lazy_stack(
            [
                TensorDict(a=torch.zeros(3, 2), batch_size=[3, 2]),
                TensorDict(c=torch.zeros(3, 2), batch_size=[3, 2]),
            ],
            stack_dim,
        )
        module = TensorDictModule(
            lambda a: a + 1,
            in_keys=["a"],
            out_keys=[("nested", "b")],
            inplace=module_inplace,
        )
        out = TensorDictSequential(module, partial_tolerant=True)(stack)
        assert out is stack
        assert (out.tensordicts[0]["nested", "b"] == 1).all()
        assert set(out.tensordicts[1].keys()) == {"c"}

    def test_partial_tolerant_lazy_stack_copies_out_keys_only(self):
        # https://github.com/pytorch/tensordict/issues/2757
        class CopyModule(TensorDictModuleBase):
            in_keys = ["a"]
            out_keys = ["b", "missing"]

            def forward(self, td):
                out = td.copy()
                out["a"] = td["a"] + 5
                out["b"] = td["a"] + 1
                out["extra"] = td["a"]
                return out

        def make_stack():
            return lazy_stack(
                [
                    TensorDict(a=torch.zeros(3), batch_size=[3]),
                    TensorDict(c=torch.zeros(3), batch_size=[3]),
                ]
            )

        out = TensorDictSequential(CopyModule(), partial_tolerant=True)(make_stack())
        member = out.tensordicts[0]
        assert set(member.keys()) == {"a", "b"}
        assert (member["a"] == 0).all()
        assert (member["b"] == 1).all()

        # A locked stack raises as with a module that writes in place, unless
        # the module returns none of its out_keys.
        in_place = TensorDictModule(lambda a: a + 1, in_keys=["a"], out_keys=["b"])
        for module in (in_place, CopyModule()):
            stack = make_stack().lock_()
            with pytest.raises(RuntimeError, match="Cannot modify locked"):
                TensorDictSequential(module, partial_tolerant=True)(stack)
        no_outputs = CopyModule()
        no_outputs.out_keys = ["missing"]
        stack = make_stack().lock_()
        out = TensorDictSequential(no_outputs, partial_tolerant=True)(stack)
        assert set(out.tensordicts[0].keys()) == {"a"}

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

    @pytest.mark.parametrize("mapping", [dict, OrderedDict, None])
    @pytest.mark.parametrize("operation", ["replace", "delete"])
    def test_sequence_mutation(self, mapping, operation):
        first = TensorDictModule(lambda x: x + 1, ["x"], ["x"])
        second = TensorDictModule(lambda x: x * 2, ["x"], ["x"])
        seq = (
            TensorDictSequential(mapping(first=first, second=second))
            if mapping is not None
            else TensorDictSequential(first, second)
        )
        index = "first" if mapping is not None else 0
        if operation == "replace":
            replacement = TensorDictModule(lambda x: x + 3, ["x"], ["x"])
            seq[index] = replacement
            assert seq[index] is replacement
            assert len(seq) == 2
            expected = 8.0
        else:
            del seq[index]
            assert len(seq) == 1
            assert seq["second" if mapping is not None else 0] is second
            expected = 2.0
        result = seq(TensorDict(x=torch.tensor(1.0)))
        torch.testing.assert_close(result["x"], torch.tensor(expected))

    @pytest.mark.parametrize("mapping", [dict, None])
    def test_sequence_mutation_updates_keys(self, mapping):
        a = TensorDictModule(lambda x: x + 1, ["x"], ["y"])
        b = TensorDictModule(lambda y: y + 1, ["y"], ["z"])
        if mapping is None:
            seq = TensorDictSequential(a, b, inplace=False)
            first, second = 0, 1
        else:
            seq = TensorDictSequential(mapping(a=a, b=b), inplace=False)
            first, second = "a", "b"
        seq[second] = TensorDictModule(lambda y: y * 10, ["y"], ["u"])
        assert seq.in_keys == ["x"]
        assert seq.out_keys == ["y", "u"]
        result = seq(TensorDict(x=torch.tensor(1.0)))
        assert set(result.keys()) == {"y", "u"}
        torch.testing.assert_close(result["u"], torch.tensor(20.0))
        del seq[first]
        assert seq.in_keys == ["y"]
        assert seq.out_keys == ["u"]
        # a plain callable is wrapped, as in the constructor
        seq = TensorDictSequential(a, b)
        seq[1] = lambda td: td.set("w", td["y"] + 1)
        torch.testing.assert_close(
            seq(TensorDict(x=torch.tensor(1.0)))["w"], torch.tensor(3.0)
        )

    def test_sequence_mutation_keeps_selected_out_keys(self):
        a = TensorDictModule(lambda x: x + 1, ["x"], ["y"])
        b = TensorDictModule(lambda y: y + 1, ["y"], ["z"])
        seq = TensorDictSequential(a, b, selected_out_keys=["z"])
        with pytest.raises(ValueError, match="selected out_keys"):
            seq[1] = TensorDictModule(lambda y: y * 10, ["y"], ["u"])
        assert seq[1] is b
        assert seq.out_keys == ["z"]
        seq[0] = TensorDictModule(lambda x: x + 2, ["x"], ["y"])
        assert seq.out_keys == ["z"]
        result = seq(TensorDict(x=torch.tensor(1.0)))
        assert "y" not in result
        torch.testing.assert_close(result["z"], torch.tensor(4.0))
        seq.reset_out_keys()
        assert seq.out_keys == ["y", "z"]

    @pytest.mark.parametrize("named", [True, False])
    def test_prob_sequence_mutation_rebuilds_det_part(self, named):
        def params(loc_shift):
            return TensorDictModule(
                lambda x: (x + loc_shift, torch.ones_like(x)),
                ["x"],
                ["loc", "scale"],
            )

        prob = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=torch.distributions.Normal,
        )
        if named:
            seq = ProbabilisticTensorDictSequential({"params": params(0), "prob": prob})
            index = "params"
        else:
            seq = ProbabilisticTensorDictSequential(params(0), prob)
            index = 0
        seq[index] = params(100)
        td = TensorDict(x=torch.ones(3), batch_size=[3])
        with set_interaction_type(InteractionType.DETERMINISTIC):
            torch.testing.assert_close(
                seq(td.clone())["action"], torch.full((3,), 101.0)
            )
        torch.testing.assert_close(
            seq.get_dist(td.clone()).mean, torch.full((3,), 101.0)
        )
        # the last module of a ProbabilisticTensorDictSequential stays probabilistic
        last = "prob" if named else 1
        with pytest.raises(TypeError, match="final module"):
            seq[last] = params(0)
        assert seq[last] is prob

    @pytest.mark.parametrize("mapping", [dict, OrderedDict])
    @pytest.mark.parametrize("key", ["x", ("nested", "x")])
    def test_named_sequence_shared_module(self, mapping, key):
        linear = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            linear.weight.fill_(2)
        shared = TensorDictModule(linear, in_keys=[key], out_keys=[key])
        seq = TensorDictSequential(
            mapping(
                first=shared,
                middle=TensorDictModule(lambda x: x + 1, [key], [key]),
                last=shared,
            )
        )
        x = torch.ones(1, requires_grad=True)
        result = seq(TensorDict({key: x}))[key]
        torch.testing.assert_close(result, torch.tensor([6.0]))
        result.sum().backward()
        torch.testing.assert_close(linear.weight.grad, torch.tensor([[5.0]]))
        torch.testing.assert_close(x.grad, torch.tensor([4.0]))

        selected = seq.select_subsequence(out_keys=[key])
        assert selected["first"] is selected["last"] is shared
        torch.testing.assert_close(
            selected(TensorDict({key: torch.ones(1)}))[key], torch.tensor([6.0])
        )

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

    @pytest.mark.parametrize("mapping", [dict, OrderedDict])
    @pytest.mark.parametrize("in_keys", [None, ["y"]])
    def test_named_nested_select_subsequence(self, mapping, in_keys):
        first = TensorDictModule(lambda x: x + 1, ["x"], ["y"])
        second = TensorDictModule(lambda y: y + 2, ["y"], ["z"])
        unused = TensorDictModule(lambda y: y + 3, ["y"], ["unused"])
        nested = TensorDictSequential(first, second, unused)
        last = TensorDictModule(lambda z: z * 2, ["z"], ["out"])
        seq = TensorDictSequential(
            mapping(
                discarded=TensorDictModule(lambda x: x - 1, ["x"], ["discarded"]),
                nested=nested,
                last=last,
            )
        )

        selected = seq.select_subsequence(in_keys=in_keys, out_keys=["out"])
        assert list(selected.module) == ["nested", "last"]
        assert selected["last"] is last
        assert selected["nested"][-1] is second
        assert len(selected["nested"]) == (2 if in_keys is None else 1)
        assert len(nested) == 3
        assert len(seq) == 3
        td = TensorDict(x=torch.tensor(0.0), y=torch.tensor(1.0))
        torch.testing.assert_close(selected(td)["out"], torch.tensor(6.0))

    @pytest.mark.parametrize("named", [False, True])
    def test_select_subsequence_keeps_options(self, named):
        first = TensorDictModule(lambda a: a + 1, ["a"], ["b"])
        second = TensorDictModule(lambda x: x + 1, ["x"], ["y"])
        modules = {"first": first, "second": second} if named else [first, second]
        seq = TensorDictSequential(modules, partial_tolerant=True, inplace=False)
        subsequences = [seq.select_subsequence(out_keys=["b", "y"])]
        # A nested sequence keeps its own options.
        outer = TensorDictSequential(seq, TensorDictModule(lambda b: b, ["b"], ["c"]))
        subsequences.append(outer.select_subsequence(out_keys=["c"])[0])
        if not named:
            subsequences.append(seq[:])
        for sub in subsequences:
            assert sub.partial_tolerant is True
            assert sub.inplace is False
            td = TensorDict(a=torch.zeros(2), batch_size=[2])  # no "x"
            out = sub(td)
            assert out is not td
            assert set(out.keys()) == {"b"}
            assert set(td.keys()) == {"a"}

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
        assert seq.out_keys_source == ["foo1", "key1", "key2"]

    def test_out_keys_setter(self):
        module = TensorDictModule(
            lambda x: (x, x + 1), in_keys=["in"], out_keys=["out1", "out2"]
        )
        seq = TensorDictSequential(module)
        module.out_keys = ["new_out1", "new_out2"]
        seq.out_keys = module.out_keys

        seq.select_out_keys("new_out2").reset_out_keys()

        assert seq.out_keys == seq.out_keys_source == ["new_out1", "new_out2"]
        out1, out2 = seq(torch.zeros(()))
        torch.testing.assert_close(out2, out1 + 1)

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

    def test_prob_key_exclusion_constructor(self):
        module1 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        module2 = TensorDictModule(
            nn.Linear(3, 4), in_keys=["key1", "key3"], out_keys=["key1"]
        )
        prob_module = ProbabilisticTensorDictModule(
            in_keys=["foo1", "key3"],
            out_keys=["key2"],
            distribution_class=Normal,
        )
        seq = TensorDictSequential(
            module1, module2, prob_module, selected_out_keys=["key2"]
        )
        assert set(seq.in_keys) == set(unravel_key_list(("key1", "key2", "key3")))
        assert seq.out_keys == ["key2"]

    def test_prob_key_exclusion_constructor_exec(self):
        module1 = TensorDictModule(
            lambda x, y: x + y, in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        module2 = TensorDictModule(
            lambda x, y: x + y, in_keys=["key1", "key3"], out_keys=["key1"]
        )
        prob_module = ProbabilisticTensorDictModule(
            in_keys={"loc": "foo1", "scale": "key3"},
            out_keys=["key2"],
            distribution_class=Normal,
        )
        seq = ProbabilisticTensorDictSequential(
            module1, module2, prob_module, selected_out_keys=["key2"]
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

    def test_prob_seq_det_part_deprecation(self):
        seq = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["loc"]),
            ProbabilisticTensorDictModule(
                in_keys=["loc"], out_keys=["sample"], distribution_class=Delta
            ),
        )
        with pytest.warns(
            DeprecationWarning,
            match=r"^ProbabilisticTensorDictSequential\.det_part is deprecated and "
            r"will be removed in TensorDict 0\.17\. Use "
            r"ProbabilisticTensorDictSequential\.get_dist_params\(\) instead\.$",
        ) as record:
            det_part = seq.det_part
        assert record[0].filename == __file__
        assert det_part is seq._det_part
        assert isinstance(det_part, TensorDictSequential)
        params = seq.get_dist_params(TensorDict(x=torch.zeros(3)))
        assert (params["loc"] == 1).all()

    def test_prob_seq_get_dist_params_return_hint(self):
        # get_dist_params returns the tensordict, not a (distribution, td) tuple
        hints = typing.get_type_hints(ProbabilisticTensorDictSequential.get_dist_params)
        assert hints["return"] is TensorDictBase

    def test_prob_seq_no_prob_modules(self):
        """Test ProbabilisticTensorDictSequential with no probabilistic modules.

        When no probabilistic modules are passed, it should initialize as a regular
        TensorDictSequential and still work correctly, including with selected_out_keys.
        """
        module1 = TensorDictModule(
            lambda x, y: x + y, in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        module2 = TensorDictModule(
            lambda x, y: x + y, in_keys=["foo1", "key3"], out_keys=["result"]
        )
        # No probabilistic modules - should still work
        seq = ProbabilisticTensorDictSequential(module1, module2)
        assert set(seq.in_keys) == {"key1", "key2", "key3"}
        assert set(seq.out_keys) == {"foo1", "result"}

        # Verify internal attributes are set correctly for no-prob-modules case
        assert seq._requires_sample is False
        assert seq._det_part is None
        assert seq.return_composite is True

        td = TensorDict(key1=1, key2=2, key3=3)
        out = seq(td)
        assert out is td
        assert out["foo1"] == 3  # 1 + 2
        assert out["result"] == 6  # 3 + 3

    def test_prob_seq_no_prob_modules_selected_out_keys(self):
        """Test ProbabilisticTensorDictSequential with no prob modules and selected_out_keys."""
        module1 = TensorDictModule(
            lambda x, y: x + y, in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        module2 = TensorDictModule(
            lambda x, y: x + y, in_keys=["foo1", "key3"], out_keys=["result"]
        )
        # No probabilistic modules, but with selected_out_keys
        seq = ProbabilisticTensorDictSequential(
            module1, module2, selected_out_keys=["result"]
        )
        assert set(seq.in_keys) == {"key1", "key2", "key3"}
        assert seq.out_keys == ["result"]

        td = TensorDict(key1=1, key2=2, key3=3)
        out = seq(td)
        assert out is td
        assert "foo1" not in out  # Should be excluded
        assert out["result"] == 6  # 3 + 3

    def test_prob_seq_subclass_without_selected_out_keys_arg(self):
        """Non-regression test: subclasses with MRO that doesn't support selected_out_keys.

        This mimics scenarios like TorchRL's SafeProbabilisticTensorDictSequential where
        an intermediate class in the MRO doesn't accept selected_out_keys.
        """

        class IntermediateSequential(TensorDictSequential):
            """A sequential that doesn't accept selected_out_keys in __init__."""

            def __init__(self, *modules, partial_tolerant=False, inplace=None):
                super().__init__(
                    *modules, partial_tolerant=partial_tolerant, inplace=inplace
                )

        class CustomProbabilisticSequential(
            ProbabilisticTensorDictSequential, IntermediateSequential
        ):
            """Subclass with IntermediateSequential in MRO."""

            def __init__(self, *modules, partial_tolerant=False):
                # This mirrors TorchRL's SafeProbabilisticTensorDictSequential pattern
                super().__init__(*modules, partial_tolerant=partial_tolerant)

        # Test with probabilistic modules - should not raise
        module1 = TensorDictModule(
            lambda x, y: x + y, in_keys=["key1", "key2"], out_keys=["foo1"]
        )
        prob_module = ProbabilisticTensorDictModule(
            in_keys={"loc": "foo1", "scale": "key3"},
            out_keys=["sample"],
            distribution_class=Normal,
        )
        seq = CustomProbabilisticSequential(module1, prob_module)
        td = TensorDict(key1=0, key2=0, key3=1)
        out = seq(td)
        assert "sample" in out

        # Test without probabilistic modules - should also not raise
        module2 = TensorDictModule(
            lambda x, y: x + y, in_keys=["foo1", "key3"], out_keys=["result"]
        )
        seq2 = CustomProbabilisticSequential(module1, module2)
        td2 = TensorDict(key1=1, key2=2, key3=3)
        out2 = seq2(td2)
        assert out2["result"] == 6

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

    @pytest.mark.parametrize("aggregate_probabilities", [None, False, True])
    @pytest.mark.parametrize("inplace", [None, False, True])
    @pytest.mark.parametrize("include_sum", [None, False, True])
    def test_probtdseq_multdist(self, include_sum, aggregate_probabilities, inplace):
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

    @pytest.mark.parametrize("nested_last", [True, False])
    @set_composite_lp_aggregate(False)
    def test_probtdseq_nested_seq_dist(self, nested_last):
        # A nested ProbabilisticTensorDictSequential adds its distribution to the
        # CompositeDistribution under its sample key, here "sample0".
        inner = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["loc0"]),
            ProbabilisticTensorDictModule(
                in_keys={"loc": "loc0"},
                out_keys=["sample0"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            ),
        )
        prob1 = ProbabilisticTensorDictModule(
            in_keys={"loc": "loc1"},
            out_keys=["sample1"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
        )
        if nested_last:
            loc1 = TensorDictModule(lambda x: x - 1, in_keys=["x"], out_keys=["loc1"])
            modules = [loc1, prob1, inner]
        else:
            loc1 = TensorDictModule(
                lambda s: s + 1, in_keys=["sample0"], out_keys=["loc1"]
            )
            modules = [inner, loc1, prob1]
        seq = ProbabilisticTensorDictSequential(*modules, return_composite=True)
        td = TensorDict(x=torch.randn(3), batch_size=[3])

        dist = seq.get_dist(td)
        assert set(dist) == {"sample0", "sample1"}
        if not nested_last:
            # loc1 is computed from the deterministic sample0, the mean x + 1
            torch.testing.assert_close(dist["sample1"].loc, td["x"] + 2)

        with set_interaction_type("random"):
            out = seq(td.copy())
        lp = seq.log_prob(out)
        assert set(lp.keys()) == {"sample0_log_prob", "sample1_log_prob"}
        torch.testing.assert_close(
            lp["sample0_log_prob"], Normal(out["loc0"], 1.0).log_prob(out["sample0"])
        )

    @pytest.mark.parametrize("aggregate", [True, False])
    @pytest.mark.parametrize("inner_return_composite", [None, True, False])
    def test_probtdseq_log_prob_custom_keys(self, aggregate, inner_return_composite):
        # log_prob() returns the keys a forward call writes, with the values
        # it writes there.
        with set_composite_lp_aggregate(aggregate):
            prob_a = ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                out_keys=["a"],
                distribution_class=Normal,
                return_log_prob=True,
                log_prob_key="lp_a",
            )
            prob_b = ProbabilisticTensorDictModule(
                in_keys={"loc": "a", "scale": "scale_b"},
                out_keys=["b"],
                distribution_class=Normal,
                return_log_prob=True,
                log_prob_key="lp_b",
            )
            if inner_return_composite is None:
                probs = [prob_a, prob_b]
            else:
                probs = [
                    ProbabilisticTensorDictSequential(
                        prob_a, prob_b, return_composite=inner_return_composite
                    )
                ]
            seq = ProbabilisticTensorDictSequential(
                TensorDictModule(
                    lambda x: (x, x.exp(), 2 * x.exp()),
                    in_keys=["x"],
                    out_keys=["loc", "scale", "scale_b"],
                ),
                *probs,
                return_composite=True,
            )
            assert set(seq.log_prob_keys) == {"lp_a", "lp_b"}
            td = seq(TensorDict(x=torch.zeros(3), batch_size=[3]))
            lp = seq.log_prob(td)
            # A nested sequence with return_composite=False adds only the
            # distribution of its last module.
            expected = {"lp_b"} if inner_return_composite is False else {"lp_a", "lp_b"}
            if aggregate:
                torch.testing.assert_close(lp, sum(td[key] for key in expected))
                return
            assert set(lp.keys()) == expected
            for key in expected:
                torch.testing.assert_close(lp[key], td[key])

    @pytest.mark.parametrize("aggregate", [True, False])
    def test_probtdseq_log_prob_key(self, aggregate):
        # The sequence gives its one log-probability key. With several, the
        # lookup fails with an AttributeError, as on main, so getattr() with a
        # default, hasattr() and inspect.getmembers() keep working.
        with set_composite_lp_aggregate(aggregate):
            prob_a = ProbabilisticTensorDictModule(
                in_keys=["loc"],
                out_keys=["a"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
                return_log_prob=True,
            )
            prob_b = ProbabilisticTensorDictModule(
                in_keys={"loc": "a"},
                out_keys=["b"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
                return_log_prob=True,
                log_prob_key="lp_b",
            )
            seq = ProbabilisticTensorDictSequential(prob_a)
            assert seq.log_prob_key == prob_a.log_prob_key
            nested = ProbabilisticTensorDictSequential(
                TensorDictModule(lambda x: x, in_keys=["x"], out_keys=["loc"]),
                seq,
            )
            assert nested.log_prob_key == prob_a.log_prob_key
            assert hasattr(nested, "log_prob_key")
            seq2 = ProbabilisticTensorDictSequential(prob_a, prob_b)
            with pytest.raises(AttributeError):
                seq2.log_prob_key
            assert getattr(seq2, "log_prob_key", None) is None
            assert not hasattr(seq2, "log_prob_key")
            # getmembers() reads every attribute, the deprecated ones too.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                assert isinstance(inspect.getmembers(seq2), list)

    @pytest.mark.parametrize("aggregate_changed", [False, True])
    def test_probtdseq_log_prob_key_composite(self, aggregate_changed):
        # A composite distribution with composite_lp_aggregate() False has
        # several log-probability keys, and a change of the setting after the
        # build makes log_prob_keys raise: log_prob_key is then missing.
        with set_composite_lp_aggregate(aggregate_changed):
            seq = ProbabilisticTensorDictSequential(
                ProbabilisticTensorDictModule(
                    in_keys=["params"],
                    out_keys=["a", "b"],
                    distribution_class=CompositeDistribution,
                    distribution_kwargs={
                        "distribution_map": {"a": Normal, "b": Normal}
                    },
                    return_log_prob=True,
                )
            )
        with set_composite_lp_aggregate(False):
            with pytest.raises(AttributeError):
                seq.log_prob_key
            assert getattr(seq, "log_prob_key", None) is None
            assert not hasattr(seq, "log_prob_key")
            if aggregate_changed:
                # getmembers() also reads log_prob_keys, which raises here.
                return
            # getmembers() reads every attribute, the deprecated ones too.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                assert isinstance(inspect.getmembers(seq), list)

    def test_probtdseq_log_prob_aggregate_changed_after_init(self):
        # Modules built while composite_lp_aggregate() was True keep the
        # default names when the sequence is scored with it False.
        with set_composite_lp_aggregate(True):
            seq = ProbabilisticTensorDictSequential(
                ProbabilisticTensorDictModule(
                    in_keys=["params"],
                    out_keys=["a", "b"],
                    distribution_class=CompositeDistribution,
                    distribution_kwargs={
                        "distribution_map": {"a": Normal, "b": Normal}
                    },
                    return_log_prob=True,
                ),
                return_composite=True,
            )
            params = TensorDict(
                {k: {"loc": torch.zeros(3), "scale": torch.ones(3)} for k in "ab"},
                batch_size=[3],
            )
            td = seq(TensorDict(params=params, batch_size=[3]))
        with set_composite_lp_aggregate(False):
            lp = seq.log_prob(td)
        assert set(lp.keys()) == {"a_log_prob", "b_log_prob"}

    def test_probtdseq_nested_last_return_composite_false(self):
        # return_composite=False passes _requires_sample into the last module.
        # A nested sequence must accept that argument and sample its out key.
        inner = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda x: x + 1, in_keys=["x"], out_keys=["loc0"]),
            ProbabilisticTensorDictModule(
                in_keys={"loc": "loc0"},
                out_keys=["sample0"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            ),
        )
        loc1 = TensorDictModule(lambda x: x - 1, in_keys=["x"], out_keys=["loc1"])
        prob1 = ProbabilisticTensorDictModule(
            in_keys={"loc": "loc1"},
            out_keys=["sample1"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
        )
        seq = ProbabilisticTensorDictSequential(
            loc1, prob1, inner, return_composite=False
        )
        x = torch.zeros(3)
        with set_interaction_type("mode"):
            out = seq(TensorDict(x=x, batch_size=[3]))
        torch.testing.assert_close(out["sample0"], x + 1)
        torch.testing.assert_close(out["sample1"], x - 1)

    def test_probtdseq_preset_last_sample_keeps_earlier_sample(self):
        # _requires_sample is False because "b" is already an out key of the
        # prefix. The earlier module must still sample "a". "b" stays preset.
        loc_a = torch.tensor([2.0, 3.0])
        preset_b = torch.tensor([7.0, 8.0])
        seq = ProbabilisticTensorDictSequential(
            ProbabilisticTensorDictModule(
                in_keys={"loc": "loc_a"},
                out_keys=["a"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            ),
            TensorDictModule(lambda b: b, in_keys=["preset_b"], out_keys=["b"]),
            ProbabilisticTensorDictModule(
                in_keys={"loc": "loc_b"},
                out_keys=["b"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            ),
        )
        assert seq.return_composite is True
        assert seq._requires_sample is False
        with set_interaction_type("mode"):
            out = seq(
                TensorDict(
                    loc_a=loc_a,
                    preset_b=preset_b,
                    loc_b=torch.zeros_like(preset_b),
                    batch_size=[2],
                )
            )
        assert "a" in out.keys()
        torch.testing.assert_close(out["a"], loc_a)
        torch.testing.assert_close(out["b"], preset_b)

    def test_probtdseq_single_module_samples_mode(self):
        seq = ProbabilisticTensorDictSequential(
            ProbabilisticTensorDictModule(
                in_keys={"loc": "loc"},
                out_keys=["sample"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            )
        )
        assert seq._requires_sample is True
        loc = torch.tensor([1.5, -2.0])
        with set_interaction_type("mode"):
            out = seq(TensorDict(loc=loc, batch_size=[2]))
        torch.testing.assert_close(out["sample"], loc)

    def test_probtdseq_requires_sample_true_samples_last_key(self):
        # The last sample key is not produced earlier, so forward samples it.
        loc_a = torch.tensor([1.0, 2.0])
        loc_b = torch.tensor([-3.0, 4.0])
        seq = ProbabilisticTensorDictSequential(
            ProbabilisticTensorDictModule(
                in_keys={"loc": "loc_a"},
                out_keys=["a"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            ),
            ProbabilisticTensorDictModule(
                in_keys={"loc": "loc_b"},
                out_keys=["b"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            ),
        )
        assert seq.return_composite is True
        assert seq._requires_sample is True
        with set_interaction_type("mode"):
            out = seq(TensorDict(loc_a=loc_a, loc_b=loc_b, batch_size=[2]))
        torch.testing.assert_close(out["a"], loc_a)
        torch.testing.assert_close(out["b"], loc_b)

    @pytest.mark.parametrize(
        "last,return_composite", [(True, True), (False, True), (True, False)]
    )
    def test_probtdseq_nested_keeps_preset_sample(self, last, return_composite):
        # The inner prefix writes "b", so the inner sequence only scores it.
        # Nesting must not make it resample "b".
        inner = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda b: b, in_keys=["preset_b"], out_keys=["b"]),
            ProbabilisticTensorDictModule(
                in_keys={"loc": "loc_b"},
                out_keys=["b"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
                return_log_prob=True,
            ),
        )
        assert inner._requires_sample is False
        prob_a = ProbabilisticTensorDictModule(
            in_keys={"loc": "loc_a"},
            out_keys=["a"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
        )
        modules = (prob_a, inner) if last else (inner, prob_a)
        seq = ProbabilisticTensorDictSequential(
            *modules, return_composite=return_composite
        )
        preset_b = torch.tensor([7.0, 8.0])
        loc_b = torch.tensor([-1.0, 4.0])
        with set_interaction_type("mode"):
            out = seq(
                TensorDict(
                    loc_a=torch.tensor([2.0, 3.0]),
                    loc_b=loc_b,
                    preset_b=preset_b,
                    batch_size=[2],
                )
            )
        torch.testing.assert_close(out["b"], preset_b)
        torch.testing.assert_close(
            out["b_log_prob"], Normal(loc_b, 1.0).log_prob(preset_b)
        )

    @pytest.mark.parametrize("last", [True, False])
    def test_probtdseq_nested_receives_kwargs(self, last):
        class Shift(nn.Module):
            def forward(self, x, shift=0.0):
                return x + shift

        def normal(loc, out):
            return ProbabilisticTensorDictModule(
                in_keys={"loc": loc},
                out_keys=[out],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            )

        inner = ProbabilisticTensorDictSequential(
            TensorDictModule(Shift(), in_keys=["x"], out_keys=["loc_i"]),
            normal("loc_i", "i"),
            return_composite=True,
        )
        modules = (normal("x", "a"), inner)
        if not last:
            modules = modules + (normal("x", "b"),)
        seq = ProbabilisticTensorDictSequential(*modules, return_composite=True)
        x = torch.zeros(2)
        with set_interaction_type("mode"):
            out = seq(TensorDict(x=x, batch_size=[2]), shift=10.0)
        torch.testing.assert_close(out["i"], x + 10.0)

    @pytest.mark.parametrize("nested", [True, False])
    def test_probtdseq_receives_kwargs_without_composite(self, nested):
        # https://github.com/pytorch/tensordict/issues/2976
        class Shift(nn.Module):
            def forward(self, x, shift=0.0):
                return x + shift

        modules = [
            TensorDictModule(Shift(), in_keys=["x"], out_keys=["loc"]),
            ProbabilisticTensorDictModule(
                in_keys=["loc"],
                out_keys=["a"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
            ),
        ]
        if nested:
            modules = [
                ProbabilisticTensorDictSequential(*modules, return_composite=False),
                TensorDictModule(Shift(), in_keys=["a"], out_keys=["loc_b"]),
                ProbabilisticTensorDictModule(
                    in_keys={"loc": "loc_b"},
                    out_keys=["b"],
                    distribution_class=Normal,
                    distribution_kwargs={"scale": 1.0},
                ),
            ]
        seq = ProbabilisticTensorDictSequential(*modules, return_composite=False)
        x = torch.zeros(2)

        def td():
            return TensorDict(x=x, batch_size=[2])

        with set_interaction_type("mode"):
            params = seq.get_dist_params(td(), shift=10.0)
            torch.testing.assert_close(params["loc"], x + 10.0)
            out = TensorDict()
            params = seq.get_dist_params(td(), out, shift=10.0)
            assert params is out
            assert "x" not in out
            torch.testing.assert_close(out["loc"], x + 10.0)
            dist = seq.get_dist(td(), shift=10.0)
            out = seq(td(), shift=10.0)
            lp = seq.log_prob(
                td().update(out.select("a", "b", strict=False)), shift=10.0
            )
        if nested:
            torch.testing.assert_close(params["loc_b"], x + 20.0)
            torch.testing.assert_close(dist.loc, x + 20.0)
            torch.testing.assert_close(out["b"], x + 20.0)
            torch.testing.assert_close(lp, dist.log_prob(x + 20.0))
        else:
            torch.testing.assert_close(dist.loc, x + 10.0)
            torch.testing.assert_close(out["a"], x + 10.0)
            torch.testing.assert_close(lp, dist.log_prob(x + 10.0))
        with pytest.raises(TypeError, match="shift"):
            seq.log_prob(out, dist=dist, shift=10.0)

    @pytest.mark.parametrize("custom", [True, False])
    @pytest.mark.parametrize("nested_last", [True, False])
    def test_probtdseq_log_prob_kwargs_last_module(self, custom, nested_last):
        # https://github.com/pytorch/tensordict/issues/2976: a last module whose
        # log_prob takes keyword arguments keeps getting them.
        class Shift(nn.Module):
            def forward(self, x, shift=0.0):
                return x + shift

        class ScaledLogProb(ProbabilisticTensorDictModule):
            def log_prob(self, tensordict, *, dist=None, shift=1.0):
                return super().log_prob(tensordict, dist=dist) * shift

        cls = ScaledLogProb if custom else ProbabilisticTensorDictModule
        last = cls(
            in_keys=["loc"],
            out_keys=["a"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
        )
        if nested_last:
            last = ProbabilisticTensorDictSequential(
                TensorDictModule(Shift(), in_keys=["x"], out_keys=["y"]),
                last,
                return_composite=False,
            )
        seq = ProbabilisticTensorDictSequential(
            TensorDictModule(Shift(), in_keys=["x"], out_keys=["loc"]),
            last,
            return_composite=False,
        )
        x = torch.zeros(2)
        td = TensorDict(x=x, a=x, batch_size=[2])
        lp = seq.log_prob(td, shift=2.0)
        if custom:
            # As on main: the keyword goes to the last module's log_prob only.
            expected = Normal(x, 1.0).log_prob(x) * 2.0
        else:
            expected = Normal(x + 2.0, 1.0).log_prob(x)
        torch.testing.assert_close(lp, expected)

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
        with set_skip_existing(["other"]):
            td = module(TensorDict({"out": torch.zeros(())}, []))
        assert (td["out"] == 1).all()
        with set_skip_existing(["out"]):
            td = module(TensorDict({"out": torch.zeros(())}, []))
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

    def test_selected_keys(self):
        value_module = TensorDictModule(
            lambda value: value + 1, in_keys=["input"], out_keys=["value"]
        )
        memory_module = TensorDictModule(
            lambda memory: memory + 1,
            in_keys=["memory"],
            out_keys=[("next", "memory")],
        )
        module = TensorDictSequential(value_module, memory_module)
        td = TensorDict(
            {
                "input": torch.zeros(()),
                "value": torch.full((), 10),
                "memory": torch.zeros(()),
                "next": {"memory": torch.full((), 10)},
            },
            [],
        )

        with set_skip_existing(["value"]):
            module(td)

        assert td["value"].item() == 10
        assert td["next", "memory"].item() == 1

    def test_nested_sequence_restores_mode(self):
        # The inner and outer sequences share the decorator of
        # TensorDictSequential.forward: the inner call must not change the
        # mode that the outer call restores.
        class Inner(TensorDictModuleBase):
            in_keys = ["in"]
            out_keys = ["out"]

            def __init__(self):
                super().__init__()
                self.seq = TensorDictSequential(
                    TensorDictModule(lambda x: x + 1, in_keys=["in"], out_keys=["out"])
                )

            @set_skip_existing(True)
            def forward(self, tensordict):
                return self.seq(tensordict)

        module = TensorDictSequential(Inner())
        td = module(TensorDict({"in": torch.zeros(())}, []))
        assert (td["out"] == 1).all()
        assert skip_existing() is False

    def test_exception_restores_mode(self):
        # The forward changes the mode without restoring it, then raises: the
        # decorator of TensorDictModule.forward must restore the mode it saw.
        def fail(x):
            set_skip_existing(True).__enter__()
            raise ValueError("fail")

        module = TensorDictModule(fail, in_keys=["in"], out_keys=["out"])
        with set_skip_existing(["other"]):
            with pytest.raises(ValueError, match="fail"):
                module(TensorDict({"in": torch.zeros(())}, []))
            assert skip_existing() == ["other"]
        assert skip_existing() is False

    def test_unchanged_mode_not_written_back(self, monkeypatch):
        # The decorator of forward writes the mode back only if the call
        # changed it, as setting an unchanged mode does nothing.
        _skip_existing = tensordict.nn.utils._skip_existing
        calls = []
        set_mode = _skip_existing.set_mode

        def counting_set_mode(mode):
            calls.append(mode)
            set_mode(mode)

        def leak(x):
            set_skip_existing(True).__enter__()
            return x + 1

        seq = TensorDictSequential(
            TensorDictModule(lambda x: x + 1, in_keys=["in"], out_keys=["mid"]),
            TensorDictModule(lambda x: x + 1, in_keys=["mid"], out_keys=["out"]),
        )
        leaky = TensorDictModule(leak, in_keys=["in"], out_keys=["out"])
        monkeypatch.setattr(_skip_existing, "set_mode", counting_set_mode)
        td = seq(TensorDict({"in": torch.zeros(())}, []))
        assert (td["out"] == 2).all()
        assert calls == []
        assert skip_existing() is False

        leaky(TensorDict({"in": torch.zeros(())}, []))
        assert calls == [True, False]
        assert skip_existing() is False


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
            assert mod.out_keys_source == ["c", "d", "e"]
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

    def test_tdmodule_missing_out_key(self, out_d_key, unpack):
        # One missing key and several keys with one missing raise the same
        # error, which is both a ValueError and a RuntimeError
        mod = TensorDictModule(
            lambda x, y: (x + 2, y + 2, x), in_keys=["a", "b"], out_keys=["c", "d", "e"]
        )
        keys = (*out_d_key, "z") if unpack else ("z",)
        for error in (ValueError, RuntimeError):
            with pytest.raises(error, match="Can't select non existent key.*z"):
                mod.select_out_keys(*keys)
        assert mod.out_keys == ["c", "d", "e"]

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
            assert mod.out_keys_source == ["c", "d", "e"]
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

    def test_tdmodule_missing_in_key(self, out_d_key, unpack):
        # strict=False passes the missing in_key "b" as None
        mod = TensorDictModule(
            lambda x, y: (x + 2, x + 2 if y is None else y + 2, x),
            in_keys=["a", "b"],
            out_keys=["c", "d", "e"],
        )
        if unpack:
            mod.select_out_keys(*out_d_key)
            td = mod(TensorDict(a=torch.zeros(())))
            assert set(td.keys()) == {"a", *out_d_key}
            td = mod(TensorDict(a=torch.zeros(())), tensordict_out=TensorDict())
            assert set(td.keys()) == {*out_d_key}
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod.select_out_keys(out_d_key)

    def test_tdmodule_keeps_other_input_entries(self, out_d_key, unpack):
        # select_out_keys removes the out-keys that are not selected, and
        # keeps the input entries, including an in-key that is also an out-key
        mod = TensorDictModule(
            lambda x, y: (x + 2, y + 2, x, y, x),
            in_keys=["a", "b"],
            out_keys=["c", "d", "e", "b", ("n", "f")],
        )
        if unpack:
            mod.select_out_keys(*out_d_key)
            td = TensorDict(a=torch.zeros(()), b=torch.ones(()), other=torch.zeros(()))
            assert mod(td) is td
            assert set(td.keys(True)) == {"a", "b", "other", *out_d_key}
            td = TensorDict(
                a=torch.zeros(()), b=torch.ones(()), n={"g": torch.zeros(())}
            )
            mod(td)
            assert set(td.keys(True)) == {"a", "b", "n", ("n", "g"), *out_d_key}
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod.select_out_keys(out_d_key)

    def test_tdbase_missing_in_key(self, out_d_key, unpack):
        class MyModule(TensorDictModuleBase):
            in_keys = ["a", "b"]
            out_keys = ["c", "d", "e"]

            def forward(self, tensordict):
                tensordict["c"] = tensordict["a"] + 2
                tensordict["d"] = tensordict.get("b", tensordict["a"]) + 2
                tensordict["e"] = tensordict["d"] + 2
                return tensordict

        mod = MyModule()
        if unpack:
            mod.select_out_keys(*out_d_key)
            td = mod(TensorDict(a=torch.zeros(())))
            assert set(td.keys()) == {"a", *out_d_key}
            td = mod(tensordict=TensorDict(a=torch.zeros(())))
            assert set(td.keys()) == {"a", *out_d_key}
        else:
            with pytest.raises(
                (RuntimeError, ValueError),
                match=r"key should be a |Can't select non existent",
            ):
                mod.select_out_keys(out_d_key)

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
    elif is_npu_available():
        module = module.to("npu:0")
        assert module.td.device.type == "npu"


@pytest.mark.parametrize(
    "original_device",
    [
        None,
        torch.device("cpu"),
        get_device(),
    ],
)
@pytest.mark.parametrize(
    "new_device",
    [
        torch.device("cpu"),
        get_device(),
    ],
)
@pytest.mark.parametrize("tc", [True, False], ids=["tc", "td"])
def test_to_context(original_device, new_device, tc):
    if tc:

        class MyTC(TensorClass):
            x: torch.Tensor
            y: torch.Tensor | None = None

    mod = TensorDictModule(
        lambda x: x + 1,
        in_keys=["x"],
        out_keys=["y"],
    )
    td = TensorDict(x=torch.zeros(3), batch_size=[3], device=original_device)
    if tc:
        td = MyTC.from_tensordict(td)
    with td.to(new_device) as td_new:
        td_new = mod(td_new)
    if original_device is not None:
        y = td["y"] if not tc else td.y
        x = td["x"] if not tc else td.x
        assert y.device == x.device
    else:
        y = td["y"] if not tc else td.y
        assert y.device == new_device

    if original_device is not None:
        x = td["x"] if not tc else td.x
        assert x.device == original_device


class _TanhNormalJoint(distributions.TransformedDistribution):
    # tanh(Normal) with joint sample-and-score methods. Without a cache, its
    # log_prob recovers the draw as atanh(sample), which fails once tanh
    # saturates in float32.
    def __init__(self, loc, scale):
        super().__init__(
            distributions.Normal(loc, scale), [distributions.TanhTransform()]
        )

    def _score(self, x):
        transform = self.transforms[0]
        y = transform(x)
        return y, self.base_dist.log_prob(x) - transform.log_abs_det_jacobian(x, y)

    def rsample_and_log_prob(self, sample_shape=()):
        return self._score(self.base_dist.rsample(sample_shape))

    def sample_and_log_prob(self, sample_shape=()):
        return self._score(self.base_dist.sample(sample_shape))


class _TanhNormalJointNoRsample(_TanhNormalJoint):
    has_rsample = False


class TestProbabilisticTensorDictModule:
    def test_cache_dist_deprecation(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ProbabilisticTensorDictModule(
                in_keys=["loc"], out_keys=["sample"], cache_dist=False
            )
        with pytest.warns(
            DeprecationWarning,
            match=r"^ProbabilisticTensorDictModule\(cache_dist=True\) is deprecated "
            r"and will be removed in TensorDict 0\.17\.$",
        ) as record:
            module = ProbabilisticTensorDictModule(
                in_keys=["loc"], out_keys=["sample"], cache_dist=True
            )
        assert record[0].filename == __file__
        assert module.cache_dist is hasattr(Delta, "update")

    def test_distribution_class_str(self):
        module = ProbabilisticTensorDictModule(
            in_keys=["param"], out_keys=["sample"], distribution_class="Delta"
        )
        assert module.distribution_class is Delta
        with pytest.warns(
            DeprecationWarning,
            match=r"^tensordict\.nn\.distributions\.distributions_maps is "
            r"deprecated and will be removed in TensorDict 0\.17\.$",
        ) as record:
            distributions_maps = tensordict.nn.distributions.distributions_maps
        assert record[0].filename == __file__
        assert distributions_maps is tensordict.nn.distributions._distributions_maps
        assert distributions_maps == {
            "normalparamextractor": NormalParamExtractor,
            "addstateindependentnormalscale": AddStateIndependentNormalScale,
            "delta": Delta,
            "onehotcategorical": OneHotCategorical,
        }
        assert "distributions_maps" not in tensordict.nn.distributions.__all__

    def test_deterministic_register_deprecation(self):
        probabilistic = tensordict.nn.probabilistic
        for name in ("dist_name", "dist_cls", "dist_has_enum_support"):
            assert not hasattr(probabilistic, name)
        with pytest.warns(
            DeprecationWarning,
            match=r"^tensordict\.nn\.probabilistic\.DETERMINISTIC_REGISTER is "
            r"deprecated and will be removed in TensorDict 0\.17\. Use a "
            r"deterministic_sample attribute on the distribution instead\.$",
        ) as record:
            register = probabilistic.DETERMINISTIC_REGISTER
        assert record[0].filename == __file__
        assert register is probabilistic._DETERMINISTIC_REGISTER
        assert register[Normal] is InteractionType.MEAN
        assert register[Categorical] is InteractionType.MODE
        assert register[distributions.LogisticNormal] is InteractionType.DETERMINISTIC
        assert register[Delta] is InteractionType.DETERMINISTIC

    @pytest.mark.parametrize(
        "name",
        [
            "CONST_SQRT_2",
            "CONST_INV_SQRT_2PI",
            "CONST_INV_SQRT_2",
            "CONST_LOG_INV_SQRT_2PI",
            "CONST_LOG_SQRT_2PI_E",
        ],
    )
    def test_truncated_normal_constants_deprecation(self, name):
        truncated_normal = tensordict.nn.distributions.truncated_normal
        with pytest.warns(
            DeprecationWarning,
            match=rf"^tensordict\.nn\.distributions\.truncated_normal\.{name} is "
            rf"deprecated and will be removed in TensorDict 0\.17\.$",
        ) as record:
            value = getattr(truncated_normal, name)
        assert record[0].filename == __file__
        assert value == getattr(truncated_normal, f"_{name}")

    def test_truncated_normal_support(self):
        a, b = torch.tensor([8.0]), torch.tensor([12.0])
        dist = TruncatedNormal(
            torch.tensor([10.0]), torch.tensor([2.0]), a, b, validate_args=True
        )
        assert torch.equal(dist.support.lower_bound, a)
        assert torch.equal(dist.support.upper_bound, b)
        sample = dist.rsample((100,))
        assert dist.support.check(sample).all()
        # log_prob and cdf check the value, not the standardised value,
        # against the support, and still accept Python numbers
        dist.log_prob(sample)
        dist.cdf(sample)
        torch.testing.assert_close(
            dist.log_prob(11.0), dist.log_prob(torch.tensor([11.0]))
        )
        torch.testing.assert_close(dist.cdf(11.0), dist.cdf(torch.tensor([11.0])))
        with pytest.raises(ValueError, match="to be within the support"):
            dist.log_prob(torch.tensor([12.5]))
        with pytest.raises(ValueError, match="to be within the support"):
            dist.cdf(7.5)

    def test_truncated_normal_number_args(self):
        dist = TruncatedNormal(0.0, 2.0, -1.0, 1.0)
        ref = TruncatedNormal(
            torch.tensor(0.0), torch.tensor(2.0), torch.tensor(-1.0), torch.tensor(1.0)
        )
        assert dist.batch_shape == ref.batch_shape == torch.Size([])
        for attr in ("scale", "mean", "variance"):
            torch.testing.assert_close(getattr(dist, attr), getattr(ref, attr))
        torch.testing.assert_close(dist.entropy(), ref.entropy())
        value = torch.tensor(0.5)
        torch.testing.assert_close(dist.log_prob(value), ref.log_prob(value))
        assert TruncatedNormal(torch.zeros(3), 2.0, -1.0, 1.0).batch_shape == (3,)
        # a Number scale is clamped as a tensor scale is
        assert TruncatedNormal(0.0, 0.0, -1.0, 1.0).scale == TruncatedNormal.eps

    def test_truncated_normal_entropy(self):
        # entropy is a method, as in torch.distributions.Distribution
        loc = torch.tensor([0.0, 1.0, -0.5])
        scale = torch.tensor([1.0, 2.0, 0.5])
        a, b = loc - 2 * scale, loc + 1.5 * scale
        dist = TruncatedNormal(loc, scale, a, b)
        entropy = dist.entropy()
        assert entropy.shape == dist.batch_shape == (3,)
        # the closed form matches a Monte Carlo estimate of -E[log p(x)]
        torch.manual_seed(0)
        sample = dist.rsample((200_000,))
        torch.testing.assert_close(
            entropy, -dist.log_prob(sample).mean(0), atol=2e-2, rtol=0
        )
        # the standard distribution exposes entropy() too
        std = TruncatedStandardNormal(a, b)
        assert std.entropy().shape == std.batch_shape == (3,)

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

    @set_composite_lp_aggregate(False)
    @pytest.mark.parametrize("inplace", [False, "empty"])
    def test_tdprobseq_inplace_false_keeps_input(self, inplace):
        # https://github.com/pytorch/tensordict/issues/2257
        model = ProbabilisticTensorDictSequential(
            TensorDictModule(
                lambda x: x + 1,
                in_keys=["input"],
                out_keys=["output"],
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
        assert output is not input
        assert set(input.keys()) == {"input"}
        assert {"output", "sample", "sample_log_prob"} <= set(output.keys())

    @pytest.mark.parametrize("return_log_prob", [True, False])
    @set_composite_lp_aggregate(False)
    @pytest.mark.parametrize("num_samples", [2, None, (2,)])
    def test_probabilistic_n_samples(self, return_log_prob, num_samples):
        prob = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1},
            return_log_prob=return_log_prob,
            num_samples=num_samples,
            default_interaction_type="random",
        )
        if num_samples is None:
            num_samples = ()
        elif num_samples == 2:
            num_samples = (2,)
        # alone
        td = TensorDict(loc=torch.randn(3, 4), batch_size=[3])
        td = prob(td)
        assert "sample" in td
        assert td.shape == (*num_samples, 3)
        assert td["sample"].shape == (*num_samples, 3, 4)
        if return_log_prob:
            assert "sample_log_prob" in td
        assert prob.dist_sample_keys == ["sample"]
        assert prob.dist_params_keys == ["loc"]

    @pytest.mark.parametrize("return_log_prob", [True, False])
    @set_composite_lp_aggregate(False)
    @pytest.mark.parametrize("interaction", [None, "mode", "mean", "deterministic"])
    def test_probabilistic_n_samples_non_random(self, return_log_prob, interaction):
        # The deterministic value is repeated num_samples times
        prob = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1},
            return_log_prob=return_log_prob,
            num_samples=3,
        )
        loc = torch.randn(2, 4)
        td = TensorDict(loc=loc, batch_size=[2])
        if interaction is None:
            td = prob(td)
        else:
            with set_interaction_type(interaction):
                td = prob(td)
        assert td.shape == (3, 2)
        assert td["sample"].shape == (3, 2, 4)
        assert (td["sample"] == loc).all()
        if return_log_prob:
            assert td["sample_log_prob"].shape == (3, 2, 4)

    @pytest.mark.parametrize("return_log_prob", [True, False])
    @set_composite_lp_aggregate(False)
    @pytest.mark.parametrize("interaction", [None, "mode", "mean", "deterministic"])
    def test_probabilistic_n_samples_non_random_composite(
        self, return_log_prob, interaction
    ):
        prob = ProbabilisticTensorDictModule(
            in_keys=["params"],
            out_keys=["a", "b"],
            distribution_class=CompositeDistribution,
            distribution_kwargs={"distribution_map": {"a": Normal, "b": Normal}},
            return_log_prob=return_log_prob,
            num_samples=3,
        )
        loc = torch.randn(2, 4)
        params = TensorDict(loc=loc, scale=torch.ones(2, 4), batch_size=[2])
        td = TensorDict(
            params=TensorDict(a=params, b=params.clone(), batch_size=[2]),
            batch_size=[2],
        )
        if interaction is None:
            td = prob(td)
        else:
            with set_interaction_type(interaction):
                td = prob(td)
        assert td.shape == (3, 2)
        for key in ("a", "b"):
            assert td[key].shape == (3, 2, 4)
            assert (td[key] == loc).all()
            if return_log_prob:
                assert td[f"{key}_log_prob"].shape == (3, 2, 4)

    @pytest.mark.parametrize(
        "dist_class", [_TanhNormalJoint, _TanhNormalJointNoRsample]
    )
    @pytest.mark.parametrize("num_samples", [None, 3])
    @set_composite_lp_aggregate(False)
    def test_joint_sample_and_log_prob(self, dist_class, num_samples):
        # The module scores the draw with the distribution's joint method,
        # rsample_and_log_prob or sample_and_log_prob depending on has_rsample
        loc = torch.tensor([[20.0, -20.0]])
        scale = torch.full_like(loc, 0.1)
        module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=dist_class,
            default_interaction_type="random",
            return_log_prob=True,
            num_samples=num_samples,
        )
        torch.manual_seed(0)
        td = module(TensorDict(loc=loc, scale=scale, batch_size=[1]))
        dist = dist_class(loc, scale)
        joint = (
            dist.rsample_and_log_prob if dist.has_rsample else dist.sample_and_log_prob
        )
        torch.manual_seed(0)
        action, log_prob = joint(module.num_samples or ())
        assert log_prob.isfinite().all()
        assert (td["action"] == action).all()
        torch.testing.assert_close(td["action_log_prob"], log_prob)

    @set_composite_lp_aggregate(False)
    def test_torch_dist_sample_then_log_prob(self):
        # torch distributions have no joint method: the module scores the
        # sample with log_prob
        module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=Normal,
            default_interaction_type="random",
            return_log_prob=True,
        )
        loc, scale = torch.randn(3, 2), torch.rand(3, 2) + 0.1
        torch.manual_seed(0)
        td = module(TensorDict(loc=loc, scale=scale, batch_size=[3]))
        torch.manual_seed(0)
        action = Normal(loc, scale).rsample()
        assert (td["action"] == action).all()
        torch.testing.assert_close(
            td["action_log_prob"], Normal(loc, scale).log_prob(action)
        )

    @pytest.mark.parametrize("return_log_prob", [True, False])
    def test_not_a_distribution(self, return_log_prob):
        # The check for a joint method must not hide the TypeError for an
        # object that is not a distribution
        class NotADist:
            def __init__(self, loc):
                self.loc = loc

        module = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["action"],
            distribution_class=NotADist,
            default_interaction_type="random",
            return_log_prob=return_log_prob,
        )
        with pytest.raises(TypeError, match="Expected Distribution"):
            module(TensorDict(loc=torch.zeros(3), batch_size=[3]))

    @pytest.mark.parametrize("return_log_prob", [True, False])
    @pytest.mark.parametrize("return_composite", [True, False])
    @pytest.mark.parametrize("default_interaction_type", ["random", "deterministic"])
    @set_composite_lp_aggregate(False)
    def test_probabilistic_seq_n_samples(
        self,
        return_log_prob,
        return_composite,
        default_interaction_type,
    ):
        aggregate_probabilities = composite_lp_aggregate()
        prob = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1},
            return_log_prob=return_log_prob,
            num_samples=2,
            default_interaction_type=default_interaction_type,
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
    @pytest.mark.parametrize("default_interaction_type", ["random", "deterministic"])
    @set_composite_lp_aggregate(False)
    def test_intermediate_probabilistic_seq_n_samples(
        self,
        return_log_prob,
        return_composite,
        default_interaction_type,
    ):
        prob = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1},
            return_log_prob=return_log_prob,
            num_samples=2,
            default_interaction_type=default_interaction_type,
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
        assert isinstance(seq[-2:], ProbabilisticTensorDictSequential)

        seq = ProbabilisticTensorDictSequential(m0, m1, m2, return_composite=True)
        assert isinstance(seq[0], ProbabilisticTensorDictModule)
        assert isinstance(seq[:2], ProbabilisticTensorDictSequential)
        assert isinstance(seq[-2:], ProbabilisticTensorDictSequential)

    @pytest.mark.parametrize("return_composite", [False, True])
    def test_prob_seq_subsequence_keeps_options(self, return_composite):
        loc = TensorDictModule(lambda x: x + 1, ["x"], ["loc"])
        prob0 = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample0"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
        )
        prob1 = ProbabilisticTensorDictModule(
            in_keys={"loc": "sample0"},
            out_keys=["sample1"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
        )
        seq = ProbabilisticTensorDictSequential(
            loc,
            prob0,
            prob1,
            return_composite=return_composite,
            partial_tolerant=True,
            inplace=False,
        )
        for sub in (seq.select_subsequence(out_keys=["sample0"]), seq[1:], seq[:1]):
            assert sub.partial_tolerant is True
            assert sub.inplace is False

    @pytest.mark.skipif(IS_FB, reason="Not working within fbcode")
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

    @pytest.mark.parametrize("error", [NotImplementedError, AttributeError])
    def test_mean_falls_back_to_empirical_estimate(self, error):
        class BrokenMeanNormal(Normal):
            @property
            def mean(self):
                raise error

        n_empirical_estimate = 8
        module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["sample"],
            distribution_class=BrokenMeanNormal,
            default_interaction_type="mean",
            n_empirical_estimate=n_empirical_estimate,
        )
        td = TensorDict(loc=torch.zeros(4), scale=torch.ones(4))
        torch.manual_seed(0)
        sample = module(td)["sample"]
        torch.manual_seed(0)
        dist = BrokenMeanNormal(td["loc"], td["scale"])
        expected = dist.rsample((n_empirical_estimate,)).mean(0)
        torch.testing.assert_close(sample, expected)

    def test_mean_does_not_sample_when_implemented(self):
        module = ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["sample"],
            distribution_class=Normal,
            default_interaction_type="mean",
        )
        td = TensorDict(loc=torch.tensor([1.0, 2.0, 3.0]), scale=torch.ones(3))
        torch.manual_seed(0)
        state_before = torch.get_rng_state()
        sample = module(td)["sample"]
        assert torch.equal(torch.get_rng_state(), state_before)
        assert torch.equal(sample, td["loc"])

    @pytest.mark.parametrize("param", ["logits", "probs"])
    def test_onehot_categorical_mode_ties(self, param):
        # Tied logits, e.g. from a zero-initialised head, give the first
        # maximum, as torch's OneHotCategorical does (pytorch/tensordict#2145).
        logits = torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.3, 2.0, 1.0]])
        expected = torch.tensor([[1, 0, 0], [1, 0, 0], [0, 1, 0]])
        kwargs = {param: logits if param == "logits" else logits.softmax(-1)}
        dist = OneHotCategorical(**kwargs)
        assert dist.mode.dtype == torch.long
        assert torch.equal(dist.mode, expected)
        assert torch.equal(dist.deterministic_sample, expected)
        assert torch.equal(dist.mode, distributions.OneHotCategorical(**kwargs).mode)
        module = ProbabilisticTensorDictModule(
            in_keys=[param],
            out_keys=["action"],
            distribution_class=OneHotCategorical,
        )
        td = module(TensorDict(kwargs, batch_size=[3]))
        assert torch.equal(td["action"], expected)

    def test_deterministic_sample_read_once(self):
        reads = []

        class CountingDelta(Delta):
            @property
            def deterministic_sample(self):
                reads.append(None)
                return super().deterministic_sample

        module = ProbabilisticTensorDictModule(
            in_keys=["param"],
            out_keys=["out"],
            distribution_class=CountingDelta,
            default_interaction_type="deterministic",
        )
        td = module(TensorDict(param=torch.ones(3)))
        assert torch.equal(td["out"], torch.ones(3))
        assert len(reads) == 1

    @pytest.mark.parametrize("param", ["logits", "probs"])
    def test_onehot_categorical_rsample(self, param):
        torch.manual_seed(0)
        logits = torch.randn(3, 4, requires_grad=True)
        if param == "logits":
            dist = OneHotCategorical(logits=logits)
        else:
            dist = OneHotCategorical(probs=logits.softmax(-1))
        assert dist.rsample().shape == (3, 4)
        sample = dist.rsample((2,))
        assert sample.shape == (2, 3, 4)
        assert ((sample == 0) | (sample == 1)).all()
        assert (sample.sum(-1) == 1).all()
        (sample * torch.arange(1.0, 5.0)).sum().backward()
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    @pytest.mark.parametrize("base", ["Delta", "LogisticNormal"])
    def test_deterministic_sample_independent(self, base):
        # Independent has no deterministic_sample, and the register maps its
        # base to DETERMINISTIC: the module uses the base's deterministic_sample.
        def distribution_class(param):
            if base == "Delta":
                base_dist = Delta(param)
            else:
                base_dist = distributions.LogisticNormal(param, torch.ones_like(param))
            return distributions.Independent(base_dist, 1)

        module = ProbabilisticTensorDictModule(
            in_keys=["param"],
            out_keys=["sample"],
            distribution_class=distribution_class,
            default_interaction_type=InteractionType.DETERMINISTIC,
        )
        td = TensorDict(param=torch.randn(3, 4, 2), batch_size=[3])
        sample = module(td)["sample"]
        expected = distribution_class(td["param"]).base_dist.deterministic_sample
        torch.testing.assert_close(sample, expected)

    # ------------------------------------------------------------------
    # generator argument: Generator object, int seed, and tensordict-key forms
    # ------------------------------------------------------------------
    @staticmethod
    def _make_normal(generator=None, default_interaction_type="random"):
        return ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
            default_interaction_type=default_interaction_type,
            generator=generator,
        )

    def test_generator_module_level_generator(self):
        """Two actors with same-seeded Generators should produce identical samples."""
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(0)
        m1 = self._make_normal(generator=g1)
        m2 = self._make_normal(generator=g2)
        # Set the global RNG to a different state to make sure we're not relying on it
        torch.manual_seed(999)
        s1 = m1(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        s2 = m2(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        assert torch.equal(s1, s2)

    def test_generator_advances_in_place(self):
        """Consecutive calls advance the generator's stream."""
        g = torch.Generator().manual_seed(0)
        m = self._make_normal(generator=g)
        s1 = m(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        s2 = m(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        assert not torch.equal(s1, s2)

    def test_generator_int_seed_equivalent_to_generator(self):
        """Module-level int seed is shorthand for Generator().manual_seed(int)."""
        m_int = self._make_normal(generator=0)
        m_gen = self._make_normal(generator=torch.Generator().manual_seed(0))
        s_int = m_int(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        s_gen = m_gen(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        assert torch.equal(s_int, s_gen)

    def test_generator_isolates_global_rng(self):
        """Sampling with a generator must not advance the global RNG."""
        g = torch.Generator().manual_seed(0)
        m = self._make_normal(generator=g)
        torch.manual_seed(1234)
        before = torch.get_rng_state()
        m(TensorDict(loc=torch.zeros(4)))
        after = torch.get_rng_state()
        assert torch.equal(before, after)

    def test_generator_independent_streams(self):
        """Two same-seeded Generators on two actors stay synchronized only if used
        independently and identically; sampling on one must not perturb the other."""
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(0)
        m1 = self._make_normal(generator=g1)
        m2 = self._make_normal(generator=g2)
        # Sample once on m1, then both — m2's first sample should match m1's first.
        s1a = m1(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        s2a = m2(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        assert torch.equal(s1a, s2a)
        s1b = m1(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        s2b = m2(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        assert torch.equal(s1b, s2b)
        assert not torch.equal(s1a, s1b)

    def test_generator_td_key_generator_form(self):
        """Generator passed via tensordict key works like a module-level Generator."""
        m_key = self._make_normal(generator="rng")
        td = TensorDict(loc=torch.zeros(4))
        td["rng"] = NonTensorData(torch.Generator().manual_seed(0))
        s_key = m_key(td)["sample"].clone()
        m_ref = self._make_normal(generator=torch.Generator().manual_seed(0))
        s_ref = m_ref(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        assert torch.equal(s_key, s_ref)

    def test_generator_td_key_int_seed_writeback(self):
        """Int seed in the tensordict is treated as a stream-key: each call seeds a
        fresh Generator, samples, then writes the next seed back to the tensordict."""
        m = self._make_normal(generator="rng")

        # Two independent runs starting from the same seed must produce the same
        # trajectory.
        def run(initial_seed, n_steps):
            td = TensorDict(loc=torch.zeros(4))
            td["rng"] = NonTensorData(initial_seed)
            samples = []
            for _ in range(n_steps):
                out = m(td)
                samples.append(out["sample"].clone())
            return samples, td["rng"]

        traj_a, last_a = run(42, 3)
        traj_b, last_b = run(42, 3)
        for a, b in zip(traj_a, traj_b):
            assert torch.equal(a, b)
        assert last_a == last_b
        # And consecutive samples in a trajectory differ.
        assert not torch.equal(traj_a[0], traj_a[1])
        assert not torch.equal(traj_a[1], traj_a[2])
        # And different starting seeds give different trajectories.
        traj_c, _ = run(43, 3)
        assert not torch.equal(traj_a[0], traj_c[0])

    def test_generator_td_key_tensor_seed(self):
        """A scalar Tensor seed in the tensordict round-trips as a Tensor."""
        m = self._make_normal(generator="rng")
        td = TensorDict(loc=torch.zeros(4), rng=torch.tensor(7, dtype=torch.int64))
        m(td)
        assert isinstance(td["rng"], torch.Tensor)
        assert td["rng"].dtype == torch.int64
        assert int(td["rng"].item()) != 7  # Got advanced.

    def test_generator_no_overhead_when_unset(self):
        """generator=None must leave the global-RNG path untouched."""
        m = self._make_normal(generator=None)
        torch.manual_seed(0)
        s1 = m(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        torch.manual_seed(0)
        s2 = m(TensorDict(loc=torch.zeros(4)))["sample"].clone()
        assert torch.equal(s1, s2)

    def test_generator_invalid_type(self):
        with pytest.raises(TypeError, match="generator must be"):
            ProbabilisticTensorDictModule(
                in_keys=["loc"],
                out_keys=["sample"],
                distribution_class=Normal,
                distribution_kwargs={"scale": 1.0},
                generator=1.5,
            )

    def test_generator_nested_key(self):
        """The generator NestedKey can be a tuple."""
        m = ProbabilisticTensorDictModule(
            in_keys=["loc"],
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
            default_interaction_type="random",
            generator=("meta", "rng"),
        )
        td = TensorDict(loc=torch.zeros(4))
        td["meta", "rng"] = NonTensorData(123)
        m(td)
        # Stream-key was written back at the nested location.
        assert td["meta", "rng"] != 123

    def test_generator_no_op_for_mode(self):
        """MODE doesn't sample, so the generator is irrelevant and not advanced."""
        g = torch.Generator().manual_seed(0)
        state_before = g.get_state().clone()
        m = self._make_normal(generator=g, default_interaction_type="mode")
        m(TensorDict(loc=torch.tensor([1.0, 2.0, 3.0])))
        assert torch.equal(g.get_state(), state_before)

    def test_generator_categorical_uses_sample_path(self):
        """Discrete distributions go through dist.sample (no rsample); generator must
        still produce reproducible results."""
        g1 = torch.Generator().manual_seed(0)
        g2 = torch.Generator().manual_seed(0)
        m1 = ProbabilisticTensorDictModule(
            in_keys={"logits": "logits"},
            out_keys=["sample"],
            distribution_class=Categorical,
            default_interaction_type="random",
            generator=g1,
        )
        m2 = ProbabilisticTensorDictModule(
            in_keys={"logits": "logits"},
            out_keys=["sample"],
            distribution_class=Categorical,
            default_interaction_type="random",
            generator=g2,
        )
        logits = torch.zeros(8, 4)
        s1 = m1(TensorDict(logits=logits))["sample"].clone()
        s2 = m2(TensorDict(logits=logits))["sample"].clone()
        assert torch.equal(s1, s2)

    def test_generator_in_sequential(self):
        """ProbabilisticActor inheritance: generator threads through Sequential."""
        prob = ProbabilisticTensorDictModule(
            in_keys={"loc": "loc"},
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
            default_interaction_type="random",
            generator=torch.Generator().manual_seed(0),
        )
        seq1 = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda x: x, in_keys=["x"], out_keys=["loc"]),
            prob,
        )
        prob2 = ProbabilisticTensorDictModule(
            in_keys={"loc": "loc"},
            out_keys=["sample"],
            distribution_class=Normal,
            distribution_kwargs={"scale": 1.0},
            default_interaction_type="random",
            generator=torch.Generator().manual_seed(0),
        )
        seq2 = ProbabilisticTensorDictSequential(
            TensorDictModule(lambda x: x, in_keys=["x"], out_keys=["loc"]),
            prob2,
        )
        td = TensorDict(x=torch.zeros(4))
        s1 = seq1(td.copy())["sample"].clone()
        s2 = seq2(td.copy())["sample"].clone()
        assert torch.equal(s1, s2)

    def test_default_out_key_warns(self):
        # Without out_keys, the sample is written at "_"; from 0.17 it is not (#2971)
        td = TensorDict(loc=torch.zeros(3), scale=torch.ones(3))
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            module = ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"], distribution_class=Normal
            )
            module.get_dist(td)
            explicit = ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"], out_keys=["_"], distribution_class=Normal
            )
            assert "_" in explicit(td.clone()).keys()
        assert not [w for w in record if issubclass(w.category, FutureWarning)]
        assert module.out_keys == ["_"]
        with pytest.warns(
            FutureWarning,
            match=r'Starting with TensorDict 0\.17, the default will change to not writing the sample at "_"',
        ) as record:
            out = module(td.clone())
        assert record[0].filename == __file__
        assert "_" in out.keys()
        # The module warns once
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert "_" in module(td.clone()).keys()


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
            assert not torch.allclose(p0, p1), (
                f"Ensemble params were not initialized correctly {p0}, {p1}"
            )

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
        assert out["dork"].shape == torch.Size([2, 5, 1]), (
            "Ensemble forward failed to expand input"
        )
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
            assert out[out_key].shape == torch.Size([4, 5, 1]), (
                f"Ensemble forward failed to expand input for {out_key}"
            )
            same_outputs = torch.isclose(
                out[out_key].repeat(4, 1, 1), out[out_key].repeat_interleave(4, dim=0)
            ).reshape(4, 4, 5, 1)
            mask_out_diags = torch.eye(4).logical_not()
            assert not torch.any(same_outputs[mask_out_diags]), (
                f"Module ensemble outputs should be different for {out_key}"
            )

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
        assert lin.reset_parameters.call_count == 2, (
            f"Reset parameters called {lin.reset_parameters.call_count} times should be 2"
        )

    def test_reset_parameters_recursive_warning(self):
        """Ensure we warn about child modules other than self.module, but not about params_td"""
        module = TensorDictModule(nn.Linear(2, 3), in_keys=["a"], out_keys=["b"])
        ens = EnsembleModule(module, num_copies=2)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="EnsembleModule.reset_parameters_recursive"
            )
            ens.reset_parameters_recursive(ens.params_td)
        ens.extra = nn.Linear(1, 1)
        with pytest.warns(UserWarning, match="only resets parameters of self.module"):
            ens.reset_parameters_recursive(ens.params_td)

    @pytest.mark.parametrize("batch_size", [[4], [2, 3]])
    def test_reset_parameters_recursive_batch_size(self, batch_size):
        """Ensure the returned parameters keep the batch dims of the input"""
        module = TensorDictModule(nn.Linear(2, 3), in_keys=["a"], out_keys=["b"])
        ens = EnsembleModule(module, num_copies=2)
        params = TensorDict.from_module(module).expand(*batch_size).clone().detach()
        out = ens.reset_parameters_recursive(params)
        assert out.batch_size == params.batch_size
        assert (out == params).all()

    @pytest.mark.parametrize("batch_size", [[4], [2, 3], [2, 3, 2]])
    def test_reset_parameters_recursive_requires_grad(self, batch_size):
        """Ensure parameters that require grad are reset in place for any batch dims"""
        module = TensorDictModule(nn.Linear(2, 3), in_keys=["a"], out_keys=["b"])
        ens = EnsembleModule(module, num_copies=2)
        params = TensorDict.from_module(module).expand(*batch_size).clone()
        assert params["module", "weight"].requires_grad
        before = params.detach().clone()
        out = ens.reset_parameters_recursive(params)
        assert out.batch_size == params.batch_size
        assert (out == params).all()
        weight = params["module", "weight"].reshape(-1, 3, 2)
        assert (weight != before["module", "weight"].reshape(-1, 3, 2)).any()
        assert (weight[0] != weight[1:]).any(dim=(-2, -1)).all()


class TestTensorDictParams:
    def test_masked_select(self):
        params = TensorDictParams(
            TensorDict(a=nn.Parameter(torch.zeros(3, 2)), batch_size=[3])
        )
        mask = torch.tensor([True, False, True])
        assert params.masked_select(mask).batch_size == torch.Size([2])
        with pytest.raises(NotImplementedError, match="masked_select"):
            params.masked_select(mask, inplace=True)
        assert params.batch_size == torch.Size([3])

    @pytest.mark.parametrize(
        "name", ["TDPARAM_HANDLED_FUNCTIONS", "implements_for_tdparam"]
    )
    def test_deprecated_module_attributes(self, name):
        with pytest.warns(
            DeprecationWarning,
            match=rf"^tensordict\.nn\.params\.{name} is deprecated and will be "
            rf"removed in TensorDict 0\.17\.$",
        ) as record:
            value = getattr(tensordict.nn.params, name)
        assert record[0].filename == __file__
        assert value is getattr(tensordict.nn.params, f"_{name}")
        assert torch.empty_like in tensordict.nn.params._TDPARAM_HANDLED_FUNCTIONS

    @pytest.mark.parametrize("filter_empty", [False, True])
    @pytest.mark.parametrize("lock", [False, True])
    @pytest.mark.parametrize("nested_params", [False, True])
    def test_capture_module_leaf_contracts(self, filter_empty, lock, nested_params):
        leaf = torch.ones(3, requires_grad=True)
        child = nn.Module()
        child.register_parameter("p", nn.Parameter(torch.ones(3)))
        child.register_buffer("b", torch.zeros(3))
        child.register_buffer("grad", leaf * 2)
        child.register_buffer("absent", None)
        module = nn.Module()
        module.left = module.right = child
        module.empty = nn.Identity()
        if nested_params:
            module.params = TensorDictParams(p=nn.Parameter(torch.ones(3))).lock_()
        captured = TensorDict.from_module(
            module, as_module=True, filter_empty=filter_empty, lock=lock
        )
        assert captured["left", "p"] is captured["right", "p"] is child.p
        if nested_params:
            assert captured["params"] is module.params
            assert module.params.is_locked is lock
        assert captured["left", "b"].data_ptr() == child.b.data_ptr()
        assert isinstance(captured["left", "b"], Buffer)
        assert captured["left", "grad"].grad_fn is child.grad.grad_fn
        captured["left", "grad"].sum().backward()
        torch.testing.assert_close(leaf.grad, torch.full_like(leaf, 2))
        assert ("empty" in captured.keys()) is (not filter_empty)
        assert "absent" not in captured["left"].keys()
        assert captured.is_locked is lock
        if not lock:
            captured["new_buffer"] = torch.ones(3)
            assert isinstance(captured["new_buffer"], Buffer)
            assert not isinstance(captured["new_buffer"], nn.Parameter)

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

    @pytest.mark.parametrize("is_leaf", [is_leaf_nontensor, is_tensor_collection])
    def test_items_values_is_leaf(self, is_leaf):
        td = TensorDict(
            a=torch.ones(3),
            s=NonTensorData("hi"),
            sub=TensorDict(b=torch.ones(3), t=NonTensorData("x")),
        )
        param_td = TensorDictParams(td, no_convert=True)
        # items() and values() list the keys that keys() lists for is_leaf
        keys = list(param_td.keys(True, True, is_leaf=is_leaf))
        if is_leaf is is_leaf_nontensor:
            assert keys == ["a", "s", ("sub", "b"), ("sub", "t")]
        else:
            assert keys == ["s", ("sub", "t"), "sub"]
        expected = [param_td._param_td.get(key) for key in keys]
        items = list(param_td.items(True, True, is_leaf=is_leaf))
        assert [key for key, _ in items] == keys
        assert all(item is exp for (_, item), exp in zip(items, expected))
        values = list(param_td.values(True, True, is_leaf=is_leaf))
        assert len(values) == len(keys)
        assert all(value is exp for value, exp in zip(values, expected))
        assert [key for key, _ in param_td.non_tensor_items()] == ["s"]
        nested = [key for key, _ in param_td.non_tensor_items(include_nested=True)]
        assert nested == ["s", ("sub", "t")]

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

    @pytest.mark.parametrize("no_convert", [False, True])
    @pytest.mark.parametrize(
        "convert",
        [
            pytest.param(lambda p: p.to(torch.float64), id="float64"),
            pytest.param(lambda p: p.to(torch.bfloat16), id="bfloat16"),
            pytest.param(lambda p: p.to("meta"), id="meta"),
            pytest.param(lambda p: p.cpu(), id="cpu"),
            pytest.param(
                lambda p: p.cuda(),
                id="cuda",
                marks=pytest.mark.skipif(
                    not torch.cuda.is_available(), reason="CUDA is not available"
                ),
            ),
        ],
    )
    def test_to_keeps_params_and_buffers(self, convert, no_convert):
        params = TensorDictParams(
            TensorDict(
                weight=nn.Parameter(torch.ones(3)),
                frozen=nn.Parameter(torch.ones(3), requires_grad=False),
                running_mean=Buffer(torch.zeros(3)),
                nested=TensorDict(count=Buffer(torch.zeros((), dtype=torch.long))),
            ),
            no_convert=no_convert,
        )
        converted = convert(params)
        assert isinstance(converted, TensorDictParams)
        assert converted.no_convert is no_convert
        assert dict(converted.named_parameters()).keys() == {"weight", "frozen"}
        assert dict(converted.named_buffers()).keys() == {
            "running_mean",
            "nested.count",
        }
        assert converted["weight"].requires_grad
        assert not converted["frozen"].requires_grad
        assert not converted["running_mean"].requires_grad

    def test_to_keeps_buffer_graph(self):
        # A buffer that tracks gradients stays a buffer in the graph after to()
        weight = nn.Parameter(torch.ones(3))
        params = TensorDictParams(
            TensorDict(expanded=weight.expand(2, 3)), no_convert=True
        )
        converted = params.to(torch.float64)
        assert dict(converted.named_buffers()).keys() == {"expanded"}
        converted["expanded"].sum().backward()
        torch.testing.assert_close(weight.grad, torch.full_like(weight, 2))

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

    def test_load_state_dict_missing_entries(self):
        # A checkpoint that lacks entries for the TensorDictParams prefix must
        # not raise: loading is a no-op for the missing submodule.
        with torch.random.fork_rng(devices=[]):
            params = TensorDict.from_module(nn.Linear(2, 2), as_module=True)

        class MyModule(nn.Module):
            pass

        module = MyModule()
        module.params = params
        before = params.detach().clone()
        # state_dict that has no key under the "params" prefix
        incompatible_keys = module.load_state_dict({}, strict=False)
        assert set(incompatible_keys.missing_keys) == {
            "params.bias",
            "params.weight",
        }
        assert (params == before).all()
        with pytest.raises(RuntimeError, match="Missing key"):
            module.load_state_dict({}, strict=True)

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

    @staticmethod
    def _params(lock=False, no_convert=False):
        return TensorDictParams(
            TensorDict(
                a=torch.randn(4, 3),
                b=torch.arange(4),
                n=TensorDict(c=torch.randn(4, 3), batch_size=[4]),
                batch_size=[4],
            ),
            lock=lock,
            no_convert=no_convert,
        )

    _MASK = torch.tensor([True, False, True, False])

    # results that keep the lock of the wrapped tensordict (D7), and the
    # results that do not
    _LOCK_OPS = {
        "expand": (lambda p: p.expand(2, 4), True),
        "reshape": (lambda p: p.reshape(2, 2), True),
        "repeat_interleave": (lambda p: p.repeat_interleave(2, dim=0), True),
        "where": (lambda p: p.where(TestTensorDictParams._MASK, 0.0), True),
        "split": (lambda p: p.split([1, 3], 0), True),
        "chunk": (lambda p: p.chunk(2, 0), True),
        "repeat": (lambda p: p.repeat(2), True),
        "contiguous": (lambda p: p.contiguous(), False),
        "masked_fill": (
            lambda p: p.masked_fill(TestTensorDictParams._MASK, -1.0),
            False,
        ),
        "unbind": (lambda p: p.unbind(0), False),
        "eq": (lambda p: p == 0, False),
        "ne": (lambda p: p != p.data, False),
        "lt": (lambda p: p < 0, False),
        "any0": (lambda p: (p > 0).any(0), False),
    }

    @pytest.mark.parametrize("no_convert", [False, True])
    @pytest.mark.parametrize("lock", [False, True])
    @pytest.mark.parametrize("op", list(_LOCK_OPS))
    def test_lock_content_result_lock(self, op, lock, no_convert):
        # The results are TensorDicts with the values of the wrapped
        # tensordict; with lock=True, some keep its lock
        fn, keeps_lock = self._LOCK_OPS[op]
        params = self._params(lock=lock, no_convert=no_convert)
        expected = fn(params.data.clone())
        result = fn(params)
        results = result if isinstance(result, tuple) else (result,)
        for r in results:
            assert type(r) is TensorDict
            assert r.is_locked is (lock and keeps_lock)
        if op in ("eq", "ne", "lt", "any0"):
            # boolean results, which hold no gradient: their lock is checked
            # above
            assert all(v.dtype is torch.bool for v in result.values(True, True))
        else:
            grads = {
                key: v.requires_grad for r in results for key, v in r.items(True, True)
            }
            assert grads == {
                "a": not no_convert,
                ("n", "c"): not no_convert,
                "b": False,
            }
        torch.testing.assert_close(
            tuple(r.detach() for r in results),
            expected if isinstance(expected, tuple) else (expected,),
        )

    def test_split_chunk_lock_content(self):
        params = self._params(lock=True)
        for pieces in (params.split([1, 3], 0), params.chunk(2, 0)):
            assert all(type(p) is TensorDict and p.is_locked for p in pieces)
            assert all(
                p["a"].requires_grad and p["n", "c"].requires_grad for p in pieces
            )

    @pytest.mark.parametrize("lock", [False, True])
    def test_split_contiguous_keep_params(self, lock):
        params = self._params(lock=lock)
        contiguous = params.contiguous()
        assert type(contiguous) is TensorDict
        assert isinstance(contiguous["a"], nn.Parameter)
        assert contiguous["a"] is params["a"]
        assert params.contiguous(inplace=True) is params
        pieces = params.split(2, 0)
        assert len(pieces) == 2
        assert all(p["a"]._base is params["a"] for p in pieces)

    @pytest.mark.parametrize("root_locked", [False, True])
    @pytest.mark.parametrize("lock", [False, True])
    def test_inplace_ops_on_root_locked(self, lock, root_locked):
        params = self._params(lock=lock)
        if root_locked:
            params.lock_()
        a = params["a"]
        with torch.no_grad():
            assert params.detach_() is params
            assert params.masked_fill_(self._MASK, -1.0) is params
            assert (params["a"][0] == -1).all()
            assert params.copy_(params.data * 0) is params
            assert (params["a"] == 0).all()
            assert params.update_({"a": torch.ones(4, 3)}) is params
            assert params.zero_() is params
            assert params.share_memory_() is params
        assert params["a"] is a
        assert params.is_shared()
        assert isinstance(params["a"], nn.Parameter)
        assert sorted(dict(params.named_parameters())) == ["a", "n.c"]

    @pytest.mark.parametrize(
        "op",
        [
            lambda p: p.rename_key_("a", "z")._param_td.get("z"),
            lambda p: p.unbind(0)[1]["a"],
            lambda p: list(p)[1]["a"],
            lambda p: p.contiguous()["a"],
            lambda p: p.reshape(2, 2, inplace=True)._param_td.get("a"),
            lambda p: p.repeat_interleave(1, dim=0, inplace=True)._param_td.get("a"),
        ],
        ids=[
            "rename_key_",
            "unbind",
            "iter",
            "contiguous",
            "reshape",
            "repeat_interleave",
        ],
    )
    def test_get_post_hook_not_written(self, op):
        # the operations read the wrapped tensordict without the get hooks,
        # and the stored values stay unchanged
        params = self._params()
        a = params["a"].detach().clone()
        params.register_get_post_hook(lambda self, value: value + 100)
        result = op(params)
        assert (result < 50).all()
        stored = params._param_td.get("z" if "z" in params.keys() else "a")
        torch.testing.assert_close(stored.detach().reshape(4, 3), a)

    @pytest.mark.parametrize("key", ["a", ("n", "c")])
    def test_get_post_hook_get_at(self, key):
        # get_at runs the hooks on the indexed value, as main does; a hook
        # that is not elementwise shows where it runs
        params = self._params()
        params.register_get_post_hook(lambda self, value: value - value.mean())
        value = params._param_td.get(key)[0]
        torch.testing.assert_close(params.get_at(key, 0), value - value.mean())

    @pytest.mark.parametrize("lock", [False, True])
    def test_comparisons_reductions(self, lock):
        params = self._params(lock=lock)
        assert (params == params.data).all()
        assert not (params != params.data).any()
        assert ((params < 0) | (params >= 0)).all()
        assert not ((params < 0) ^ (params < 0)).any()
        assert (params <= params.data).all() and (params >= params.data).all()
        assert not (params > params.data).any()

    @pytest.mark.parametrize("no_convert", [False, True])
    @pytest.mark.parametrize("lock", [False, True])
    def test_update_registers_params(self, lock, no_convert):
        params = self._params(lock=lock, no_convert=no_convert)
        assert params.update({"q": torch.ones(4), ("n", "r"): torch.ones(4, 2)}) is (
            params
        )
        assert params._param_td.is_locked is lock
        if no_convert:
            assert sorted(dict(params.named_parameters())) == []
            assert sorted(dict(params.named_buffers())) == [
                "a",
                "b",
                "n.c",
                "n.r",
                "q",
            ]
        else:
            assert sorted(dict(params.named_parameters())) == ["a", "n.c", "n.r", "q"]
            assert sorted(dict(params.named_buffers())) == ["b"]
        a = params["a"]
        with torch.no_grad():
            assert params.update(params.data * 0, inplace=True) is params
        assert params["a"] is a
        assert (a == 0).all()

    def test_update_ignore_lock_warns(self):
        params = self._params(lock=True)
        with pytest.warns(DeprecationWarning, match="ignore_lock") as record:
            assert params.update({"q": torch.ones(4)}, ignore_lock=True) is params
        assert record[0].filename == __file__
        assert sorted(dict(params.named_parameters())) == ["a", "n.c", "q"]

    @pytest.mark.parametrize("dict_source", [False, True], ids=["td", "dict"])
    @pytest.mark.parametrize("lock", [False, True])
    def test_update_nested_source(self, lock, dict_source):
        # a nested entry of the source goes into the nested tensordict, also
        # with lock=True content, and the module registers the new leaves
        params = self._params(lock=lock)
        source = TensorDict(
            a=torch.ones(4, 3),
            n=TensorDict(c=torch.ones(4, 3), q=torch.ones(4, 2), batch_size=[4]),
            batch_size=[4],
        )
        if dict_source:
            source = source.to_dict()
        assert params.update(source) is params
        assert params._param_td.is_locked is lock
        registered = {**dict(params.named_parameters()), **dict(params.named_buffers())}
        entries = {
            ".".join(key) if isinstance(key, tuple) else key: value
            for key, value in params._param_td.items(True, True)
        }
        assert registered.keys() == entries.keys()
        assert all(registered[key] is entries[key] for key in entries)
        assert (params["n", "c"] == 1).all()

    def test_update_nested_content_locked(self):
        # only a nested tensordict of the content is locked: update() unlocks
        # it as well, and leaves the root unlocked
        params = self._params()
        params._param_td["n"].lock_()
        assert params.update({"n": {"q": torch.ones(4)}}) is params
        assert not params._param_td.is_locked
        assert (params["n", "q"] == 1).all()

    def test_update_root_locked(self):
        params = self._params().lock_()
        with pytest.raises(RuntimeError, match="locked"):
            params.update({"q": torch.ones(4)})
        with pytest.raises(RuntimeError, match="locked"):
            with torch.no_grad():
                params.update(params.data * 0, inplace=True)

    def test_update_update_batch_size(self):
        params = self._params()
        source = TensorDict(
            a=torch.zeros(5, 3),
            b=torch.zeros(5, dtype=torch.long),
            n=TensorDict(c=torch.zeros(5, 3), batch_size=[5]),
            batch_size=[5],
        )
        result = params.update(source, update_batch_size=True)
        assert result is params
        assert params.batch_size == (5,)
        assert isinstance(params["a"], nn.Parameter)
        assert params["a"].shape == (5, 3)

    def test_replace_does_not_modify_arg(self):
        # https://github.com/pytorch/tensordict/issues/2949
        params = TensorDictParams(TensorDict(a=nn.Parameter(torch.ones(3))))
        td = TensorDict(a=torch.zeros(3))
        td_replace = td.replace(params, b=torch.ones(3))
        assert set(params.keys()) == {"a"}
        assert [name for name, _ in params.named_parameters()] == ["a"]
        assert td_replace["a"] is params["a"]
        assert (td_replace["b"] == 1).all()

    @pytest.mark.parametrize("no_convert", [False, True])
    @pytest.mark.parametrize("lock", [False, True])
    def test_setitem(self, lock, no_convert):
        params = self._params(lock=lock, no_convert=no_convert)
        a = params["a"]
        params["q"] = torch.ones(4)
        params["n", "r"] = torch.ones(4, 2)
        params["a"] = torch.zeros(4, 3)
        expected = (
            (["b", "a", "n.c", "n.r", "q"], [])
            if no_convert
            else (["b"], ["a", "n.c", "n.r", "q"])
        )
        assert sorted(dict(params.named_buffers())) == sorted(expected[0])
        assert sorted(dict(params.named_parameters())) == sorted(expected[1])
        assert params["a"] is not a
        assert (params["a"] == 0).all()
        a = params["a"]
        with torch.no_grad():
            params[0] = params.data[1].clone() + 1
            params[1:3] = 0
            params[3] = {"a": torch.full((3,), 5.0)}
        assert params["a"] is a
        assert (a[0] == 1).all() and (a[1:3] == 0).all() and (a[3] == 5).all()
        assert params._param_td.is_locked is lock
        params.lock_()
        with pytest.raises(RuntimeError, match="locked"):
            params["z"] = torch.ones(4)
        with pytest.raises(RuntimeError, match="locked"):
            with torch.no_grad():
                params[0] = 0
        assert "z" not in params.keys()

    def test_setitem_self(self):
        # #2741: these wrote params into the tensordict that it wraps
        params = self._params()
        keys = set(params.keys(True))
        msg = r"Cannot set a tensordict inside itself \(key 'self'\)"
        with pytest.raises(ValueError, match=msg):
            params["self"] = params
        with pytest.raises(ValueError, match=msg):
            params.set("self", params)
        with pytest.raises(ValueError, match=msg):
            params.set("self", item=params)
        with pytest.raises(ValueError, match=msg):
            params.update({"self": params})
        assert set(params.keys(True)) == keys

    @pytest.mark.parametrize("no_convert", [False, True])
    @pytest.mark.parametrize("lock", [False, True])
    def test_del_(self, lock, no_convert):
        params = self._params(lock=lock, no_convert=no_convert)
        assert params.del_("a") is params
        assert params.del_(("n", "c")) is params
        assert list(params.keys(True)) == ["b", "n"]
        assert dict(params.named_parameters()) == {}
        assert list(dict(params.named_buffers())) == ["b"]
        assert params._param_td.is_locked is lock
        with pytest.raises(KeyError):
            params.del_("missing")
        with pytest.raises(KeyError):
            params.del_(())
        params.lock_()
        with pytest.raises(RuntimeError, match="locked"):
            params.del_("b")
        assert "b" in params.keys()

    @pytest.mark.parametrize("lock", [False, True])
    def test_iter_popitem_rename_key_get_at(self, lock):
        params = self._params(lock=lock)
        assert [td["b"] for td in params] == list(params["b"].unbind(0))
        assert all(type(td) is TensorDict for td in params)
        assert params.get_at(("n", "c"), 0).shape == (3,)
        assert params.get_at(("n", "c"), 0).requires_grad
        key, value = params.popitem()
        assert key == "n"
        assert type(value) is TensorDict
        assert sorted(dict(params.named_parameters())) == ["a"]
        assert params.rename_key_("a", "z") is params
        assert list(params.keys()) == ["b", "z"]
        assert sorted(dict(params.named_parameters())) == ["z"]
        assert params._param_td.is_locked is lock
        params.lock_()
        with pytest.raises(RuntimeError, match="locked"):
            params.rename_key_("z", "y")
        with pytest.raises(RuntimeError, match="locked"):
            params.popitem()

    @pytest.mark.parametrize("lock", [False, True])
    def test_new_unsafe_base_keywords(self, lock):
        # the keywords that the base code passes; a dict gives a TensorDict
        params = self._params()
        result = type(params)._new_unsafe(
            source={"a": torch.zeros(4)},
            batch_size=torch.Size([4]),
            device=None,
            names=None,
            lock=lock,
            is_shared=False,
            is_memmap=False,
        )
        assert type(result) is TensorDict
        assert result.batch_size == (4,)
        assert result.is_locked is lock
        with pytest.raises(TypeError, match="both parameters and source"):
            type(params)._new_unsafe(
                {"a": torch.zeros(4)}, source={"a": torch.zeros(4)}
            )

    @staticmethod
    def _params_non_tensor(lock):
        params = TensorDictParams(
            TensorDict(
                a=torch.ones(3), s=NonTensorData("old", batch_size=[3]), batch_size=[3]
            ),
            lock=lock,
        )
        params.set("ns", NonTensorStack("x", "y", "z"))
        return params

    @staticmethod
    def _non_tensor_source():
        return TensorDict(
            a=torch.zeros(3),
            s=NonTensorData("new", batch_size=[3]),
            ns=NonTensorStack("p", "q", "r"),
            batch_size=[3],
        )

    _NON_TENSOR_OPS = {
        "update_": lambda td, src: td.update_(src),
        "update_select": lambda td, src: td.update_(src.select("s")),
        "set_": lambda td, src: td.set_("s", src.get("s")),
        "copy_": lambda td, src: td.copy_(src),
    }

    @pytest.mark.parametrize("lock", [False, True])
    @pytest.mark.parametrize("op", list(_NON_TENSOR_OPS))
    def test_inplace_writes_non_tensor(self, op, lock):
        # #2263: the in-place methods write the non-tensor leaves, as on the
        # wrapped tensordict
        fn = self._NON_TENSOR_OPS[op]
        params = self._params_non_tensor(lock)
        with torch.no_grad():
            assert fn(params, self._non_tensor_source()) is params
        dense = self._params_non_tensor(lock)._param_td.clone()
        with torch.no_grad():
            fn(dense, self._non_tensor_source())
        assert dense["s"] == "new"
        assert params["s"] == "new"
        assert params.get("ns").tolist() == dense.get("ns").tolist()
        torch.testing.assert_close(params["a"].data, dense["a"])
        assert isinstance(params["a"], nn.Parameter)

    _NON_TENSOR_AT_OPS = {
        "set_at_": lambda td, src: td.set_at_("s", "new", 1),
        "set_at_slice": lambda td, src: td.set_at_("s", "new", slice(0, 2)),
        "set_at_stack": lambda td, src: td.set_at_("ns", "new", 1),
        "update_at_": lambda td, src: td.update_at_(src[1], 1),
    }

    @pytest.mark.parametrize("lock", [False, True])
    @pytest.mark.parametrize("op", list(_NON_TENSOR_AT_OPS))
    def test_inplace_at_writes_non_tensor(self, op, lock):
        # #2740: set_at_ and update_at_ write the non-tensor leaves at the
        # index, as on the wrapped tensordict
        fn = self._NON_TENSOR_AT_OPS[op]
        params = self._params_non_tensor(lock)
        with torch.no_grad():
            assert fn(params, self._non_tensor_source()) is params
        dense = self._params_non_tensor(lock)._param_td.clone()
        fn(dense, self._non_tensor_source())
        old = self._params_non_tensor(lock)
        written = [dense.get(key).tolist() for key in ("s", "ns")]
        assert written != [old.get(key).tolist() for key in ("s", "ns")]
        assert [params.get(key).tolist() for key in ("s", "ns")] == written
        torch.testing.assert_close(params["a"].data, dense["a"])
        assert isinstance(params["a"], nn.Parameter)
        assert params._param_td.is_locked is old._param_td.is_locked

    @pytest.mark.parametrize("op", ["set_at_", "update_at_"])
    def test_inplace_at_writes_non_tensor_nested_params(self, op):
        # #2740: a string in a TensorDictParams nested in a locked one
        params = TensorDictParams(
            TensorDict(
                a=torch.ones(3),
                m=TensorDictParams(
                    TensorDict(s="old", c=torch.ones(3), batch_size=[3])
                ),
                batch_size=[3],
            )
        ).lock_()
        src = TensorDict(
            a=torch.zeros(3),
            m=TensorDict(s="new", c=torch.zeros(3), batch_size=[3]),
            batch_size=[3],
        )
        with torch.no_grad():
            if op == "set_at_":
                params.set_at_(("m", "s"), "new", 1)
            else:
                params.update_at_(src[1], 1)
        assert params.get(("m", "s")).tolist() == ["old", "new", "old"]
        if op == "update_at_":
            assert params.get(("m", "c")).tolist() == [1.0, 0.0, 1.0]
        assert params.is_locked
        assert params.get("m").is_locked

    def test_data_update_non_tensor(self):
        # .data builds new non-tensor leaves, so plain td.data.update_() does
        # not write them
        td = TensorDict(
            a=torch.ones(3), s=NonTensorData("old", batch_size=[3]), batch_size=[3]
        )
        td.data.update_(self._non_tensor_source().select("a", "s"))
        assert td["s"] == "old"
        assert (td["a"] == 0).all()


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

        if mode in (True,):
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

    @pytest.mark.parametrize("constructor", ["init", "from_distributions"])
    def test_event_shape(self, constructor):
        # The sample's batch size is batch_shape: no dims are left for events.
        if constructor == "init":
            params = TensorDict(
                {
                    "cont": {"loc": torch.zeros(3, 4), "scale": torch.ones(3, 4)},
                    ("nested", "disc"): {"logits": torch.zeros(3, 10)},
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
        else:
            dist = CompositeDistribution.from_distributions(
                TensorDict(batch_size=[3]),
                {
                    "cont": distributions.Normal(torch.zeros(3, 4), torch.ones(3, 4)),
                    ("nested", "disc"): distributions.Categorical(
                        logits=torch.zeros(3, 10)
                    ),
                },
            )
        assert dist.event_shape == torch.Size([])
        assert dist.batch_shape + dist.event_shape == dist.sample().batch_size
        assert dist.sample((2,)).batch_size == torch.Size([2, 3])

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

    def _dist_with_name_map(self, constructor, dist_name, name_map):
        # Params are stored under the unraveled name, which is what both
        # constructors look up. The distribution key itself may still be a
        # nested tuple that unravels to a different object.
        key = unravel_key(dist_name)
        if constructor == "init":
            params = TensorDict(
                {key: {"loc": torch.zeros(3), "scale": torch.ones(3)}},
                [3],
            )
            return CompositeDistribution(
                params,
                distribution_map={dist_name: distributions.Normal},
                name_map=name_map,
            )
        params = TensorDict({}, batch_size=[3])
        return CompositeDistribution.from_distributions(
            params,
            {dist_name: distributions.Normal(torch.zeros(3), torch.ones(3))},
            name_map=name_map,
        )

    @pytest.mark.parametrize("constructor", ["init", "from_distributions"])
    def test_name_map_tuple_key(self, constructor):
        # ("a",) unravels to "a". Looking the raw key up in name_map misses.
        dist_name = ("a",)
        assert unravel_key(dist_name) != dist_name
        dist = self._dist_with_name_map(constructor, dist_name, {dist_name: "b"})
        assert list(dist.dists) == ["b"], list(dist.dists)
        assert "b" in dist.sample().keys()
        assert "a" not in dist.sample().keys()

    @pytest.mark.parametrize("constructor", ["init", "from_distributions"])
    def test_name_map_nested_tuple_key(self, constructor):
        # (("nested",), "disc") unravels to ("nested", "disc"), not itself.
        dist_name = (("nested",), "disc")
        assert unravel_key(dist_name) != dist_name
        dist = self._dist_with_name_map(constructor, dist_name, {dist_name: "b"})
        assert list(dist.dists) == ["b"], list(dist.dists)
        sample = dist.sample()
        assert "b" in sample.keys()
        assert ("nested", "disc") not in sample.keys(True, True)

    @pytest.mark.parametrize("constructor", ["init", "from_distributions"])
    def test_name_map_string_key(self, constructor):
        dist = self._dist_with_name_map(constructor, "a", {"a": "b"})
        assert list(dist.dists) == ["b"], list(dist.dists)
        assert "b" in dist.sample().keys()
        assert "a" not in dist.sample().keys()

    @pytest.mark.parametrize("constructor", ["init", "from_distributions"])
    def test_name_map_unknown_key_ignored(self, constructor):
        # A name_map shared by several modules may name other distributions.
        dist = self._dist_with_name_map(
            constructor, "a", {"a": "b", "nope": "x", ("nope",): "y"}
        )
        assert list(dist.dists) == ["b"], list(dist.dists)

    @pytest.mark.parametrize("dist_name", ["a", ("a",), (("nested",), "disc")])
    @pytest.mark.parametrize("raw_key", [True, False])
    def test_extra_kwargs_unraveled_key(self, dist_name, raw_key):
        # extra_kwargs is looked up with the unraveled name, then with the key
        # as written in distribution_map.
        key = unravel_key(dist_name)
        params = TensorDict(
            {key: {"loc": torch.zeros(3), "scale": torch.ones(3)}},
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={dist_name: distributions.Normal},
            extra_kwargs={dist_name if raw_key else key: {"validate_args": False}},
        )
        assert dist[key]._validate_args is False

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

    @pytest.mark.parametrize("method", ["sample", "rsample"])
    @pytest.mark.parametrize("shape", [[], [4], [4, 2]])
    def test_sample_list_shape(self, method, shape):
        # A list works as sample shape, as it does for torch distributions
        params = TensorDict(
            {
                "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
                ("nested", "cont"): {
                    "loc": torch.randn(3, 4),
                    "scale": torch.rand(3, 4),
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
        torch.manual_seed(0)
        sample = getattr(dist, method)(shape)
        torch.manual_seed(0)
        expected = getattr(dist, method)(tuple(shape))
        assert sample.batch_size == torch.Size(shape) + params.batch_size
        assert (sample == expected).all()

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

    @pytest.mark.parametrize("aggregate", [False, True])
    @pytest.mark.parametrize("method", ["sample", "rsample"])
    def test_sample_and_log_prob(self, aggregate, method):
        # The joint methods give the log-probability in the layout of log_prob
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
                ("nested", "disc"): distributions.RelaxedOneHotCategorical,
            },
            extra_kwargs={("nested", "disc"): {"temperature": torch.tensor(1.0)}},
        )
        with set_composite_lp_aggregate(aggregate):
            torch.manual_seed(0)
            sample, lp = getattr(dist, f"{method}_and_log_prob")((2,))
            torch.manual_seed(0)
            expected_sample = getattr(dist, method)((2,))
            expected_lp = dist.log_prob(expected_sample)
        assert (sample == expected_sample).all()
        assert sample.batch_size == (2, 3)
        if aggregate:
            assert isinstance(lp, torch.Tensor)
        else:
            assert isinstance(lp, TensorDict)
            assert set(lp.keys(True, True)) == set(expected_lp.keys(True, True))
        torch.testing.assert_close(lp, expected_lp)

    @pytest.mark.parametrize("aggregate", [False, True])
    def test_prob_module_joint_sample_and_log_prob(self, aggregate):
        # Components with a joint method are scored with it, the others with
        # sample then log_prob
        loc = torch.tensor([[20.0, -20.0]])
        scale = torch.full_like(loc, 0.1)
        logits = torch.randn(1, 5)
        with set_composite_lp_aggregate(aggregate):
            module = ProbabilisticTensorDictModule(
                in_keys=["params"],
                out_keys=["act", ("nested", "disc")],
                distribution_class=CompositeDistribution,
                distribution_kwargs={
                    "distribution_map": {
                        "act": _TanhNormalJoint,
                        ("nested", "disc"): distributions.Categorical,
                    }
                },
                default_interaction_type="random",
                return_log_prob=True,
            )
            td = TensorDict(
                params={
                    "act": {"loc": loc, "scale": scale},
                    ("nested", "disc"): {"logits": logits},
                },
                batch_size=[1],
            )
            torch.manual_seed(0)
            td = module(td)
        torch.manual_seed(0)
        act, act_lp = _TanhNormalJoint(loc, scale).sample_and_log_prob()
        disc_dist = distributions.Categorical(logits=logits)
        disc = disc_dist.sample()
        disc_lp = disc_dist.log_prob(disc)
        assert act_lp.isfinite().all()
        assert (td["act"] == act).all()
        assert (td["nested", "disc"] == disc).all()
        torch.testing.assert_close(td["act_log_prob"], act_lp)
        torch.testing.assert_close(td["nested", "disc_log_prob"], disc_lp)
        if aggregate:
            torch.testing.assert_close(td["sample_log_prob"], act_lp.sum(-1) + disc_lp)
        else:
            assert "sample_log_prob" not in td

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

    @pytest.mark.parametrize("lp_aggregate", [True, False])
    def test_entropy_truncated_normal(self, lp_aggregate):
        params = TensorDict(
            x=TensorDict(
                loc=torch.zeros(3),
                scale=torch.ones(3),
                a=-torch.ones(3),
                b=torch.ones(3),
                batch_size=[3],
            ),
            batch_size=[3],
        )
        dist = CompositeDistribution(params, distribution_map={"x": TruncatedNormal})
        expected = TruncatedNormal(
            torch.zeros(3), torch.ones(3), -torch.ones(3), torch.ones(3)
        ).entropy()
        with set_composite_lp_aggregate(lp_aggregate):
            ent = dist.entropy()
        if lp_aggregate:
            assert isinstance(ent, torch.Tensor)
            torch.testing.assert_close(ent, expected)
        else:
            assert isinstance(ent, TensorDict)
            torch.testing.assert_close(ent["x_entropy"], expected)

    @pytest.mark.parametrize("lp_aggregate", [True, False])
    @pytest.mark.parametrize("aggregate_probabilities", [True, False])
    def test_entropy_aggregate_probabilities_deprecated(
        self, lp_aggregate, aggregate_probabilities
    ):
        params = TensorDict(
            cont={"loc": torch.zeros(3), "scale": torch.ones(3)},
            disc={"logits": torch.zeros(3, 4)},
            batch_size=[3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={"cont": distributions.Normal, "disc": Categorical},
        )
        with set_composite_lp_aggregate(lp_aggregate):
            expected = dist.entropy()
            with pytest.warns(
                DeprecationWarning,
                match=r"^CompositeDistribution\.entropy\(aggregate_probabilities=\.\.\.\), "
                r"which has no effect, is deprecated and will be removed in "
                r"TensorDict 0\.17\. Use set_composite_lp_aggregate instead\.$",
            ) as record:
                ent = dist.entropy(aggregate_probabilities=aggregate_probabilities)
        assert record[0].filename == __file__
        assert type(ent) is type(expected)
        assert (ent == expected).all()

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

    def test_icdf_without_cdf_entries(self):
        # Without <sample_name>_cdf entries, icdf computes the cdf of the samples
        params = TensorDict(
            {
                "cont": {"loc": torch.randn(3, 4), "scale": torch.rand(3, 4)},
                ("nested", "cont"): {
                    "loc": torch.randn(3, 4),
                    "scale": torch.rand(3, 4),
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
        sample = dist.sample((4,))
        expected = dist.icdf(dist.cdf(sample.clone()))
        result = dist.icdf(sample.clone())
        for key in ("cont_icdf", ("nested", "cont_icdf")):
            torch.testing.assert_close(result[key], expected[key])
        with pytest.raises(
            KeyError,
            match=r"Neither \('nested', 'cont'\) nor \('nested', 'cont_cdf'\)",
        ):
            dist.icdf(sample.exclude(("nested", "cont")))

    def test_icdf(self):
        # The cdf-icdf round trip loses float32 precision far in the tails, and
        # about 1% of unseeded draws land there and fail assert_close.
        torch.manual_seed(0)
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

    def test_deterministic_sample_independent(self):
        # Independent has no deterministic_sample: the base's one is used, as in
        # ProbabilisticTensorDictModule.
        params = TensorDict(
            {
                "delta": {"param": torch.randn(3, 4, 2)},
                ("nested", "logistic"): {
                    "loc": torch.randn(3, 4, 2),
                    "scale": torch.ones(3, 4, 2),
                },
            },
            [3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "delta": lambda param: distributions.Independent(Delta(param), 1),
                ("nested", "logistic"): lambda loc, scale: distributions.Independent(
                    distributions.LogisticNormal(loc, scale), 1
                ),
            },
        )
        sample = dist.deterministic_sample
        assert sample.batch_size == params.batch_size
        torch.testing.assert_close(sample["delta"], params["delta", "param"])
        logistic = distributions.LogisticNormal(**params["nested", "logistic"])
        torch.testing.assert_close(
            sample["nested", "logistic"], logistic.deterministic_sample
        )

    @pytest.mark.parametrize("registered", [True, False])
    def test_deterministic_sample_no_fallback(self, registered):
        # TransformedDistribution has no deterministic_sample, mean or mode: the
        # error names it, and no warning announces a fallback (#2083). The
        # register maps it to MEAN; an unregistered subclass falls back on its
        # support.
        class _Transformed(distributions.TransformedDistribution):
            pass

        dist_cls = distributions.TransformedDistribution if registered else _Transformed
        params = TensorDict(
            x=TensorDict(loc=torch.zeros(3, 2), scale=torch.ones(3, 2)),
            batch_size=[3],
        )
        dist = CompositeDistribution(
            params,
            distribution_map={
                "x": lambda loc, scale: dist_cls(
                    distributions.Normal(loc, scale), [distributions.TanhTransform()]
                )
            },
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", message="deterministic_sample wasn't found"
            )
            with pytest.raises(
                NotImplementedError,
                match=f"{dist_cls.__name__}'> has no deterministic_sample",
            ):
                dist.deterministic_sample

        # A fallback that works still warns.
        class _Normal(distributions.Normal):
            pass

        dist = CompositeDistribution(params, distribution_map={"x": _Normal})
        with pytest.warns(UserWarning, match="is falling back on mean"):
            sample = dist.deterministic_sample
        torch.testing.assert_close(sample["x"], params["x", "loc"])

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
        "interaction", [InteractionType.MODE, InteractionType.RANDOM]
    )
    @pytest.mark.parametrize("nested_log_prob", [False, True])
    @set_composite_lp_aggregate(False)
    def test_prob_module_custom_log_prob_keys(self, interaction, nested_log_prob):
        params = TensorDict(
            {
                "a": {"loc": torch.zeros(3, 2), "scale": torch.ones(3, 2)},
                ("nested", "b"): {
                    "loc": torch.ones(3, 2),
                    "scale": torch.ones(3, 2),
                },
            },
            batch_size=[3],
        )
        sample_keys = ["a", ("nested", "b")]
        log_prob_keys = (
            [("scores", "a"), ("scores", "b")] if nested_log_prob else ["lp_a", "lp_b"]
        )
        module = ProbabilisticTensorDictModule(
            in_keys=["params"],
            distribution_class=CompositeDistribution,
            distribution_kwargs={
                "distribution_map": dict.fromkeys(sample_keys, distributions.Normal)
            },
            default_interaction_type=interaction,
            return_log_prob=True,
            log_prob_keys=log_prob_keys,
        )
        td = TensorDict(params=params, batch_size=[3])
        sample = module(td)
        assert all(key in sample for key in module.out_keys)
        assert "a_log_prob" not in sample
        assert ("nested", "b_log_prob") not in sample
        expected = module.log_prob(sample)
        for key in log_prob_keys:
            torch.testing.assert_close(sample[key], expected[key])

        # Dispatch must return the same declared outputs, including both scores.
        a, b, lp_a, lp_b = module(params=params)
        for key, value, log_prob in zip(sample_keys, (a, b), (lp_a, lp_b)):
            dist = distributions.Normal(**params[key])
            torch.testing.assert_close(log_prob, dist.log_prob(value))

    @pytest.mark.parametrize(
        "interaction", [InteractionType.MODE, InteractionType.RANDOM]
    )
    @pytest.mark.parametrize("partial", [False, True])
    @set_composite_lp_aggregate(False)
    def test_prob_module_swapped_log_prob_keys(self, interaction, partial):
        # Each custom key is the default key of the other sample.
        params = TensorDict(
            a={"loc": torch.zeros(3), "scale": torch.ones(3)},
            b={"loc": torch.zeros(3), "scale": torch.full((3,), 10.0)},
            batch_size=[3],
        )
        distribution_kwargs = {
            "distribution_map": {"a": distributions.Normal, "b": distributions.Normal}
        }
        if partial:
            # As TorchRL builds composites: the module does not see the map
            distribution_class = functools.partial(
                CompositeDistribution, **distribution_kwargs
            )
            distribution_kwargs = {}
        else:
            distribution_class = CompositeDistribution
        module = ProbabilisticTensorDictModule(
            in_keys=["params"],
            out_keys=["a", "b"],
            distribution_class=distribution_class,
            distribution_kwargs=distribution_kwargs,
            default_interaction_type=interaction,
            return_log_prob=True,
            log_prob_keys=["b_log_prob", "a_log_prob"],
        )
        sample = module(TensorDict(params=params, batch_size=[3]))
        lp = module.log_prob(sample)
        for out_key, lp_key in (("a", "b_log_prob"), ("b", "a_log_prob")):
            expected = distributions.Normal(**params[out_key]).log_prob(sample[out_key])
            torch.testing.assert_close(sample[lp_key], expected)
            torch.testing.assert_close(lp[lp_key], expected)
        assert set(lp.keys()) == {"a_log_prob", "b_log_prob"}

    @pytest.mark.parametrize("aggregate", [True, False])
    @pytest.mark.parametrize("partial", [False, True])
    def test_prob_module_log_prob_keys_next_to_samples(self, aggregate, partial):
        # The samples and their log-probabilities share the "act" sub-tensordict.
        distribution_kwargs = {
            "distribution_map": {"a": distributions.Normal, "b": distributions.Normal},
            "name_map": {"a": ("act", "a"), "b": ("act", "b")},
        }
        if partial:
            distribution_class = functools.partial(
                CompositeDistribution, **distribution_kwargs
            )
            distribution_kwargs = {}
        else:
            distribution_class = CompositeDistribution
        if aggregate:
            # One out key holds both samples
            out_keys = ["act"]
            lp_kwargs = {"log_prob_key": ("act", "sample_log_prob")}
        else:
            out_keys = [("act", "a"), ("act", "b")]
            lp_kwargs = {"log_prob_keys": [("act", "lp_a"), ("act", "lp_b")]}
        with set_composite_lp_aggregate(aggregate):
            module = ProbabilisticTensorDictModule(
                in_keys=["params"],
                out_keys=out_keys,
                distribution_class=distribution_class,
                distribution_kwargs=distribution_kwargs,
                default_interaction_type=InteractionType.MODE,
                return_log_prob=True,
                **lp_kwargs,
            )
            params = TensorDict(
                {k: {"loc": torch.zeros(3), "scale": torch.ones(3)} for k in "ab"},
                batch_size=[3],
            )
            sample = module(TensorDict(params=params, batch_size=[3]))
            lp = module.log_prob(sample)
        expected = {
            key: distributions.Normal(**params[key]).log_prob(sample["act", key])
            for key in "ab"
        }
        if aggregate:
            total = expected["a"] + expected["b"]
            torch.testing.assert_close(sample["act", "sample_log_prob"], total)
            torch.testing.assert_close(lp, total)
        else:
            for key in "ab":
                torch.testing.assert_close(sample["act", f"lp_{key}"], expected[key])
                torch.testing.assert_close(lp["act", f"lp_{key}"], expected[key])
        torch.testing.assert_close(sample["act", "a"], torch.zeros(3))

    @pytest.mark.parametrize("aggregate", [True, False])
    @pytest.mark.parametrize("where", ["equal", "inside", "parent"])
    def test_prob_module_log_prob_key_clashes_with_sample(self, aggregate, where):
        # A log-probability written there would overwrite the sample ("act", "a").
        clash = {"equal": ("act", "a"), "inside": ("act", "a", "lp"), "parent": "act"}[
            where
        ]
        distribution_kwargs = {
            "distribution_map": {"a": distributions.Normal, "b": distributions.Normal},
            "name_map": {"a": ("act", "a"), "b": ("act", "b")},
        }
        if aggregate:
            out_keys = ["act"]
            lp_kwargs = {"log_prob_key": clash}
        else:
            out_keys = [("act", "a"), ("act", "b")]
            lp_kwargs = {"log_prob_keys": [clash, ("act", "lp_b")]}

        def factory(*args, **kwargs):
            return CompositeDistribution(*args, **distribution_kwargs, **kwargs)

        with set_composite_lp_aggregate(aggregate):
            kwargs = {"in_keys": ["params"], "out_keys": out_keys, **lp_kwargs}
            with pytest.raises(ValueError, match=re.escape(repr(clash))):
                ProbabilisticTensorDictModule(
                    distribution_class=CompositeDistribution,
                    distribution_kwargs=distribution_kwargs,
                    return_log_prob=True,
                    **kwargs,
                )
            # Without return_log_prob, nothing is written at the key
            ProbabilisticTensorDictModule(
                distribution_class=CompositeDistribution,
                distribution_kwargs=distribution_kwargs,
                **kwargs,
            )
            # The sample keys of a partial or a factory are not known
            for distribution_class in (
                functools.partial(CompositeDistribution, **distribution_kwargs),
                factory,
            ):
                ProbabilisticTensorDictModule(
                    distribution_class=distribution_class,
                    return_log_prob=True,
                    **kwargs,
                )

    @set_composite_lp_aggregate(False)
    def test_prob_module_default_log_prob_key_inside_sample(self):
        # The default key ("dirich", "categ_log_prob") lies inside the sample
        # "dirich", and is not checked.
        kwargs = {
            "in_keys": ["params"],
            "out_keys": [("dirich", "categ")],
            "distribution_class": CompositeDistribution,
            "distribution_kwargs": {
                "distribution_map": {
                    "dirich": distributions.Dirichlet,
                    "categ": distributions.Categorical,
                }
            },
            "return_log_prob": True,
        }
        module = ProbabilisticTensorDictModule(**kwargs)
        assert module.log_prob_keys == [("dirich", "categ_log_prob")]
        ProbabilisticTensorDictModule(
            log_prob_keys=[("dirich", "categ_log_prob")], **kwargs
        )

    @pytest.mark.parametrize("lp_key", ["action", ("action", "lp")])
    def test_prob_module_single_dist_log_prob_key_clashes_with_sample(self, lp_key):
        with pytest.raises(ValueError, match=re.escape(repr(lp_key))):
            ProbabilisticTensorDictModule(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=distributions.Normal,
                return_log_prob=True,
                log_prob_key=lp_key,
            )

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

    # log_prob_keys require composite_lp_aggregate() to be False
    @pytest.mark.parametrize(
        "aggregate,custom_keys", [(True, False), (False, False), (False, True)]
    )
    @pytest.mark.parametrize("sample_keys", [["a"], ["a", "b"]])
    @pytest.mark.parametrize("partial", [False, True])
    def test_prob_module_seq_scores_preset_samples(
        self, aggregate, custom_keys, sample_keys, partial
    ):
        # The prefix writes the samples, so the last module only scores them.
        distribution_kwargs = {
            "distribution_map": dict.fromkeys(sample_keys, distributions.Normal)
        }
        if partial:
            distribution_class = functools.partial(
                CompositeDistribution, **distribution_kwargs
            )
            distribution_kwargs = {}
        else:
            distribution_class = CompositeDistribution
        with set_composite_lp_aggregate(aggregate):
            prob = ProbabilisticTensorDictModule(
                in_keys=["params"],
                out_keys=sample_keys,
                distribution_class=distribution_class,
                distribution_kwargs=distribution_kwargs,
                return_log_prob=True,
                log_prob_keys=[f"lp_{key}" for key in sample_keys]
                if custom_keys
                else None,
            )
            out_keys = [
                ("params", key, p) for key in sample_keys for p in ("loc", "scale")
            ]
            prefix = TensorDictModule(
                lambda x: (x, x.exp()) * len(sample_keys) + (x + 1,) * len(sample_keys),
                in_keys=["x"],
                out_keys=out_keys + sample_keys,
            )
            seq = ProbabilisticTensorDictSequential(prefix, prob)
            assert seq._requires_sample is False
            td = TensorDict(x=torch.zeros(3), batch_size=[3])
            expected = prob.log_prob(prefix(td.clone()))
            out = seq(td)
            for key in sample_keys:
                torch.testing.assert_close(out[key], torch.ones(3))
            if aggregate:
                torch.testing.assert_close(out[prob.log_prob_key], expected)
            else:
                assert set(expected.keys()) == set(prob.log_prob_keys)
                for key in prob.log_prob_keys:
                    torch.testing.assert_close(out[key], expected[key])

    @pytest.mark.parametrize("aggregate", [True, False])
    def test_prob_module_scores_preset_samples_under_out_key(self, aggregate):
        # The samples of both distributions are given under the one out key
        # "act", which _requires_sample=False scores as a whole.
        with set_composite_lp_aggregate(aggregate):
            prob = ProbabilisticTensorDictModule(
                in_keys=["params"],
                out_keys=["act"],
                distribution_class=CompositeDistribution,
                distribution_kwargs={
                    "distribution_map": {
                        "a": distributions.Normal,
                        "b": distributions.Normal,
                    }
                },
                return_log_prob=True,
            )
            params = TensorDict(
                a={"loc": torch.zeros(3), "scale": torch.ones(3)},
                b={"loc": torch.ones(3), "scale": torch.full((3,), 2.0)},
                batch_size=[3],
            )
            act = TensorDict(a=torch.ones(3), b=torch.zeros(3), batch_size=[3])
            out = prob(
                TensorDict(params=params, act=act, batch_size=[3]),
                _requires_sample=False,
            )
            expected = {
                key: distributions.Normal(**params[key]).log_prob(act[key])
                for key in "ab"
            }
            if aggregate:
                torch.testing.assert_close(
                    out["sample_log_prob"], expected["a"] + expected["b"]
                )
            else:
                # The entries of the composite land under "act_log_prob"
                for key in "ab":
                    torch.testing.assert_close(
                        out["act_log_prob", f"{key}_log_prob"], expected[key]
                    )
            assert (out["act"] == act).all()

    def test_partial_name_map_default_out_keys_warns(self):
        # Without out_keys, a component that name_map leaves out is written but
        # is not in out_keys; from 0.17 it is (#2972)
        def make(name_map, out_keys=None):
            return ProbabilisticTensorDictModule(
                in_keys=["params"],
                out_keys=out_keys,
                distribution_class=CompositeDistribution,
                distribution_kwargs={
                    "distribution_map": {"a": Normal, "b": Normal},
                    "name_map": name_map,
                },
                return_log_prob=True,
            )

        with set_composite_lp_aggregate(False):
            with warnings.catch_warnings(record=True) as record:
                warnings.simplefilter("always")
                make({"a": "x", "b": "y"})
                make({("a",): "x", "b": "y"})
                make(None)
                assert make({"a": "x"}, out_keys=["x", "b"]).out_keys == [
                    "x",
                    "b",
                    "x_log_prob",
                    "b_log_prob",
                ]
            assert not [w for w in record if issubclass(w.category, FutureWarning)]
            with pytest.warns(
                FutureWarning,
                match=r"Starting with TensorDict 0\.17, the default will change to "
                r"out_keys that include them",
            ) as record:
                module = make({"a": "x"})
            assert record[0].filename == __file__
            assert module.out_keys == ["x", "x_log_prob"]


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

    @pytest.mark.parametrize("loc_shape", [(), (4,), (3, 4)])
    def test_add_scale_scalar_shape(self, loc_shape):
        # The default scale is 0-dim and broadcasts to any loc
        module_normal = AddStateIndependentNormalScale(init_value=1.0)
        assert module_normal.state_independent_scale.shape == ()
        loc = torch.randn(loc_shape)
        out_loc, scale = module_normal(loc)
        assert out_loc is loc
        assert scale.shape == loc_shape
        torch.testing.assert_close(scale, torch.ones(loc_shape).exp())

    @pytest.mark.parametrize("loc_shape", [(5,), (3, 5), (4,)])
    def test_add_scale_shape_mismatch(self, loc_shape):
        module_normal = AddStateIndependentNormalScale((2, 4))
        with pytest.raises(RuntimeError, match="Last dimensions of loc"):
            module_normal(torch.randn(loc_shape))


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

    def test_plain_tensor_to_module_preserves_module_state_by_default(self, as_module):
        module = nn.Linear(4, 2)
        module.weight.requires_grad_(False)
        params = TensorDict.from_module(module, as_module=as_module).data.detach()
        state_dict_keys = set(module.state_dict())

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            params.to_module(module)

        assert set(module.state_dict()) == state_dict_keys
        assert isinstance(module.weight, nn.Parameter)
        assert isinstance(module.bias, nn.Parameter)
        assert module.weight.requires_grad is False
        assert module.bias.requires_grad is True
        torch.testing.assert_close(module.weight, params["weight"])
        torch.testing.assert_close(module.bias, params["bias"])

    def test_to_module_rejects_preserve_module_state_none(self, as_module):
        module = nn.Linear(4, 2)
        weight = module.weight
        params = TensorDict.from_module(module, as_module=as_module).data.clone()

        with pytest.raises(TypeError, match="preserve_module_state=None"):
            params.to_module(module, preserve_module_state=None)
        assert module.weight is weight

    @pytest.mark.parametrize("parameter_source", [False, True])
    def test_to_module_can_preserve_module_state(self, as_module, parameter_source):
        module = nn.Linear(4, 2)
        module.weight.requires_grad_(False)
        params = TensorDict.from_module(module, as_module=as_module).data.detach()
        if parameter_source:
            params["weight"] = nn.Parameter(params["weight"], requires_grad=True)
        state_dict_keys = set(module.state_dict())

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            params.to_module(module, preserve_module_state=True)

        assert set(module.state_dict()) == state_dict_keys
        assert isinstance(module.weight, nn.Parameter)
        assert isinstance(module.bias, nn.Parameter)
        assert module.weight.requires_grad is False
        assert module.bias.requires_grad is True
        torch.testing.assert_close(module.weight, params["weight"])
        torch.testing.assert_close(module.bias, params["bias"])

    @pytest.mark.parametrize("source", ["computed", "leaf"])
    @pytest.mark.parametrize("custom_setattr", [False, True])
    def test_preserve_module_state_keeps_gradients(
        self, as_module, source, custom_setattr
    ):
        class MyLinear(nn.Linear):
            def __setattr__(self, key, value):
                return super().__setattr__(key, value)

        module = MyLinear(4, 2) if custom_setattr else nn.Linear(4, 2)
        weight, bias = module.weight, module.bias
        params = TensorDict.from_module(module, as_module=as_module)
        if source == "computed":
            params = params.apply(lambda p: p * 2)
            grad_targets = [weight, bias]
        else:
            params = params.data.clone().requires_grad_()
            grad_targets = [params["weight"], params["bias"]]
        state_dict_keys = set(module.state_dict())

        with params.to_module(module):
            assert set(module.state_dict()) == state_dict_keys
            assert set(dict(module.named_parameters())) == {"weight", "bias"}
            y = module(torch.randn(3, 4))
        y.sum().backward()

        assert module.weight is weight
        assert module.bias is bias
        for target in grad_targets:
            assert target.grad is not None

    def test_to_module_context_manager_restores_on_error(self, as_module):
        # The module gets its original parameters back also when the body raises
        module = nn.Linear(4, 2)
        weight, bias = module.weight, module.bias
        params = TensorDict.from_module(module, as_module=as_module)
        zeros = params.detach().clone().zero_()
        with pytest.raises(ValueError, match="error in the body"):
            with zeros.to_module(module) as swap:
                assert (module.weight == 0).all()
                raise ValueError("error in the body")
        assert module.weight is weight
        assert module.bias is bias
        assert not swap._last_op_queue

    def test_to_module_context_manager_keeps_source_leaves(self, as_module):
        # The with block does not leave the nn.Parameter wrappers it builds
        # for the module in the tensordict (#3115)
        module = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
        module[0].bias.requires_grad_(False)
        params = TensorDict.from_module(module, as_module=as_module)
        params = params.detach().clone()
        if not as_module:
            params["0", "bias"] = nn.Parameter(params["0", "bias"])
            params["1", "weight"].requires_grad_()
            params["1", "running_mean"] = nn.Parameter(
                params["1", "running_mean"], requires_grad=False
            )
        leaves = dict(params.items(include_nested=True, leaves_only=True))
        types = {key: type(leaf) for key, leaf in leaves.items()}
        requires_grad = {key: leaf.requires_grad for key, leaf in leaves.items()}
        with params.to_module(module):
            assert isinstance(module[0].weight, nn.Parameter)
            assert module[0].bias.requires_grad is False
            assert "running_mean" in module[1]._buffers
            module(torch.randn(2, 3))
        for key, leaf in params.items(include_nested=True, leaves_only=True):
            assert leaf is leaves[key], key
            assert type(leaf) is types[key], key
            assert leaf.requires_grad is requires_grad[key], key

        # A value that the block assigns is written back, even when it views
        # the storage of the leaf it replaces
        linear = nn.Linear(4, 4)
        flat = torch.randn(4, 8)
        params = TensorDict(weight=flat[:, :4], bias=torch.zeros(4))
        with params.to_module(linear):
            linear.weight = nn.Parameter(flat[:, 4:])
        assert (params["weight"] == flat[:, 4:]).all()
        with params.to_module(linear):
            linear.weight = nn.Parameter(params["weight"].t())
        assert (params["weight"] == flat[:, 4:].t()).all()

        # The wrapper of a leaf that requires grad is written back, so the
        # gradient that lands on it is not lost
        module = nn.Module()
        module.register_buffer("scale", torch.ones(3))
        params = TensorDict(scale=nn.Parameter(torch.ones(3)))
        with params.to_module(module):
            (module.scale * 2).sum().backward()
        assert params["scale"].grad is not None

        # The parameters that a lazy module materializes in the block still
        # land in the tensordict
        lazy = nn.LazyLinear(4)
        params = TensorDict.from_module(lazy, as_module=as_module)
        with params.to_module(lazy):
            lazy(torch.randn(2, 3))
        assert params["weight"].shape == (4, 3)
        assert not isinstance(params["weight"], nn.UninitializedParameter)

    def test_to_module_context_manager_shared_submodule(self, as_module):
        # A submodule used twice gets the tensors of its first entry; the
        # with block leaves the other entry with its own tensors (#3115)
        lin = nn.Linear(3, 3)
        module = nn.Sequential(lin, nn.ReLU(), lin)
        params = TensorDict.from_module(module, as_module=as_module)
        params = params.detach().clone()
        params["0"] = params["0"].apply(lambda x: torch.full_like(x, 1.0))
        params["2"] = params["2"].apply(lambda x: torch.full_like(x, 2.0))
        if not as_module:
            params["0", "weight"] = nn.Parameter(params["0", "weight"])
            params["2", "weight"] = nn.Parameter(params["2", "weight"])
        leaves = dict(params.items(include_nested=True, leaves_only=True))
        with params.to_module(module):
            assert (lin.weight == 1).all()
            module(torch.randn(2, 3)).sum().backward()
        for key, leaf in params.items(include_nested=True, leaves_only=True):
            assert leaf is leaves[key], key
        assert (params["2"] == 2).all()
        if not as_module:
            assert params["0", "weight"].grad is not None
            assert params["2", "weight"].grad is None

        # A value that the block assigns to the submodule is written back to
        # both entries
        weight = nn.Parameter(torch.zeros(3, 3))
        with params.to_module(module):
            lin.weight = weight
        assert params["0", "weight"] is weight
        assert params["2", "weight"] is weight

    def test_to_module_context_manager_keeps_source_locked(self, as_module):
        # The exit of the block unlocks the source only to write back the
        # tensors that the block replaced, so the source keeps its lock and
        # its shared-memory state, and a sub-tensordict of a locked
        # tensordict can be the source
        module = nn.Sequential(nn.Linear(4, 3), nn.Linear(3, 2))
        weight = module[0].weight
        params = TensorDict.from_module(module, as_module=as_module)
        params = params.data.clone().share_memory_()
        assert params.is_locked
        with params["0"].to_module(module[0], preserve_module_state=False):
            assert module[0].weight is params["0", "weight"]
        assert module[0].weight is weight
        with params.to_module(module, preserve_module_state=False):
            assert module[1].weight is params["1", "weight"]
        assert params.is_locked
        assert params._is_shared
        # memo=None is accepted by the public to_module
        with params.to_module(module, memo=None, preserve_module_state=False):
            assert module[1].weight is params["1", "weight"]
        assert module[0].weight is weight

    @pytest.mark.parametrize("custom_setattr", [False, True])
    def test_to_module_from_load_memmap(self, as_module, custom_setattr, tmpdir):
        class MyLinear(nn.Linear):
            def __setattr__(self, key, value):
                return super().__setattr__(key, value)

        module = MyLinear(4, 2) if custom_setattr else nn.Linear(4, 2)
        module.weight.requires_grad_(False)
        TensorDict.from_module(module, as_module=as_module).memmap(tmpdir)
        params = TensorDict.load_memmap(tmpdir)
        state_dict_keys = set(module.state_dict())

        params.to_module(module)

        assert set(module.state_dict()) == state_dict_keys
        assert type(module.weight) is nn.Parameter
        assert type(module.bias) is nn.Parameter
        assert module.weight.requires_grad is False
        assert module.bias.requires_grad is True
        # the parameters share the memory-mapped storage
        assert module.weight.data_ptr() == params["weight"].data_ptr()
        assert module.bias.data_ptr() == params["bias"].data_ptr()

    def test_to_module_keeps_parameter_when_conversion_fails(self, as_module):
        class DetachToTensor(torch.Tensor):
            # detach() returns a plain Tensor, which nn.Parameter rejects
            __torch_function__ = torch._C._disabled_torch_function_impl

        module = nn.Linear(4, 2)
        weight = module.weight
        # TensorDictParams stores a plain Tensor, so only a TensorDict keeps
        # the subclass up to to_module
        params = TensorDict(weight=torch.zeros(2, 4).as_subclass(DetachToTensor))

        with pytest.raises(RuntimeError, match="Parameter"):
            params.to_module(module)

        assert module.weight is weight
        assert set(dict(module.named_parameters())) == {"weight", "bias"}

    def test_plain_tensor_to_module_can_keep_current_behavior(self, as_module):
        module = nn.Linear(4, 2)
        params = TensorDict.from_module(module, as_module=as_module).data.detach()

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            params.to_module(module, preserve_module_state=False)

        assert "weight" not in module.state_dict()
        assert "bias" not in module.state_dict()
        assert not isinstance(module.weight, nn.Parameter)
        assert not isinstance(module.bias, nn.Parameter)

    @pytest.mark.parametrize("return_swap", [False, True])
    def test_to_module_inplace_preserves_state(self, as_module, return_swap):
        module = nn.Linear(4, 2)
        params = TensorDict.from_module(module, as_module=as_module)
        params = params.data.clone()
        params.zero_()
        state_dict_keys = set(module.state_dict())
        weight = module.weight
        bias = module.bias

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            params.to_module(module, inplace=True, return_swap=return_swap)

        assert set(module.state_dict()) == state_dict_keys
        assert module.weight is weight
        assert module.bias is bias
        assert (module.weight == 0).all()
        assert (module.bias == 0).all()

    def test_preserve_module_state_with_custom_setattr(self, as_module):
        class MyLinear(nn.Linear):
            def __setattr__(self, key, value):
                return super().__setattr__(key, value)

        module = MyLinear(4, 2)
        params = TensorDict.from_module(module, as_module=as_module).data.detach()
        state_dict_keys = set(module.state_dict())

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            params.to_module(module, preserve_module_state=True)

        assert set(module.state_dict()) == state_dict_keys
        assert isinstance(module.weight, nn.Parameter)
        assert isinstance(module.bias, nn.Parameter)

    @pytest.mark.parametrize("tagged_tensor", [False, True])
    def test_preserve_module_state_keeps_buffers(self, as_module, tagged_tensor):
        module = nn.Module()
        module.register_buffer("buffer", torch.ones(3))
        value = nn.Parameter(torch.zeros(3))
        if tagged_tensor:
            value = torch.zeros(3)
            value._is_param = True
        params = TensorDict({"buffer": value}, [])

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            params.to_module(module, preserve_module_state=True)

        assert "buffer" in module.state_dict()
        assert "buffer" in dict(module.named_buffers())
        assert "buffer" not in dict(module.named_parameters())
        assert not isinstance(module.buffer, nn.Parameter)
        torch.testing.assert_close(module.buffer, torch.zeros(3))

    @pytest.mark.parametrize("with_hook", [False, True])
    def test_partial_swap_registration_order(self, with_hook, as_module):
        module = nn.Module()
        module.a = nn.Parameter(torch.zeros(3))
        module.b = nn.Parameter(torch.zeros(3))
        module.register_buffer("c", torch.zeros(3))
        module.register_buffer("d", torch.zeros(3))
        original, original_buffer = module.a, module.c
        replacement = nn.Parameter(torch.ones(3))
        params = TensorDict(a=replacement, c=torch.ones(3))
        if as_module:
            params = TensorDictParams(params, no_convert=True)
        seen = []

        def hook(module, name, value):
            seen.append(value)

        handle = (
            torch.nn.modules.module.register_module_parameter_registration_hook(hook)
            if with_hook
            else contextlib.nullcontext()
        )
        with handle:
            with params.to_module(module):
                assert module.a is replacement
                assert list(module._parameters) == ["b", "a"]
                assert list(module._buffers) == ["d", "c"]
                module.a.sum().backward()
        assert module.a is original and module.c is original_buffer
        torch.testing.assert_close(replacement.grad, torch.ones(3))
        if with_hook:
            assert len(seen) == 2 and seen[0] is replacement and seen[1] is original

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
        params0.to_module(module, inplace=inplace, preserve_module_state=False)
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
        params.to_module(module, inplace=inplace, preserve_module_state=False)
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
        with params0.to_module(module, inplace=inplace, preserve_module_state=False):
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
        with params_meta.to_module(module, preserve_module_state=False):
            module_meta = copy.deepcopy(module)
        y1 = module(*x)
        with params.to_module(module_meta, preserve_module_state=False):
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
        with params.detach().to_module(linear, preserve_module_state=False):
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

    def test_method_instances(self):
        # Each instance runs the method with its own self, and the decorated
        # method does not keep the first instance alive.
        class A:
            def __init__(self, k):
                self.k = k

            @as_tensordict_module(in_keys="c", out_keys="d")
            def func(self, c):
                return c * self.k

        a, b = A(2), A(10)
        assert a.func(TensorDict(c=1))["d"] == 2
        assert b.func(TensorDict(c=1))["d"] == 10
        assert a.func(c=torch.tensor(1)) == 2
        assert b.func(c=torch.tensor(1)) == 10
        a_ref = weakref.ref(a)
        del a
        assert a_ref() is None

    @pytest.mark.parametrize("in_keys", ["c", ["c"]])
    @pytest.mark.parametrize("out_keys", ["d", ["d"]])
    def test_free_func(self, in_keys, out_keys):
        @as_tensordict_module(in_keys=in_keys, out_keys=out_keys)
        def func(c):
            return c + 1

        assert func(TensorDict(c=0))["d"] == 1


class InputTensorClass(TensorClass):
    """Test input TensorClass with two tensor fields."""

    a: torch.Tensor
    b: torch.Tensor


class AddDiffResult(TensorClass):
    """Test output TensorClass for add/diff operations."""

    added: torch.Tensor
    substracted: torch.Tensor


class OutputTensorClass(TensorClass):
    """Test output TensorClass with nested structure."""

    input: InputTensorClass
    result: AddDiffResult


class AddDiffModule(TensorClassModuleBase[InputTensorClass, AddDiffResult]):
    """Test module that adds and subtracts two tensors."""

    def forward(self, x: InputTensorClass) -> AddDiffResult:
        return AddDiffResult(
            added=(x.a + x.b), substracted=(x.a - x.b), batch_size=x.batch_size
        )


# String annotations are what `from __future__ import annotations` produces.
class StringAnnotationInput(TensorClass):
    """Test input TensorClass with string and optional annotations."""

    a: "torch.Tensor"
    b: "torch.Tensor"
    mask: torch.Tensor | None = None


class StringAnnotationOutput(TensorClass):
    """Test output TensorClass with string annotations for nested fields."""

    input: "StringAnnotationInput"
    result: "AddDiffResult"


class StringAnnotationModule(
    TensorClassModuleBase[StringAnnotationInput, StringAnnotationOutput]
):
    """Test module whose input and output TensorClasses use string annotations."""

    def forward(self, x: StringAnnotationInput) -> StringAnnotationOutput:
        result = AddDiffResult(
            added=x.a + x.b, substracted=x.a - x.b, batch_size=x.batch_size
        )
        return StringAnnotationOutput(input=x, result=result, batch_size=x.batch_size)


class TestTensorClassModule(TensorClassModuleBase[InputTensorClass, OutputTensorClass]):
    """Test module with nested TensorClass output."""

    def __init__(self) -> None:
        super().__init__()
        self.add_diff = AddDiffModule()

    def forward(self, x: InputTensorClass) -> OutputTensorClass:
        return OutputTensorClass(
            input=x, result=self.add_diff(x), batch_size=x.batch_size
        )


class NestedInputModule(TensorClassModuleBase[OutputTensorClass, AddDiffResult]):
    """Test module that reads the fields of a nested TensorClass input."""

    def forward(self, x: OutputTensorClass) -> AddDiffResult:
        return AddDiffResult(
            added=x.input.a + x.result.added,
            substracted=x.input.b - x.result.substracted,
            batch_size=x.batch_size,
        )


class TestTensorClassModuleForward:
    """Tests for TensorClassModule forward pass."""

    def test_forward(self) -> None:
        """Test basic forward pass with TensorClass input."""
        module = TestTensorClassModule()
        value = InputTensorClass(a=10, b=5, batch_size=[])
        output = module.forward(value)
        assert isinstance(output, OutputTensorClass)
        assert output.result.added == 15
        assert output.result.substracted == 5

    def test_td_forward(self) -> None:
        """Test forward pass with TensorDict input via wrapper."""
        td_module = TestTensorClassModule().as_td_module()
        value = InputTensorClass(a=10, b=5, batch_size=[])
        td_output = td_module(value.to_tensordict())
        assert td_output["result", "added"] == 15
        assert td_output["result", "substracted"] == 5

    def test_td_forward_nested_input(self) -> None:
        """Test that the wrapper passes nested inputs to forward as TensorClasses."""
        td_module = NestedInputModule().as_td_module()
        value = TestTensorClassModule()(InputTensorClass(a=10, b=5, batch_size=[]))
        td_output = td_module(value.to_tensordict())
        assert td_output["added"] == 25
        assert td_output["substracted"] == 0

    @pytest.mark.parametrize("training", [True, False])
    def test_wrapper_training_mode(self, training: bool) -> None:
        """Test that the wrapper starts in the training mode of the module."""
        module = TestTensorClassModule().train(training)
        td_module = module.as_td_module()
        assert td_module.training is training
        assert module.training is training
        td_module = td_module.select_out_keys(("result", "added"))
        assert td_module.training is training

    def test_wrapper_keys(self) -> None:
        """Test that wrapper correctly extracts in_keys and out_keys."""
        module = TestTensorClassModule()
        td_module = module.as_td_module()
        assert set(td_module.in_keys) == {"a", "b"}
        assert set(td_module.out_keys) == {
            ("input", "a"),
            ("input", "b"),
            ("result", "added"),
            ("result", "substracted"),
        }

    def test_wrapper_keys_from_string_and_optional_annotations(self) -> None:
        """Test that wrapper keys are read from string and optional annotations."""
        td_module = StringAnnotationModule().as_td_module()
        assert set(td_module.in_keys) == {"a", "b", "mask"}
        assert set(td_module.out_keys) == {
            ("input", "a"),
            ("input", "b"),
            ("input", "mask"),
            ("result", "added"),
            ("result", "substracted"),
        }
        value = StringAnnotationInput(a=10, b=5, batch_size=[])
        td_output = td_module(value.to_tensordict())
        assert td_output["result", "added"] == 15
        assert td_output["result", "substracted"] == 5

    def test_td_forward_extra_keys(self) -> None:
        """Test that the wrapper ignores the entries that are not input fields."""
        td_module = TestTensorClassModule().as_td_module()
        td = TensorDict(a=10, b=5, other=0)
        td_output = td_module(td)
        assert td_output["result", "added"] == 15
        assert td_output["result", "substracted"] == 5
        assert set(td_output["input"].keys()) == {"a", "b"}
        assert set(td.keys()) == {"a", "b", "other"}

    def test_td_forward_in_sequence(self) -> None:
        """Test that the wrapper runs after another module in a TensorDictSequential."""
        seq = TensorDictSequential(
            TensorDictModule(lambda a: a + 1, in_keys=["a"], out_keys=["c"]),
            TestTensorClassModule().as_td_module(),
        )
        td_output = seq(TensorDict(a=10, b=5))
        assert td_output["result", "added"] == 15
        assert td_output["result", "substracted"] == 5
        assert set(td_output["input"].keys()) == {"a", "b"}

    def test_td_forward_extra_args_warn(self) -> None:
        """Test that the wrapper warns about the arguments it ignores (#2979)."""
        td_module = TestTensorClassModule().as_td_module()
        td = TensorDict(a=10, b=5)
        out = TensorDict()
        with pytest.warns(FutureWarning, match="'tensordict_out'.*TensorDict 0.17"):
            td_output = td_module(td, tensordict_out=out)
        assert td_output is not out
        assert out.is_empty()
        with pytest.warns(FutureWarning, match="positional argument.*, 'foo'"):
            td_output = td_module(td, 1, foo=3)
        assert td_output["result", "added"] == 15
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            assert td_module(td)["result", "added"] == 15
            assert td_module(a=10, b=5)[0] == 10

    def test_td_forward_nested_input_extra_keys(self) -> None:
        """Test that the wrapper ignores the entries next to nested input fields."""
        value = TestTensorClassModule()(InputTensorClass(a=10, b=5, batch_size=[]))
        td = value.to_tensordict()
        td["other"] = torch.zeros(())
        td_output = NestedInputModule().as_td_module()(td)
        assert td_output["added"] == 25
        assert td_output["substracted"] == 0


@pytest.mark.skipif(not _has_onnx, reason="ONNX is not available")
class TestONNXExport:
    """Tests for ONNX export functionality."""

    def test_onnx_export_module(self, tmp_path: pathlib.Path) -> None:
        """Test ONNX export of TensorClassModule."""
        tc_module = TestTensorClassModule()
        tc_module.eval()
        tc_input = InputTensorClass(
            a=torch.tensor([10.0], dtype=torch.float),
            b=torch.tensor([5.0], dtype=torch.float),
            batch_size=[1],
        )
        torch_input = tc_input.to_tensordict().to_dict()

        td_module = tc_module.as_td_module().select_out_keys(
            ("result", "added"), ("result", "substracted")
        )
        output_names = [
            v if isinstance(v, str) else "_".join(v) for v in td_module.out_keys
        ]

        onnx_program = torch.onnx.export(
            model=td_module, kwargs=torch_input, output_names=output_names, dynamo=True
        )

        path = tmp_path / "file.onnx"
        onnx_program.save(str(path))

        import onnxruntime

        ort_session = onnxruntime.InferenceSession(
            path, providers=["CPUExecutionProvider"]
        )

        def to_numpy(tensor):
            return (
                tensor.detach().cpu().numpy()
                if tensor.requires_grad
                else tensor.cpu().numpy()
            )

        output_names = [output.name for output in ort_session.get_outputs()]

        onnxruntime_input = {k: to_numpy(v) for k, v in torch_input.items()}

        onnxruntime_outputs = ort_session.run(None, onnxruntime_input)
        onnxruntime_output_dict = dict(zip(output_names, onnxruntime_outputs))

        tc_outputs = tc_module(tc_input)

        torch.testing.assert_close(
            torch.as_tensor(onnxruntime_output_dict["result_added"]),
            tc_outputs.result.added,
        )
        torch.testing.assert_close(
            torch.as_tensor(onnxruntime_output_dict["result_substracted"]),
            tc_outputs.result.substracted,
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_non_tensorclass_conversion_error(self) -> None:
        """Test that conversion to TensorDictModule fails for non-TensorClass types."""

        class BadModule(TensorClassModuleBase[torch.Tensor, torch.Tensor]):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x + 1

        module = BadModule()
        with pytest.raises(
            ValueError,
            match="Only TensorClassModuleBase implementations with both input and output type as TensorClass",
        ):
            module.as_td_module()

    @pytest.mark.parametrize("subscript", ["none", "typevars"])
    def test_missing_type_arguments_conversion_error(self, subscript) -> None:
        """Test that conversion fails with a ValueError when the types are not set."""
        if subscript == "none":
            base = TensorClassModuleBase
        else:
            base = TensorClassModuleBase[InputClass, OutputClass]

        class UnsetModule(base):
            def forward(self, x):
                return x

        module = UnsetModule()
        with pytest.raises(ValueError, match="input/output types not set"):
            module.as_td_module()

    def test_field_named_fields(self) -> None:
        """Test that a field named ``fields`` is read as a key."""

        class FieldsInput(TensorClass):
            fields: torch.Tensor

        class FieldsModule(TensorClassModuleBase[FieldsInput, AddDiffResult]):
            def forward(self, x: FieldsInput) -> AddDiffResult:
                return AddDiffResult(
                    added=x.fields + 1, substracted=x.fields - 1, batch_size=[]
                )

        td_module = FieldsModule().as_td_module()
        assert td_module.in_keys == ["fields"]
        assert td_module(TensorDict(fields=torch.ones(())))["added"] == 2

    def test_batch_size_preservation(self) -> None:
        """Test that batch size is correctly preserved through forward pass."""
        module = AddDiffModule()
        batch_sizes = [[], [3], [2, 3], [1, 2, 3]]

        for batch_size in batch_sizes:
            if batch_size:
                input_tc = InputTensorClass(
                    a=torch.randn(*batch_size),
                    b=torch.randn(*batch_size),
                    batch_size=batch_size,
                )
            else:
                input_tc = InputTensorClass(
                    a=torch.randn(()),
                    b=torch.randn(()),
                    batch_size=batch_size,
                )
            output = module(input_tc)
            assert output.batch_size == torch.Size(batch_size)
            assert output.added.shape == torch.Size(batch_size)
            assert output.substracted.shape == torch.Size(batch_size)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
