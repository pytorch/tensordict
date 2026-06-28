# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import math
import operator
import os
import platform
import sys
import warnings
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from packaging import version
from tensordict import (
    capture_non_tensor_stack,
    lazy_stack,
    LazyStackedTensorDict,
    set_capture_non_tensor_stack,
    tensorclass,
    TensorClass,
    TensorDict,
    UnbatchedTensor,
)
from tensordict.tensorclass import MetaData, NonTensorData, NonTensorStack
from tensordict.utils import _pass_through, is_non_tensor, LinkedList, set_list_to_stack
from torch import multiprocessing as mp

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        DummyPicklableClass,
        get_available_devices,
        is_npu_available,
    )
else:
    IS_FB = False
    from _utils_internal import (
        DummyPicklableClass,
        get_available_devices,
        is_npu_available,
    )


_has_streaming = importlib.util.find_spec("streaming", None) is not None

try:
    import h5py  # noqa

    _has_h5py = True
except ImportError:
    _has_h5py = False
TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

_has_onnx = importlib.util.find_spec("onnxruntime", None) is not None

_v2_5 = TORCH_VERSION >= version.parse("2.5.0")
PYTORCH_TEST_FBCODE = os.getenv("PYTORCH_TEST_FBCODE")

_IS_OSX = platform.system() == "Darwin"
_IS_WINDOWS = sys.platform == "win32"

TD_BATCH_SIZE = 4
HAS_NESTED_TENSOR = (
    getattr(torch, "_nested_compute_contiguous_strides_offsets", None) is not None
)

# Capture all warnings
pytestmark = [
    pytest.mark.filterwarnings("error"),
    pytest.mark.filterwarnings(
        "ignore:There is a performance drop because we have not yet implemented the batching rule"
    ),
    pytest.mark.filterwarnings(
        "ignore:A destination should be provided when cloning a PersistentTensorDict"
    ),
    pytest.mark.filterwarnings(
        "ignore:Replacing an array with another one is inefficient"
    ),
    pytest.mark.filterwarnings(
        "ignore:Indexing an h5py.Dataset object with a boolean mask that needs broadcasting does not work directly"
    ),
    pytest.mark.filterwarnings(
        "ignore:The PyTorch API of nested tensors is in prototype"
    ),
    pytest.mark.filterwarnings(
        "ignore:Lazy modules are a new feature under heavy development so changes to the API or functionality"
    ),
    pytest.mark.filterwarnings(
        "ignore:The content of the stacked NonTensorData objects matched in value but not identity"
    ),
    pytest.mark.filterwarnings(
        "ignore:No PYTORCH_KERNEL_CACHE_PATH or HOME environment variable set"
    ),
    pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning"),
]

mp_ctx = "spawn"
cur_device = "cpu"
npu_device_count = 0
if torch.cuda.is_available():
    cur_device = "cuda"
elif is_npu_available():
    cur_device = "npu"
    npu_device_count = torch.npu.device_count()


class TestNonTensorData:
    @tensorclass
    class SomeTensorClass:
        a: str
        b: torch.Tensor

    @pytest.fixture
    def non_tensor_data(self):
        return TensorDict(
            {
                "1": 1,
                "nested": {
                    "int": NonTensorData(3, batch_size=[]),
                    "str": NonTensorData("a string!", batch_size=[]),
                    "bool": NonTensorData(True, batch_size=[]),
                },
            },
            batch_size=[],
        )

    @set_capture_non_tensor_stack(False)
    def test_consolidate_nested(self):
        import pickle

        td = TensorDict(
            a=TensorDict(b=self.SomeTensorClass(a="a string!", b=torch.randn(10))),
            c=TensorDict(d=NonTensorData("another string!")),
        )
        td = lazy_stack([td.clone(), td.clone()])
        td = lazy_stack([td.clone(), td.clone()], -1)

        tdc = td.consolidate()

        assert (tdc == td).all()

        tdr = pickle.loads(pickle.dumps(td))
        assert (tdr == td).all()

        tdcr = pickle.loads(pickle.dumps(tdc))
        assert (tdcr == td).all()

    def test_comparison(self, non_tensor_data):
        non_tensor_data = non_tensor_data.exclude(("nested", "str"))
        assert (non_tensor_data | non_tensor_data).get_non_tensor(("nested", "bool"))
        assert not (non_tensor_data ^ non_tensor_data).get_non_tensor(
            ("nested", "bool")
        )
        assert (non_tensor_data == non_tensor_data).get_non_tensor(("nested", "bool"))
        assert not (non_tensor_data != non_tensor_data).get_non_tensor(
            ("nested", "bool")
        )

    def test_expand(self):
        d = NonTensorData(0, batch_size=(3,))
        d_expand = d.expand((2, 3))
        assert d_expand.shape == (2, 3)
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, NonTensorStack)
        d = NonTensorData(0, batch_size=(1,))
        d_expand = d.expand((2, 3))
        assert d_expand.shape == (2, 3)
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, NonTensorStack)

    def test_expand_nested(self):
        d = TensorDict(foo=NonTensorData(0, batch_size=(3,)), batch_size=(3,))
        d_expand = d.expand((2, 3)).get("foo")
        assert d_expand.shape == (2, 3)
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        d = TensorDict(foo=NonTensorData(0, batch_size=(1,)), batch_size=(1,))
        d_expand = d.expand((2, 3)).get("foo")
        assert d_expand.shape == (2, 3)
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]

    @pytest.mark.parametrize(
        "in_out", [(None, None), (0, -1), (None, 0), (0, None), (0, 0)]
    )
    def test_flatten_empty(self, in_out):
        ntd = NonTensorData(0, batch_size=())
        assert ntd.shape == ()
        ntdflat = ntd.flatten(in_out[0], in_out[1])
        assert ntdflat.shape == (1,)
        assert ntdflat.tolist() == [0]

    def test_from_list(self):
        nd = NonTensorStack.from_list(
            [[True, "b", torch.randn(())], ["another", 0, NonTensorData("final")]]
        )
        assert isinstance(nd, NonTensorStack)
        assert nd.shape == (2, 3)
        assert nd[0, 0].data
        assert nd[0, 1].data == "b"
        assert isinstance(nd[0, 2].data, torch.Tensor)
        assert nd[1, 0].data == "another"
        assert nd[1, 1].data == 0
        assert nd[1, 2].data == "final"

    @set_list_to_stack(True)
    def test_linked_list(self):
        td = TensorDict(a=["foo", "bar"], batch_size=(2,))
        assert isinstance(td["a"], LinkedList)
        td["a"][0] = "baz"
        assert td["a"] == ["baz", "bar"]
        td["a"][:2] = ["baz", "qux"]
        assert td["a"] == ["baz", "qux"]

        result = td["a"]
        assert result._td() is td.get("a")
        result.extend(["baz", "qux"])
        assert result._td() is None

    def test_new_empty_nontensorstack(self):
        td = TensorDict(a=NonTensorStack("a", "b").unsqueeze(-1), batch_size=(2,))
        assert isinstance(td.new_empty((4,), empty_lazy=True).get("a"), NonTensorStack)
        assert isinstance(td.new_empty((1,), empty_lazy=True).get("a"), NonTensorStack)

    def test_new_empty_setitem(self):
        td = TensorDict(
            a=TensorDict(
                b=NonTensorStack("a", "b", "c").unsqueeze(-1), batch_size=(3,)
            ),
            batch_size=(3,),
        ).to_lazystack()
        tdz = td.new_zeros((4,), empty_lazy=True)
        tdz[torch.tensor([True, True, False, True])] = td
        assert tdz.get(("a", "b")).tolist() == [["a"], ["b"], ["a"], ["c"]]

    def test_new_empty_setitem_2(self):
        td = TensorDict(
            a=TensorDict(b=NonTensorStack("a"), batch_size=(1,)), batch_size=(1,)
        ).to_lazystack()
        tdz = td.new_zeros((4,), empty_lazy=True)
        td["a", "b"] = "new"
        tdz[torch.tensor([False, False, False, True])] = td
        assert tdz["a", "b"][-1] == "new"

    def test_non_tensor_call(self):
        td0 = TensorDict({"a": 0, "b": 0})
        td1 = TensorDict({"a": 1, "b": 1})
        td_func = TensorDict({"a": lambda x, y: x - y, "b": lambda x, y: x + y})
        td = td0.apply(lambda x, y, func: func(x, y), td1, td_func)
        assert td["a"] == -1
        assert td["b"] == 1

    def test_non_tensor_from_list(self):
        class X(TensorClass):
            non_tensor: str = None

        x = X(batch_size=3)
        x.non_tensor = NonTensorStack.from_list(["a", "b", "c"])
        assert x[0].non_tensor == "a"
        assert x[1].non_tensor == "b"

        x = X(non_tensor=NonTensorStack("a", "b", "c"), batch_size=3)
        assert x[0].non_tensor == "a"
        assert x[1].non_tensor == "b"

    def test_nontensor_dict(self, non_tensor_data):
        assert (
            TensorDict.from_dict(non_tensor_data.to_dict(), auto_batch_size=True)
            == non_tensor_data
        ).all()

    def test_nontensor_tensor(self):
        t1 = torch.tensor([1, 2, 3], dtype=torch.float)
        t2 = torch.tensor([1, 2, 3, 4], dtype=torch.float)
        stack = NonTensorStack(NonTensorData(t1), NonTensorData(t2))  # this works fine
        assert all(isinstance(t, torch.Tensor) for t in stack.tolist())
        stack = torch.stack(
            [NonTensorData(t1), NonTensorData(t2)]
        )  # this triggers an exception
        assert all(isinstance(t, torch.Tensor) for t in stack.tolist())

    def test_repeat(self):
        stack = NonTensorStack(
            NonTensorData("a", batch_size=(3,)),
            NonTensorData("b", batch_size=(3,)),
            stack_dim=1,
        )
        assert stack.shape == (3, 2)
        r = stack.repeat(1, 2)
        assert r.shape == (3, 4)
        assert r[0].tolist() == ["a", "b", "a", "b"]
        assert r[:, 0].tolist() == ["a", "a", "a"]
        assert r[:, -1].tolist() == ["b", "b", "b"]

    def test_repeat_interleave(self):
        stack = NonTensorStack(
            NonTensorData("a", batch_size=(3,)),
            NonTensorData("b", batch_size=(3,)),
            stack_dim=1,
        )
        assert stack.shape == (3, 2)
        r = stack.repeat_interleave(3, dim=1)
        assert isinstance(r, NonTensorStack)
        assert r[0].tolist() == ["a", "a", "a", "b", "b", "b"]

    def test_set(self, non_tensor_data):
        non_tensor_data.set(("nested", "another_string"), "another string!")
        assert (
            non_tensor_data.get(("nested", "another_string")).data == "another string!"
        )
        assert (
            non_tensor_data.get_non_tensor(("nested", "another_string"))
            == "another string!"
        )

    def test_setitem_edge_case(self):
        s = NonTensorStack("a string")
        t = NonTensorStack("another string")
        s[0][True] = t
        assert s[0].data == "another string"
        for i in (None, True):
            s = NonTensorStack("0", "1")
            t = NonTensorStack(NonTensorStack("2", "3"), stack_dim=1)
            assert t.batch_size == (2, 1)
            s[:, i] = t
            assert s.tolist() == ["2", "3"]

    def test_stack(self, non_tensor_data):
        assert (
            LazyStackedTensorDict.lazy_stack([non_tensor_data, non_tensor_data], 0).get(
                ("nested", "int")
            )
            == NonTensorData(3, batch_size=[2])
        ).all()
        assert torch.stack([non_tensor_data, non_tensor_data], 0).get_non_tensor(
            ("nested", "int")
        ) == [3, 3]
        with set_capture_non_tensor_stack(True):
            assert capture_non_tensor_stack()
            assert (
                torch.stack([non_tensor_data, non_tensor_data], 0).get_non_tensor(
                    ("nested", "int")
                )
                == 3
            )
        with set_capture_non_tensor_stack(False):
            assert not capture_non_tensor_stack()
            assert torch.stack([non_tensor_data, non_tensor_data], 0).get_non_tensor(
                ("nested", "int")
            ) == [3, 3]

        assert capture_non_tensor_stack(allow_none=True) is None
        assert isinstance(
            torch.stack([non_tensor_data, non_tensor_data], 0).get(("nested", "int")),
            NonTensorStack,
        )
        with set_capture_non_tensor_stack(False):
            assert isinstance(
                torch.stack([non_tensor_data, non_tensor_data], 0).get(
                    ("nested", "int")
                ),
                NonTensorStack,
            )

        non_tensor_copy = non_tensor_data.clone()
        non_tensor_copy.get(("nested", "int")).data = 4
        assert isinstance(
            torch.stack([non_tensor_data, non_tensor_copy], 0).get(("nested", "int")),
            LazyStackedTensorDict,
        )

    def test_stack_consolidate(self):
        td = torch.stack(
            [
                TensorDict(a="a string", b="b string"),
                TensorDict(a="another string", b="bnother string"),
            ]
        )
        tdc = td.consolidate()
        assert (tdc == td).all()
        assert tdc["a"] == ["a string", "another string"]

    def test_assign_non_tensor(self):
        data = TensorDict({}, [1, 10])

        data[0, 0] = TensorDict({"a": 0, "b": "a string!"}, [])

        assert data["b"] == "a string!"
        assert data.get("b").tolist() == [["a string!"] * 10]
        data[0, 1] = TensorDict({"a": 0, "b": "another string!"}, [])
        assert data.get("b").tolist() == [
            ["a string!"] + ["another string!"] + ["a string!"] * 8
        ]

        data = TensorDict({}, [1, 10])

        data[0, 0] = TensorDict({"a": 0, "b": "a string!"}, [])

        data[0, 5:] = TensorDict({"a": torch.zeros(5), "b": "another string!"}, [5])
        assert data.get("b").tolist() == [["a string!"] * 5 + ["another string!"] * 5]

        data = TensorDict({}, [1, 10])

        data[0, 0] = TensorDict({"a": 0, "b": "a string!"}, [])

        data[0, 0::2] = TensorDict(
            {"a": torch.zeros(5, dtype=torch.long), "b": "another string!"}, [5]
        )
        assert data.get("b").tolist() == [["another string!", "a string!"] * 5]

        data = TensorDict({}, [1, 10])

        data[0, 0] = TensorDict({"a": 0, "b": "a string!"}, [])

        data[0] = TensorDict(
            {"a": torch.zeros(10, dtype=torch.long), "b": "another string!"}, [10]
        )
        assert data.get("b").tolist() == [["another string!"] * 10]

    def test_ignore_lock(self):
        td = TensorDict({"a": {"b": "1"}}, batch_size=[10])
        td.lock_()
        td[0] = TensorDict({"a": {"b": "0"}}, [])
        assert td.is_locked
        assert td["a"].is_locked
        assert td[0]["a", "b"] == "0"
        assert td[1]["a", "b"] == "1"

    PAIRS = [
        ("something", "something else"),
        (0, 1),
        (0.0, 1.0),
        ([0, "something", 2], [9, "something else", 11]),
        ({"key1": 1, 2: 3}, {"key1": 4, 5: 6}),
    ]

    @pytest.mark.parametrize("pair", PAIRS)
    @pytest.mark.parametrize("strategy", ["shared", "memmap"])
    @pytest.mark.parametrize("update", ["update_", "update-inplace", "update"])
    def test_shared_memmap_single(self, pair, strategy, update, tmpdir):
        val0, val1 = pair
        td = TensorDict({"val": NonTensorData(data=val0, batch_size=[])}, [])
        if strategy == "shared":
            td.share_memory_()
        elif strategy == "memmap":
            td.memmap_(tmpdir)
        else:
            raise RuntimeError

        # Test that the Value is unpacked
        assert "data" not in td.get("val").__dict__
        assert td.get("val").data == val0
        assert td["val"] == val0

        # Check shared status
        if strategy == "shared":
            assert td._is_shared
            assert td.get("val")._is_shared
            assert td.get("val")._tensordict._is_shared
        elif strategy == "memmap":
            assert td._is_memmap
            assert td.get("val")._is_memmap
            assert td.get("val")._tensordict._is_memmap

            # check that the json has been updated
            td_load = TensorDict.load_memmap(tmpdir)
            assert td["val"] == td_load["val"]
            # with open(Path(tmpdir) / "val" / "meta.json") as file:
            #     print(json.load(file))

        # Update in place
        if update == "setitem":
            td["val"] = val1
        elif update == "update_":
            td.get("val").update_(
                NonTensorData(data=val1, batch_size=[]), non_blocking=False
            )
        elif update == "update-inplace":
            td.get("val").update(
                NonTensorData(data=val1, batch_size=[]),
                inplace=True,
                non_blocking=False,
            )
        elif update == "update":
            with pytest.raises(RuntimeError, match="lock"):
                td.get("val").update(
                    NonTensorData(data="something else", batch_size=[])
                )
            return

        # Test that the Value is unpacked
        assert td.get("val").data == val1
        assert td["val"] == val1

        # Check shared status
        if strategy == "shared":
            assert td._is_shared
            assert td.get("val")._is_shared
            assert td.get("val")._tensordict._is_shared
        elif strategy == "memmap":
            assert td._is_memmap
            assert td.get("val")._is_memmap
            assert td.get("val")._tensordict._is_memmap

            # check that the json has been updated
            td_load = TensorDict.load_memmap(tmpdir)
            assert td["val"] == td_load["val"]
            # with open(Path(tmpdir) / "val" / "meta.json") as file:
            #     print(json.load(file))

    @staticmethod
    def _run_worker(td, val1, update):
        set_list_to_stack(True).set()
        # Update in place
        if update == "setitem":
            td["val"] = NonTensorData(val1)
        elif update == "update_":
            td.get("val").update_(NonTensorData(data=val1), non_blocking=False)
        elif update == "update-inplace":
            td.get("val").update(
                NonTensorData(data=val1),
                inplace=True,
                non_blocking=False,
            )
        else:
            raise NotImplementedError
        # Test that the Value is unpacked
        assert td.get("val").data == val1
        assert td["val"] == val1

    @pytest.mark.slow
    @set_list_to_stack(True)
    @pytest.mark.parametrize("pair", PAIRS)
    @pytest.mark.parametrize("strategy", ["shared", "memmap"])
    @pytest.mark.parametrize("update", ["update_", "update-inplace"])
    def test_shared_memmap_mult(self, pair, strategy, update, tmpdir):
        from tensordict.tensorclass import _from_shared_nontensor

        val0, val1 = pair
        td = TensorDict({"val": NonTensorData(data=val0, batch_size=[])}, [])
        if strategy == "shared":
            td.share_memory_()
        elif strategy == "memmap":
            td.memmap_(tmpdir, share_non_tensor=True)
        else:
            raise RuntimeError

        # Test that the Value is unpacked
        assert td.get("val").data == val0
        assert td["val"] == val0

        # Check shared status
        if strategy == "shared":
            assert td._is_shared
            assert td.get("val")._is_shared
            assert td.get("val")._tensordict._is_shared
        elif strategy == "memmap":
            assert td._is_memmap
            assert td.get("val")._is_memmap
            assert td.get("val")._tensordict._is_memmap

            # check that the json has been updated
            td_load = TensorDict.load_memmap(tmpdir)
            assert td["val"] == td_load["val"]
            # with open(Path(tmpdir) / "val" / "meta.json") as file:
            #     print(json.load(file))

        proc = mp.get_context(mp_ctx).Process(
            target=self._run_worker, args=(td, val1, update)
        )
        proc.start()
        proc.join()

        # Test that the Value is unpacked
        assert _from_shared_nontensor(td.get("val")._non_tensordict["data"]) == val1
        assert td.get("val").data == val1
        assert td["val"] == val1

        # Check shared status
        if strategy == "shared":
            assert td._is_shared
            assert td.get("val")._is_shared
            assert td.get("val")._tensordict._is_shared
        elif strategy == "memmap":
            assert td._is_memmap
            assert td.get("val")._is_memmap
            assert td.get("val")._tensordict._is_memmap

            # check that the json has been updated
            td_load = TensorDict.load_memmap(tmpdir)
            assert td["val"] == td_load["val"]
            # with open(Path(tmpdir) / "val" / "meta.json") as file:
            #     print(json.load(file))

    @pytest.mark.parametrize("json_serializable", [True, False])
    @pytest.mark.parametrize("device", [None, *get_available_devices()])
    def test_memmap_stack(self, tmpdir, json_serializable, device):
        if json_serializable:
            data = torch.stack(
                [
                    NonTensorData(data=0, device=device),
                    NonTensorData(data=1, device=device),
                ]
            )

        else:
            data = torch.stack(
                [
                    NonTensorData(data=DummyPicklableClass(0), device=device),
                    NonTensorData(data=DummyPicklableClass(1), device=device),
                ]
            )
        data = torch.stack([data] * 3)
        data_memmap = data.memmap(tmpdir)
        device_str = "null" if device is None else f'"{device}"'
        with open(f"{tmpdir}/meta.json") as f:
            if json_serializable:
                assert (
                    f.read()
                    == f'{{"_type":"<class \'tensordict.tensorclass.NonTensorStack\'>","stack_dim":0,"device":{device_str},"data":[[0,1],[0,1],[0,1]]}}'
                )
            else:
                assert (
                    f.read()
                    == f'{{"_type":"<class \'tensordict.tensorclass.NonTensorStack\'>","stack_dim":0,"device":{device_str},"data":"pickle.pkl"}}'
                )
        data_recon = TensorDict.load_memmap(tmpdir)
        assert data_recon.batch_size == data.batch_size
        assert data_recon.device == data.device
        assert data_recon.tolist() == data.tolist()
        assert data_memmap[0].is_memmap()
        assert data_memmap.is_memmap()
        assert data_memmap._is_memmap

    @pytest.mark.skipif(IS_FB, reason="deactivating on fbcode")
    def test_memmap_stack_updates(self, tmpdir):
        data = torch.stack([NonTensorData(data=0), NonTensorData(data=1)], 0)
        assert is_non_tensor(data)
        data = torch.stack([data] * 3)
        assert is_non_tensor(data), data
        data = data.clone()
        assert is_non_tensor(data)
        data.memmap_(tmpdir)
        data_recon = TensorDict.load_memmap(tmpdir)
        assert data.tolist() == data_recon.tolist()
        assert data.is_memmap()
        assert data._is_memmap
        assert data[0, 0]._is_memmaped_from_above()

        data_other = torch.stack([NonTensorData(data=2), NonTensorData(data=3)], 0)
        data_other = torch.stack([data_other] * 3)
        with pytest.raises(RuntimeError, match="locked"):
            data.update(data_other)
        data.update(data_other, inplace=True, non_blocking=False)
        assert data[0, 0]._is_memmaped_from_above()
        assert data.is_memmap()
        assert data._is_memmap
        assert data.tolist() == [[2, 3]] * 3
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist()

        data_other = torch.stack([NonTensorData(data=4), NonTensorData(data=5)], 0)
        data_other = torch.stack([data_other] * 3)
        data.update_(data_other)
        assert data[0, 0]._is_memmaped_from_above()
        assert data.is_memmap()
        assert data._is_memmap
        assert data.tolist() == [[4, 5]] * 3
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist()

        data.update(NonTensorData(data=6), inplace=True, non_blocking=False)
        assert data.is_memmap()
        assert data._is_memmap
        assert data.tolist() == [[6] * 2] * 3
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist()

        data.update_(NonTensorData(data=7))
        assert data.is_memmap()
        assert data._is_memmap
        assert data.tolist() == [[7] * 2] * 3
        assert data.tolist() == TensorDict.load_memmap(tmpdir).tolist()

        assert data[0, 0]._is_memmaped_from_above()
        # Should raise an exception
        assert isinstance(data[0, 0], NonTensorData)
        with pytest.raises(
            RuntimeError,
            match="Cannot update a leaf NonTensorDataBase from a memmaped parent NonTensorStack",
        ):
            data[0, 0].update(NonTensorData(data=1), inplace=True, non_blocking=False)

        # Should raise an exception
        with pytest.raises(
            RuntimeError,
            match="Cannot update a leaf NonTensorDataBase from a memmaped parent NonTensorStack",
        ):
            data[0].update(NonTensorData(data=1), inplace=True, non_blocking=False)

        # should raise an exception
        with pytest.raises(
            ValueError,
            match="Cannot update a NonTensorDataBase object with a NonTensorStack",
        ):
            out = NonTensorData(data=1).update(data, inplace=True, non_blocking=False)
        # as suggested by the error message this works
        out = (
            NonTensorData(data=1, batch_size=data.batch_size)
            .maybe_to_stack()
            .update(data, inplace=True, non_blocking=False)
        )
        assert out.tolist() == data.tolist()

        data[0, 0] = NonTensorData(data=99)
        assert data.tolist() == [[99, 7], [7, 7], [7, 7]]
        assert (
            data.tolist() == TensorDict.load_memmap(tmpdir).tolist()
        ), TensorDict.load_memmap(tmpdir).tolist()

        data.update_at_(NonTensorData(data=99), (0, 1))
        assert data.tolist() == [[99, 99], [7, 7], [7, 7]], data.tolist()
        assert (
            data.tolist() == TensorDict.load_memmap(tmpdir).tolist()
        ), TensorDict.load_memmap(tmpdir).tolist()

    def test_shared_limitations(self):
        # Sharing a special type works but it's locked for writing
        @dataclass
        class MyClass:
            string: str

        val0 = MyClass(string="a string!")

        td = TensorDict({"val": NonTensorData(data=val0, batch_size=[])}, [])
        td.share_memory_()

        # with pytest.raises(RuntimeError)
        val1 = MyClass(string="another string!")
        with pytest.raises(ValueError, match="Failed to update 'val' in tensordict"):
            td.update(
                TensorDict({"val": NonTensorData(data=val1, batch_size=[])}, []),
                inplace=True,
                non_blocking=False,
            )
        with pytest.raises(
            NotImplementedError, match="Updating MyClass within a shared/memmaped"
        ):
            td.update_(TensorDict({"val": NonTensorData(data=val1, batch_size=[])}, []))

        # We can update a batched NonTensorData to a NonTensorStack if it's not already shared
        td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
        assert td.get("val").tolist() == [0, 1] * 5
        td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        td.share_memory_()
        with pytest.raises(
            RuntimeError,
            match="You're attempting to update a leaf in-place with a shared",
        ):
            td[1::2] = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])

    def _update_stack(self, td):
        td[1::2] = TensorDict({"val": NonTensorData(data=3, batch_size=[5])}, [5])

    @pytest.mark.slow
    @pytest.mark.parametrize("update", ["update_at_", "slice"])
    @pytest.mark.parametrize(
        "strategy,share_non_tensor",
        [["shared", None], ["memmap", True], ["memmap", False]],
    )
    def test_shared_stack(self, strategy, update, share_non_tensor, tmpdir):
        td = TensorDict({"val": NonTensorData(data=0, batch_size=[10])}, [10])
        newdata = TensorDict({"val": NonTensorData(data=1, batch_size=[5])}, [5])
        if update == "slice":
            td[1::2] = newdata
        elif update == "update_at_":
            td.update_at_(newdata, slice(1, None, 2))
        else:
            raise NotImplementedError
        if strategy == "shared":
            td.share_memory_()
        elif strategy == "memmap":
            td.memmap_(tmpdir, share_non_tensor=share_non_tensor)
        else:
            raise NotImplementedError
        assert td.get("val").tolist() == [0, 1] * 5

        newdata = TensorDict({"val": NonTensorData(data=2, batch_size=[5])}, [5])
        if update == "slice":
            td[1::2] = newdata
        elif update == "update_at_":
            td.update_at_(newdata, slice(1, None, 2))
        else:
            raise NotImplementedError

        assert td.get("val").tolist() == [0, 2] * 5
        if strategy == "memmap":
            assert TensorDict.load_memmap(tmpdir).get("val").tolist() == [0, 2] * 5

        proc = mp.get_context(mp_ctx).Process(target=self._update_stack, args=(td,))
        proc.start()
        proc.join()
        if share_non_tensor in (True, None):
            assert td.get("val").tolist() == [0, 3] * 5
        else:
            assert td.get("val").tolist() == [0, 2] * 5

        if strategy == "memmap":
            assert TensorDict.load_memmap(tmpdir).get("val").tolist() == [0, 3] * 5

    def test_squeeze_unsqueeze(self):
        td = TensorDict(foo=NonTensorData(3))
        tdu = td.unsqueeze(0)
        assert isinstance(tdu.get("foo"), NonTensorStack)
        assert tdu["foo"] == [3]
        assert isinstance(tdu.squeeze()["foo"], int)
        assert isinstance(tdu.squeeze().get("foo"), NonTensorData)
        assert isinstance(tdu.squeeze(0)["foo"], int)
        assert isinstance(tdu.squeeze(0).get("foo"), NonTensorData)

    def test_view(self):
        td = NonTensorStack(*[str(i) for i in range(60)])
        tdv = td.view(3, 4, 5)
        assert isinstance(tdv, NonTensorStack)
        assert isinstance(tdv[0], NonTensorStack)
        assert isinstance(tdv[0, 0], NonTensorStack)
        assert tdv.shape == (3, 4, 5)
        assert tdv.view(60).shape == (60,)
        assert tdv.view(60).tolist() == [str(i) for i in range(60)]
        assert tdv.flatten().tolist() == [str(i) for i in range(60)]

    def test_where(self):
        condition = torch.tensor([True, False])
        tensor = NonTensorStack(
            *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
        )
        other = NonTensorStack(
            *[NonTensorData(["b"]), NonTensorData(["b"])], batch_size=(2,)
        )
        out = NonTensorStack(
            *[NonTensorData(["none"]), NonTensorData(["none"])], batch_size=(2,)
        )
        result = tensor.where(condition=condition, other=other, out=out, pad=0)
        assert result.tolist() == [["a"], ["b"]]
        condition = torch.tensor([True, False])
        tensor = NonTensorStack(
            *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
        )
        other = NonTensorStack(
            *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
        )
        out = NonTensorStack(
            *[NonTensorData(["a"]), NonTensorData(["a"])], batch_size=(2,)
        )
        result = tensor.where(condition=condition, other=other, out=out, pad=0)
        assert result.tolist() == [["a"], ["a"]]

    def test_linked_list_str(self):
        td = TensorDict(a=NonTensorStack("foo", "bar"), batch_size=(2,))
        # str must not double-wrap: list has no __str__ of its own, so a
        # naive super().__str__() would bounce back to LinkedList.__repr__.
        assert str(td["a"]) == "LinkedList(['foo', 'bar'])"
        assert repr(td["a"]) == "LinkedList(['foo', 'bar'])"

    def test_nontensorstack_boolean_mask(self):
        td = TensorDict(
            {
                "x": torch.arange(3),
                "instr": NonTensorStack("a", "b", "c"),
            },
            batch_size=[3],
        )
        mask = torch.tensor([True, False, True])
        masked = td[mask]
        assert isinstance(masked.get("instr"), NonTensorStack)
        assert masked.get("instr").batch_size == torch.Size([2])
        assert masked.get("instr").tolist() == ["a", "c"]
        assert masked["instr"] == ["a", "c"]
        # all-False mask
        empty = td[torch.zeros(3, dtype=torch.bool)]
        assert empty.batch_size == torch.Size([0])
        assert empty.get("instr").batch_size == torch.Size([0])
        assert empty.get("instr").tolist() == []

    def test_nontensorstack_boolean_mask_2d(self):
        instr = torch.stack(
            [NonTensorStack(*[f"{i}{j}" for j in range(2)]) for i in range(3)]
        )
        td = TensorDict(
            {"instr": instr, "x": torch.arange(6).view(3, 2)}, batch_size=[3, 2]
        )
        # mask along dim 0 of a 2D NonTensorStack used to raise an IndexError
        mask = torch.tensor([True, False, True])
        masked = td[mask]
        assert masked.get("instr").tolist() == [["00", "01"], ["20", "21"]]
        assert masked["x"].tolist() == [[0, 1], [4, 5]]
        # full-shape mask flattens
        full_mask = torch.tensor([[True, False], [False, True], [True, True]])
        assert td[full_mask].get("instr").tolist() == ["00", "11", "20", "21"]
        # boolean setitem with nested stacks
        value = TensorDict(
            {
                "instr": torch.stack(
                    [NonTensorStack("p", "q"), NonTensorStack("r", "s")]
                ),
                "x": torch.zeros(2, 2, dtype=torch.long),
            },
            batch_size=[2, 2],
        )
        td[mask] = value
        assert td.get("instr").tolist() == [["p", "q"], ["10", "11"], ["r", "s"]]

    def test_nontensorstack_scalar_bool_index(self):
        stack = NonTensorStack("a", "b")
        assert stack[torch.tensor(True)].tolist() == [["a", "b"]]
        assert stack[torch.tensor(False)].batch_size == torch.Size([0, 2])
        indexed = stack[(torch.tensor(True),)]
        assert isinstance(indexed, NonTensorStack)
        assert indexed.batch_size == torch.Size([1, 2])
        assert indexed.tolist() == [["a", "b"]]
        # masking to empty keeps the NonTensorStack type
        empty = stack[torch.zeros(2, dtype=torch.bool)]
        assert isinstance(empty, NonTensorStack)
        assert empty.batch_size == torch.Size([0])
        assert empty.tolist() == []


class TestMetaData:
    def test_typed_metadata(self):
        d = MetaData[int](0, batch_size=(3,))
        assert d.data == 0
        assert isinstance(d, MetaData[int])
        with pytest.raises(TypeError, match="Expected data of type int, got str"):
            MetaData[int]("a string")
        cls = MetaData[int]
        assert issubclass(cls, MetaData)
        d = cls(0, batch_size=(3,))
        assert isinstance(d, MetaData)
        # Test caching
        assert isinstance(d, MetaData[int])

    def test_expand(self):
        d = MetaData(0, batch_size=(3,))
        d_expand = d.expand((2, 3))
        assert d_expand.shape == (2, 3)
        assert d_expand.data == 0
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, MetaData)
        d = MetaData(0, batch_size=(1,))
        d_expand = d.expand((2, 3))
        assert d_expand.shape == (2, 3)
        assert d_expand.data == 0
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, MetaData)

    def test_expand_nested(self):
        d = TensorDict(foo=MetaData(0, batch_size=(3,)), batch_size=(3,))
        d_expand = d.expand((2, 3)).get("foo")
        assert d_expand.shape == (2, 3)
        assert d_expand.data == 0
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, MetaData)
        d = TensorDict(foo=MetaData(0, batch_size=(1,)), batch_size=(1,))
        d_expand = d.expand((2, 3)).get("foo")
        assert d_expand.shape == (2, 3)
        assert d_expand.data == 0
        assert d_expand.tolist() == [[0 for _ in range(3)] for _ in range(2)]
        assert isinstance(d_expand, MetaData)

    def test_metadata(self):
        foo = MetaData(3)
        assert is_non_tensor(foo)
        assert _pass_through(foo)

    def test_squeeze_unsqueeze(self):
        td = TensorDict(foo=MetaData(3))
        tdu = td.unsqueeze(0)
        assert isinstance(tdu.get("foo"), MetaData)
        assert tdu["foo"] == 3
        assert isinstance(tdu.squeeze()["foo"], int)
        assert isinstance(tdu.squeeze().get("foo"), MetaData)
        assert isinstance(tdu.squeeze(0)["foo"], int)
        assert isinstance(tdu.squeeze(0).get("foo"), MetaData)

    def test_stack(self):
        data = "a string"
        td0 = TensorDict(foo=MetaData(data))
        td1 = TensorDict(foo=MetaData(data))
        assert isinstance(td0.get("foo"), MetaData)
        assert isinstance(td1.get("foo"), MetaData)
        tds = torch.stack([td0, td1])
        foo = tds.get("foo")
        assert isinstance(foo, MetaData)
        assert foo.batch_size == (2,)
        another_data = "another string"
        td1 = TensorDict(foo=MetaData(another_data))
        tds = torch.stack([td0, td1])
        foo = tds.get("foo")
        assert isinstance(foo, NonTensorStack)
        assert foo.batch_size == (2,)

    def test_auto_batch_size_scalar_metadata(self):
        # Regression test for #1696: a scalar MetaData entry must not
        # constrain auto_batch_size_ to an empty batch size.
        td = TensorDict(data=torch.zeros(16, 10), meta=MetaData(slice(-1)))
        td.auto_batch_size_(2)
        assert td.batch_size == torch.Size([16, 10])
        td = TensorDict(data=torch.zeros(16, 10), meta=MetaData(slice(-1)))
        td.auto_batch_size_()
        assert td.batch_size == torch.Size([16, 10])
        # scalar NonTensorData behaves the same way
        td = TensorDict(data=torch.zeros(16, 10), meta=NonTensorData("hi"))
        td.auto_batch_size_(2)
        assert td.batch_size == torch.Size([16, 10])


class TestUnbatchedTensor:
    @staticmethod
    def _result_or_exception(fun, arg):
        try:
            return False, fun(arg)
        except Exception as err:
            return True, (type(err), str(err))

    @classmethod
    def _assert_same_conversion(cls, fun, tensor):
        expected_is_exc, expected = cls._result_or_exception(fun, tensor)
        actual_is_exc, actual = cls._result_or_exception(fun, UnbatchedTensor(tensor))

        assert actual_is_exc is expected_is_exc
        if expected_is_exc:
            assert actual[0] is expected[0]
            assert actual[1] == expected[1]
        else:
            assert actual == expected
            assert not isinstance(actual, UnbatchedTensor)

    def test_auto_batch_size(self):
        td = TensorDict(a=UnbatchedTensor(0), b=torch.randn(2, 3)).auto_batch_size_(
            batch_dims=2
        )
        assert td.shape == (2, 3)
        assert td["a"] == 0

    def test_unbatched(self):
        assert UnbatchedTensor._pass_through
        td = TensorDict(
            a=UnbatchedTensor(torch.randn(10)),
            b=torch.randn(3),
            batch_size=(3,),
        )
        assert _pass_through(td.get("a"))
        assert isinstance(td["a"], torch.Tensor)
        assert isinstance(td.get("a"), UnbatchedTensor)

    def test_unbatched_shape_ops(self):
        td = TensorDict(
            a=UnbatchedTensor(torch.randn(10)),
            b=torch.randn(3),
            batch_size=(3,),
        )
        # get item
        assert td[0]["a"] is td["a"]
        assert td[:]["a"] is td["a"]

        unbind = td.unbind(0)[0]
        assert unbind["a"] is td["a"]
        assert unbind.batch_size == ()

        split = td.split(1)[0]
        assert split["a"] is td["a"]
        assert split.batch_size == (1,)
        assert td.split((2, 1))[0]["a"] is td["a"]

        reshape = td.reshape((1, 3))
        assert reshape["a"] is td["a"]
        assert reshape.batch_size == (1, 3)
        transpose = reshape.transpose(0, 1)
        assert transpose["a"] is td["a"]
        assert transpose.batch_size == (3, 1)
        permute = reshape.permute(1, 0)
        assert permute["a"] is td["a"]
        assert permute.batch_size == (3, 1)
        squeeze = reshape.squeeze()
        assert squeeze["a"] is td["a"]
        assert squeeze.batch_size == (3,)

        view = td.view((1, 3))
        assert view["a"] is td["a"]
        assert view.batch_size == (1, 3)
        unsqueeze = td.unsqueeze(0)
        assert unsqueeze["a"] is td["a"]
        assert unsqueeze.batch_size == (1, 3)
        gather = td.gather(0, torch.tensor((0,)))
        assert gather["a"] is td["a"]
        assert gather.batch_size == (1,)

        unflatten = td.unflatten(0, (1, 3))
        assert unflatten["a"] is td["a"]
        assert unflatten.batch_size == (1, 3)

        flatten = unflatten.flatten(0, 1)
        assert flatten["a"] is td["a"]
        assert flatten.batch_size == (3,)

    def test_unbatched_torch_func(self):
        td = TensorDict(
            a=UnbatchedTensor(torch.randn(10)),
            b=torch.randn(3),
            batch_size=(3,),
        )
        assert torch.unbind(td, 0)[0]["a"] is td["a"]
        assert torch.stack([td, td], 0)[0]["a"] is td["a"]
        assert torch.cat([td, td], 0)[0]["a"] is td["a"]
        assert (torch.ones_like(td)["a"] == 1).all()
        assert torch.unsqueeze(td, 0)["a"] is td["a"]
        assert torch.squeeze(td)["a"] is td["a"]
        unflatten = torch.unflatten(td, 0, (1, 3))
        assert unflatten["a"] is td["a"]
        flatten = torch.flatten(unflatten, 0, 1)
        assert flatten["a"] is td["a"]
        permute = torch.permute(unflatten, (1, 0))
        assert permute["a"] is td["a"]
        transpose = torch.transpose(unflatten, 1, 0)
        assert transpose["a"] is td["a"]

    def test_unbatched_other_ops(self):
        td = TensorDict(
            a=UnbatchedTensor(torch.randn(10)),
            b=torch.randn(3),
            c_d=UnbatchedTensor(torch.randn(10)),
            batch_size=(3,),
        )
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif is_npu_available():
            device = torch.device("npu:0")
        else:
            device = torch.device("cpu")
        assert td.copy()["a"] is td["a"]
        assert td.int()["a"].dtype == torch.int
        assert td.to(device)["a"].device == device
        assert td.select("a")["a"] is td["a"]
        assert td.exclude("b")["a"] is td["a"]
        assert td.unflatten_keys(separator="_")["c", "d"] is td["c_d"]
        assert td.unflatten_keys(separator="_").flatten_keys()["c.d"] is td["c_d"]

    def test_unbatched_getitem_returns_unbatched(self):
        data = torch.randn(7, 11)
        td = TensorDict(
            a=UnbatchedTensor(data),
            b=torch.randn(3, 4),
            batch_size=(3, 4),
        )
        result = td["a"]
        assert isinstance(result, torch.Tensor)
        assert isinstance(result, UnbatchedTensor)
        assert result.data_ptr() == data.data_ptr()

    def test_unbatched_stack_same_data_no_warning(self):
        data = torch.randn(5)
        td1 = TensorDict(a=torch.ones(3), ub=UnbatchedTensor(data), batch_size=[3])
        td2 = TensorDict(a=torch.ones(3), ub=UnbatchedTensor(data), batch_size=[3])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            stacked = torch.stack([td1, td2])
        assert stacked["ub"].data_ptr() == data.data_ptr()

    def test_unbatched_stack_different_data_warns(self):
        td1 = TensorDict(
            a=torch.ones(3), ub=UnbatchedTensor(torch.zeros(5)), batch_size=[3]
        )
        td2 = TensorDict(
            a=torch.ones(3), ub=UnbatchedTensor(torch.ones(5)), batch_size=[3]
        )
        with pytest.warns(UserWarning, match="different data storage"):
            torch.stack([td1, td2])

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    def test_unbatched_vmap(self):
        from torch import vmap

        data = torch.randn(7, 11)
        td = TensorDict(
            a=torch.randn(4, 3, 5),
            unbatched=UnbatchedTensor(data),
            batch_size=[4, 3],
        )
        result = vmap(lambda x: x)(td)
        assert result.batch_size == torch.Size([4, 3])
        assert isinstance(result.get("unbatched"), UnbatchedTensor)
        assert result.get("unbatched").data_ptr() == data.data_ptr()
        assert result.get("unbatched").batch_size == torch.Size([4, 3])
        assert result["a"].shape == (4, 3, 5)

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    def test_unbatched_vmap_return_unbatched_tensor(self):
        from torch import vmap

        data = torch.randn(7, 11)
        td = TensorDict(
            a=torch.randn(4, 3, 5),
            unbatched=UnbatchedTensor(data),
            batch_size=[4, 3],
        )
        result = vmap(lambda x: x["unbatched"])(td)
        assert isinstance(result, UnbatchedTensor)
        assert result.data_ptr() == data.data_ptr()
        assert result.batch_size == torch.Size([4, 3])

    @pytest.mark.skipif(PYTORCH_TEST_FBCODE, reason="vmap now working in fbcode")
    def test_unbatched_vmap_direct_input(self):
        from torch import vmap

        data = torch.randn(7, 11)
        unbatched = UnbatchedTensor(data, batch_size=[4])
        result = vmap(lambda x: x)(unbatched)
        assert isinstance(result, UnbatchedTensor)
        assert result.data_ptr() == data.data_ptr()
        assert result.batch_size == torch.Size([4])

    def test_unbatched_tensorclass_attr_returns_unbatched(self):
        @tensorclass
        class MyClass:
            config: torch.Tensor
            value: torch.Tensor

        data = torch.tensor([1.0, 2.0])
        tc = MyClass(
            config=UnbatchedTensor(data),
            value=torch.randn(3),
            batch_size=[3],
        )
        assert isinstance(tc.config, torch.Tensor)
        assert isinstance(tc.config, UnbatchedTensor)
        assert tc.config.data_ptr() == data.data_ptr()

        result = tc.get("config")
        assert isinstance(result, torch.Tensor)
        assert isinstance(result, UnbatchedTensor)
        assert result.data_ptr() == data.data_ptr()

    def test_unbatched_tensor_is_tensor(self):
        data = torch.randn(5, 3)
        ut = UnbatchedTensor(data)
        assert isinstance(ut, torch.Tensor)
        assert isinstance(ut, UnbatchedTensor)
        assert ut.shape == data.shape
        assert ut.data_ptr() == data.data_ptr()

    def test_unbatched_tolist(self):
        data = torch.tensor([1, 2, 3])
        ut = UnbatchedTensor(data)
        assert ut.tolist() == [1, 2, 3]

    @pytest.mark.parametrize(
        "data",
        [
            torch.tensor(3),
            torch.tensor([3]),
            torch.tensor([[3]]),
            torch.tensor(True),
            torch.tensor(1.25),
            torch.tensor(1 + 2j),
            torch.tensor([1, 2]),
            torch.empty(0),
        ],
    )
    def test_unbatched_python_conversions_match_tensor(self, data):
        conversion_funs = [
            str,
            format,
            lambda x: format(x, ".2f"),
            bool,
            int,
            float,
            complex,
            operator.index,
            round,
            lambda x: round(x, 1),
            math.trunc,
            math.floor,
            math.ceil,
            lambda x: x.item(),
        ]
        for fun in conversion_funs:
            self._assert_same_conversion(fun, data)

    def test_unbatched_index_protocol(self):
        data = torch.tensor(1)
        ut = UnbatchedTensor(data)
        assert [10, 20, 30][ut] == [10, 20, 30][data]
        assert hex(ut) == hex(data)

    def test_unbatched_numpy_array_conversion(self):
        data = torch.tensor([[1, 2], [3, 4]])
        ut = UnbatchedTensor(data)
        np.testing.assert_array_equal(np.asarray(ut), np.asarray(data))
        np.testing.assert_array_equal(
            np.asarray(ut, dtype=np.float64), np.asarray(data, dtype=np.float64)
        )

    def test_unbatched_string_representation(self):
        data = torch.tensor([1.0, 2.0])
        ut = UnbatchedTensor(data)
        assert str(ut) == str(data)
        assert repr(ut) == f"UnbatchedTensor({data!r})"

    def test_auto_batch_size_nontensor_not_excluded(self):
        td = TensorDict.from_dict(
            {"query": ["str1", "str2", "str3", "str4"]},
            auto_batch_size=True,
            batch_dims=1,
        )
        assert td.batch_size == torch.Size([4])
        assert "query" in td.keys()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
