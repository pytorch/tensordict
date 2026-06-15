# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import gc
import importlib.util
import os
import pathlib
import platform
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from packaging import version
from tensordict import LazyStackedTensorDict, TensorDict
from tensordict._td import is_tensor_collection
from tensordict.memmap import MemoryMappedTensor
from tensordict.nn import TensorDictParams
from tensordict.tensorclass import NonTensorData, NonTensorStack
from torch import multiprocessing as mp, nn

if os.getenv("PYTORCH_TEST_FBCODE"):
    IS_FB = True
    from pytorch.tensordict.test._utils_internal import (
        is_npu_available,
        TestTensorDictsBase,
    )
else:
    IS_FB = False
    from _utils_internal import is_npu_available, TestTensorDictsBase

try:
    from functorch import dim as ftdim

    _has_funcdim = True
except ImportError:
    from tensordict.utils import _ftdim_mock as ftdim

    _has_funcdim = False

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


@pytest.mark.slow
class TestMPInplace:
    @classmethod
    def _remote_process(
        cls, worker_id, command_pipe_child, command_pipe_parent, tensordict
    ):
        command_pipe_parent.close()
        while True:
            cmd, val = command_pipe_child.recv()
            if cmd == "recv":
                b = tensordict.get("b")
                assert (b == val).all()
                command_pipe_child.send("done")
            elif cmd == "send":
                a = torch.ones(2) * val
                tensordict.set_("a", a)
                assert (
                    tensordict.get("a") == a
                ).all(), f'found {a} and {tensordict.get("a")}'
                command_pipe_child.send("done")
            elif cmd == "set_done":
                tensordict.set_("done", torch.ones(1, dtype=torch.bool))
                command_pipe_child.send("done")
            elif cmd == "set_undone_":
                tensordict.set_("done", torch.zeros(1, dtype=torch.bool))
                command_pipe_child.send("done")
            elif cmd == "update":
                tensordict.update_(
                    TensorDict(
                        source={"a": tensordict.get("a").clone() + 1},
                        batch_size=tensordict.batch_size,
                    )
                )
                command_pipe_child.send("done")
            elif cmd == "update_":
                tensordict.update_(
                    TensorDict(
                        source={"a": tensordict.get("a").clone() - 1},
                        batch_size=tensordict.batch_size,
                    )
                )
                command_pipe_child.send("done")

            elif cmd == "close":
                command_pipe_child.close()
                break

    @classmethod
    def _driver_func(cls, tensordict, tensordict_unbind):
        procs = []
        children = []
        parents = []

        for i in range(2):
            command_pipe_parent, command_pipe_child = mp.Pipe()
            proc = mp.get_context(mp_ctx).Process(
                target=cls._remote_process,
                args=(i, command_pipe_child, command_pipe_parent, tensordict_unbind[i]),
            )
            proc.start()
            command_pipe_child.close()
            parents.append(command_pipe_parent)
            children.append(command_pipe_child)
            procs.append(proc)

        try:
            b = torch.ones(2, 1) * 10
            tensordict.set_("b", b)
            assert (tensordict["b"] == 10).all()
            for i in range(2):
                parents[i].send(("recv", 10))
                is_done = parents[i].recv()
                assert is_done == "done"

            for i in range(2):
                parents[i].send(("send", i))
                is_done = parents[i].recv()
                assert is_done == "done"
            a = tensordict.get("a").clone()
            assert (a[0] == 0).all()
            assert (a[1] == 1).all()

            assert not tensordict.get("done").any()
            for i in range(2):
                parents[i].send(("set_done", i))
                is_done = parents[i].recv()
                assert is_done == "done"
            assert tensordict.get("done").all()

            for i in range(2):
                parents[i].send(("set_undone_", i))
                is_done = parents[i].recv()
                assert is_done == "done"
            assert not tensordict.get("done").any()

            a_prev = tensordict.get("a").clone().contiguous()
            for i in range(2):
                parents[i].send(("update_", i))
                is_done = parents[i].recv()
                assert is_done == "done"
            new_a = tensordict.get("a").clone().contiguous()
            torch.testing.assert_close(a_prev - 1, new_a)

            a_prev = tensordict.get("a").clone().contiguous()
            for i in range(2):
                parents[i].send(("update", i))
                is_done = parents[i].recv()
                assert is_done == "done"
            new_a = tensordict.get("a").clone().contiguous()
            torch.testing.assert_close(a_prev + 1, new_a)

            for i in range(2):
                parents[i].send(("close", None))
        finally:
            try:
                for i in range(2):
                    procs[i].join(timeout=1)
            except Exception:
                for i in range(2):
                    if procs[i].is_alive():
                        procs[i].terminate()

    @pytest.mark.parametrize(
        "td_type",
        [
            "memmap",
            "memmap_stack",
            "contiguous",
            "stack",
        ],
    )
    @pytest.mark.parametrize("unbind_as", ["iter", "subtd", "unbind"])
    def test_mp(self, td_type, unbind_as):
        tensordict = TensorDict(
            source={
                "a": torch.randn(2, 2),
                "b": torch.randn(2, 1),
                "done": torch.zeros(2, 1, dtype=torch.bool),
            },
            batch_size=[2],
        )
        if td_type == "contiguous":
            tensordict = tensordict.share_memory_()
        elif td_type == "stack":
            tensordict = TensorDict.lazy_stack(
                [
                    tensordict[0].clone().share_memory_(),
                    tensordict[1].clone().share_memory_(),
                ],
                0,
            )
        elif td_type == "memmap":
            tensordict = tensordict.memmap_()
        elif td_type == "memmap_stack":
            tensordict = TensorDict.lazy_stack(
                [
                    tensordict[0].clone().memmap_(),
                    tensordict[1].clone().memmap_(),
                ],
                0,
            )
        else:
            raise NotImplementedError
        if unbind_as == "iter":
            tdunbind = tensordict
        elif unbind_as == "subtd":
            tdunbind = (
                tensordict._get_sub_tensordict(0),
                tensordict._get_sub_tensordict(1),
            )
        elif unbind_as == "unbind":
            tdunbind = tensordict.unbind(0)
        else:
            raise NotImplementedError

        self._driver_func(tensordict, tdunbind)


@pytest.mark.slow
@pytest.mark.parametrize(
    "td_name,device",
    TestTensorDictsBase.TYPES_DEVICES,
)
class TestTensorDictMP(TestTensorDictsBase):
    # Tests sharing a locked tensordict

    @staticmethod
    def add1(x):
        return x + 1

    @staticmethod
    def add1_app(x):
        return x.apply(lambda x: x + 1)

    @staticmethod
    def matmul_app_error(x):
        # non point-wise ops are not supported
        return x @ 1

    @pytest.mark.parametrize(
        "chunksize,num_chunks", [[None, 2], [4, None], [None, None], [2, 2]]
    )
    def test_chunksize_num_chunks(
        self, td_name, device, chunksize, num_chunks, _pool_fixt, dim=0
    ):
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            with pytest.raises(
                RuntimeError, match="Cannot call map on a TensorDictParams object"
            ):
                td.map(self.matmul_app_error, dim=dim, pool=_pool_fixt)
            return
        if chunksize is not None and num_chunks is not None:
            with pytest.raises(ValueError, match="but not both"):
                td.map(
                    self.write_pid,
                    dim=dim,
                    chunksize=chunksize,
                    num_chunks=num_chunks,
                    pool=_pool_fixt,
                )
            return
        mapped = td.map(
            self.write_pid,
            dim=dim,
            chunksize=chunksize,
            num_chunks=num_chunks,
            pool=_pool_fixt,
        )
        pids = mapped.get("pid").unique()
        if chunksize is not None:
            assert pids.numel() == -(td.shape[0] // -chunksize)
        elif num_chunks is not None:
            assert pids.numel() == num_chunks

    @pytest.mark.parametrize("dim", [-2, -1, 0, 1, 2, 3])
    def test_map(self, td_name, device, dim, _pool_fixt):
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            with pytest.raises(
                RuntimeError, match="Cannot call map on a TensorDictParams object"
            ):
                td.map(self.add1_app, dim=dim, pool=_pool_fixt)
            return
        assert (
            td.map(self.add1_app, dim=dim, pool=_pool_fixt) == td.apply(self.add1)
        ).all()

    @pytest.mark.parametrize("dim", [-2, -1, 0, 1, 2, 3])
    def test_map_exception(self, td_name, device, dim, _pool_fixt):
        td = getattr(self, td_name)(device)
        if td_name == "td_params":
            with pytest.raises(
                RuntimeError, match="Cannot call map on a TensorDictParams object"
            ):
                td.map(self.matmul_app_error, dim=dim, pool=_pool_fixt)
            return
        with pytest.raises(TypeError, match="unsupported operand"):
            td.map(self.matmul_app_error, dim=dim, pool=_pool_fixt)

    def test_sharing_locked_td(self, td_name, device):
        td = getattr(self, td_name)(device)
        if td_name in ("sub_td", "sub_td2"):
            pytest.skip("cannot lock sub-tds")
        if td_name in ("td_h5",):
            pytest.skip("h5 files should not be opened across different processes.")
        ctx = mp.get_context(mp_ctx)
        q = ctx.Queue(1)
        p = ctx.Process(target=self.worker_lock, args=(td.lock_(), q))
        p.start()
        try:
            assert q.get(timeout=30) == "succeeded"
        finally:
            try:
                p.join(timeout=1)
            except Exception:
                if p.is_alive():
                    p.terminate()

    @staticmethod
    def worker_lock(td, q):
        assert td.is_locked
        for val in td.values(True):
            if is_tensor_collection(val):
                assert val.is_locked
                assert val._lock_parents_weakrefs
        assert not td._lock_parents_weakrefs
        q.put("succeeded")

    @staticmethod
    def write_pid(x):
        return TensorDict({"pid": os.getpid()}, []).expand(x.shape)


@pytest.fixture(scope="class")
def _pool_fixt():
    with mp.get_context(mp_ctx).Pool(10) as pool:
        yield pool


@pytest.mark.skipif(not _has_funcdim, reason="functorch.dim could not be found")
class TestFCD(TestTensorDictsBase):
    """Test stack for first-class dimension."""

    @pytest.mark.parametrize("td_name,device", TestTensorDictsBase.TYPES_DEVICES)
    def test_fcd(self, td_name, device):
        td = getattr(self, td_name)(device)
        d0 = ftdim.dims(1)
        if isinstance(td, LazyStackedTensorDict) and td.stack_dim == 0:
            with pytest.raises(ValueError, match="Cannot index"):
                td[d0]
        elif td_name == "memmap_td":
            with pytest.raises(
                ValueError,
                match="Using first class dimension indices with MemoryMappedTensor",
            ):
                td[d0]
        else:
            assert td[d0].shape == td.shape[1:]
        d0, d1 = ftdim.dims(2)
        if isinstance(td, LazyStackedTensorDict) and td.stack_dim in (0, 1):
            with pytest.raises(ValueError, match="Cannot index"):
                td[d0, d1]
        elif td_name == "memmap_td":
            with pytest.raises(
                ValueError,
                match="Using first class dimension indices with MemoryMappedTensor",
            ):
                td[d0, d1]
        else:
            assert td[d0, d1].shape == td.shape[2:]
        d0, d1, d2 = ftdim.dims(3)
        if isinstance(td, LazyStackedTensorDict) and td.stack_dim in (0, 1, 2):
            with pytest.raises(ValueError, match="Cannot index"):
                td[d0, d1, d2]
        elif td_name == "memmap_td":
            with pytest.raises(
                ValueError,
                match="Using first class dimension indices with MemoryMappedTensor",
            ):
                td[d0, d1, d2]
        else:
            assert td[d0, d1, d2].shape == td.shape[3:]
        d0 = ftdim.dims(1)
        if isinstance(td, LazyStackedTensorDict) and td.stack_dim == 1:
            with pytest.raises(ValueError, match="Cannot index"):
                td[:, d0]
        elif td_name == "memmap_td":
            with pytest.raises(
                ValueError,
                match="Using first class dimension indices with MemoryMappedTensor",
            ):
                td[:, d0]
        else:
            assert td[:, d0].shape == torch.Size((td.shape[0], *td.shape[2:]))

    @pytest.mark.parametrize("td_name,device", TestTensorDictsBase.TYPES_DEVICES_NOLAZY)
    def test_fcd_names(self, td_name, device):
        td = getattr(self, td_name)(device)
        td.names = ["a", "b", "c", "d"]
        d0 = ftdim.dims(1)
        if isinstance(td, LazyStackedTensorDict) and td.stack_dim == 0:
            with pytest.raises(ValueError, match="Cannot index"):
                td[d0]
        elif td_name == "memmap_td":
            with pytest.raises(
                ValueError,
                match="Using first class dimension indices with MemoryMappedTensor",
            ):
                td[d0]
        else:
            assert td[d0].names == ["b", "c", "d"]
        d0, d1 = ftdim.dims(2)
        if isinstance(td, LazyStackedTensorDict) and td.stack_dim in (0, 1):
            with pytest.raises(ValueError, match="Cannot index"):
                td[d0, d1]
        elif td_name == "memmap_td":
            with pytest.raises(
                ValueError,
                match="Using first class dimension indices with MemoryMappedTensor",
            ):
                td[d0, d1]
        else:
            assert td[d0, d1].names == ["c", "d"]
        d0, d1, d2 = ftdim.dims(3)
        if isinstance(td, LazyStackedTensorDict) and td.stack_dim in (0, 1, 2):
            with pytest.raises(ValueError, match="Cannot index"):
                td[d0, d1, d2]
        elif td_name == "memmap_td":
            with pytest.raises(
                ValueError,
                match="Using first class dimension indices with MemoryMappedTensor",
            ):
                td[d0, d1, d2]
        else:
            assert td[d0, d1, d2].names == ["d"]
        d0 = ftdim.dims(1)
        if isinstance(td, LazyStackedTensorDict) and td.stack_dim == 1:
            with pytest.raises(ValueError, match="Cannot index"):
                td[:, d0]
        elif td_name == "memmap_td":
            with pytest.raises(
                ValueError,
                match="Using first class dimension indices with MemoryMappedTensor",
            ):
                td[:, d0]
        else:
            assert td[:, d0].names == ["a", "c", "d"]

    @pytest.mark.parametrize("as_module", [False, True])
    def test_modules(self, as_module):
        modules = [
            lambda: nn.Linear(3, 4),
            lambda: nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 4)),
            lambda: nn.Transformer(
                16,
                4,
                2,
                2,
                8,
                batch_first=True,
            ),
            lambda: nn.Sequential(nn.Conv2d(3, 4, 3), nn.Conv2d(4, 4, 3)),
        ]
        inputs = [
            lambda: (torch.randn(2, 3),),
            lambda: (torch.randn(2, 3),),
            lambda: (torch.randn(2, 3, 16), torch.randn(2, 3, 16)),
            lambda: (torch.randn(2, 3, 16, 16),),
        ]
        param_batch = 5
        for make_module, make_input in zip(modules, inputs):
            module = make_module()
            td = TensorDict.from_module(module, as_module=as_module)
            td = td.expand(param_batch).clone()
            d0 = ftdim.dims(1)
            td = TensorDictParams(td)[d0]
            td.to_module(module, preserve_module_state=False)
            y = module(*make_input())
            assert y.dims == (d0,)
            assert y._tensor.shape[0] == param_batch


@pytest.mark.slow
@pytest.mark.skipif(_IS_OSX, reason="Pool execution in osx can hang forever.")
class TestMap:
    """Tests for TensorDict.map that are independent from tensordict's type."""

    @staticmethod
    def _set_2(td):
        return td.set("2", 2)

    @classmethod
    def get_rand_incr(cls, td):
        # torch
        td["r"] = td["r"] + torch.randint(0, 100, ()).item()
        # numpy
        td["s"] = td["s"] + np.random.randint(0, 100, ()).item()
        return td

    def test_map_seed(self):
        pytest.skip(
            reason="Using max_tasks_per_child is unstable and can cause multiple processes to start over even though all jobs are completed",
        )
        gc.collect()

        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        td = TensorDict(
            {
                "r": torch.zeros(20, dtype=torch.int),
                "s": torch.zeros(20, dtype=torch.int),
                "c": torch.arange(20),
            },
            batch_size=[20],
        )
        generator = torch.Generator()
        # we use 4 workers with max 5 items each,
        # making sure that no worker does more than any other.
        generator.manual_seed(0)
        td_out_0 = td.map(
            TestMap.get_rand_incr,
            num_workers=4,
            generator=generator,
            chunksize=1,
            max_tasks_per_child=5,
        )
        generator.manual_seed(0)
        td_out_1 = td.map(
            TestMap.get_rand_incr,
            num_workers=4,
            generator=generator,
            chunksize=1,
            max_tasks_per_child=5,
        )
        # we cannot know which worker picks which job, but since they will all have
        # a seed from 0 to 4 and produce 1 number each, we can chekc that
        # those numbers are exactly what we were expecting.
        assert (td_out_0["r"].sort().values == td_out_1["r"].sort().values).all(), (
            td_out_0["r"].sort().values,
            td_out_1["r"].sort().values,
        )
        assert (td_out_0["s"].sort().values == td_out_1["s"].sort().values).all(), (
            td_out_0["s"].sort().values,
            td_out_1["s"].sort().values,
        )

    def test_map_seed_single(self):
        gc.collect()
        # A cheap version of the previous test
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        td = TensorDict(
            {
                "r": torch.zeros(20, dtype=torch.int),
                "s": torch.zeros(20, dtype=torch.int),
                "c": torch.arange(20),
            },
            batch_size=[20],
        )
        generator = torch.Generator()
        # we use 4 workers with max 5 items each,
        # making sure that no worker does more than any other.
        generator.manual_seed(0)
        td_out_0 = td.map(
            TestMap.get_rand_incr,
            num_workers=1,
            generator=generator,
            chunksize=1,
        )
        generator.manual_seed(0)
        td_out_1 = td.map(
            TestMap.get_rand_incr,
            num_workers=1,
            generator=generator,
            chunksize=1,
        )
        # we cannot know which worker picks which job, but since they will all have
        # a seed from 0 to 4 and produce 1 number each, we can chekc that
        # those numbers are exactly what we were expecting.
        assert (td_out_0["r"].sort().values == td_out_1["r"].sort().values).all(), (
            td_out_0["r"].sort().values,
            td_out_1["r"].sort().values,
        )
        assert (td_out_0["s"].sort().values == td_out_1["s"].sort().values).all(), (
            td_out_0["s"].sort().values,
            td_out_1["s"].sort().values,
        )

    @pytest.mark.parametrize(
        "chunksize,num_chunks", [[0, None], [2, None], [None, 5], [None, 10]]
    )
    @pytest.mark.parametrize("h5", [False, True])
    @pytest.mark.parametrize("has_out", [False, True])
    def test_index_with_generator(self, chunksize, num_chunks, h5, has_out, tmpdir):
        if h5 and not _has_h5py:
            pytest.skip("h5py not installed")
        gc.collect()
        input = TensorDict({"a": torch.arange(10), "b": torch.arange(10)}, [10])
        if h5:
            tmpdir = pathlib.Path(tmpdir)
            input_h5 = input.to_h5(tmpdir / "file.h5")
            assert input.shape == input_h5.shape
            input = input_h5
        if has_out:
            output_generator = torch.zeros_like(
                self.selectfn(input.to_tensordict(retain_none=True))
            )
            output_split = torch.zeros_like(
                self.selectfn(input.to_tensordict(retain_none=True))
            )
        else:
            output_generator = None
            output_split = None
        with mp.get_context(mp_ctx).Pool(2) as pool:
            output_generator = input.map(
                self.selectfn,
                num_workers=2,
                index_with_generator=True,
                num_chunks=num_chunks,
                chunksize=chunksize,
                out=output_generator,
                pool=pool,
            )
            output_split = input.map(
                self.selectfn,
                num_workers=2,
                index_with_generator=True,
                num_chunks=num_chunks,
                chunksize=chunksize,
                out=output_split,
                pool=pool,
            )
        assert (output_generator == output_split).all()

    def test_map_unbind(self):
        gc.collect()
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        td0 = TensorDict({"0": 0}, [])
        td1 = TensorDict({"1": 1}, [])
        td = LazyStackedTensorDict.lazy_stack([td0, td1], 0)
        td_out = td.map(self._set_2, chunksize=0, num_workers=4)
        assert td_out[0]["0"] == 0
        assert td_out[1]["1"] == 1
        assert (td_out["2"] == 2).all()

    @staticmethod
    def _assert_is_memmap(data):
        assert isinstance(data["tensor"], MemoryMappedTensor)

    @pytest.mark.parametrize("chunksize", [0, 5])
    def test_map_inplace(self, chunksize):
        gc.collect()
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")
        # Tests that we can return None values
        # Also tests that MemoryMapped id is kept using multiprocessing
        data = TensorDict({"tensor": torch.zeros(10)}, [10]).memmap_()
        data.map(self._assert_is_memmap, chunksize=chunksize, num_workers=2)

    @staticmethod
    def selectfn(input):
        return input.select("a")

    @pytest.mark.parametrize("chunksize", [0, 5])
    @pytest.mark.parametrize("mmap", [True, False])
    @pytest.mark.parametrize("start_method", [None, mp_ctx])
    def test_map_with_out(self, mmap, chunksize, tmpdir, start_method):
        gc.collect()
        tmpdir = Path(tmpdir)
        input = TensorDict({"a": torch.arange(10), "b": torch.arange(10)}, [10])
        if mmap:
            input.memmap_(tmpdir / "input")
        out = TensorDict({"a": torch.zeros(10, dtype=torch.int)}, [10])
        if mmap:
            out.memmap_(tmpdir / "output")
        input.map(
            self.selectfn,
            num_workers=2,
            chunksize=chunksize,
            out=out,
            mp_start_method=start_method,
        )
        assert (out["a"] == torch.arange(10)).all(), (chunksize, mmap)

    @classmethod
    def nontensor_check(cls, td):
        td["check"] = td["non_tensor"] == (
            "a string!" if (td["tensor"] % 2) == 0 else "another string!"
        )
        return td

    def test_map_non_tensor(self):
        gc.collect()
        # with NonTensorStack
        td = TensorDict(
            {"tensor": torch.arange(10), "non_tensor": "a string!"}, batch_size=[10]
        )
        td[1::2] = TensorDict({"non_tensor": "another string!"}, [5])
        td = td.map(self.nontensor_check, chunksize=0)
        assert td["check"].all()
        # with NonTensorData
        td = TensorDict(
            {"tensor": torch.zeros(10, dtype=torch.int), "non_tensor": "a string!"},
            batch_size=[10],
        )
        td = td.map(self.nontensor_check, chunksize=0)
        assert td["check"].all()

    @staticmethod
    def _return_identical(td):
        return td.clone()

    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize(
        "chunksize,num_chunks", [[0, None], [11, None], [None, 11]]
    )
    def test_map_iter(self, chunksize, num_chunks, shuffle):
        gc.collect()
        torch.manual_seed(0)
        td = TensorDict(
            {
                "a": torch.arange(100),
                "b": {
                    "c": torch.arange(100, 200),
                    "d": NonTensorStack(*[NonTensorData(str(i)) for i in range(100)]),
                },
            },
            batch_size=[100],
        )
        strings = set()
        a_elts = set()
        c_elts = set()
        data_prev = None
        for data in td.map_iter(
            self._return_identical,
            shuffle=shuffle,
            num_chunks=num_chunks,
            chunksize=chunksize,
        ):
            if data_prev is not None and data.shape == data_prev.shape:
                if shuffle:
                    assert (data != data_prev + data.numel()).any()
                else:
                    assert (
                        data.filter_non_tensor_data()
                        == data_prev.filter_non_tensor_data() + data.numel()
                    ).all()
            d = data["b", "d"]
            if not isinstance(d, str):
                strings.update(d)
                a_elts.update(data["a"].tolist())
                c_elts.update(data["b", "c"].tolist())
            else:
                strings.add(d)
                a_elts.add(data["a"].item())
                c_elts.add(data["b", "c"].item())
            data_prev = data

        assert a_elts == set(range(100))
        assert c_elts == set(range(100, 200))
        assert strings == {str(i) for i in range(100)}

    @pytest.mark.parametrize("shuffle", [False, True])
    @pytest.mark.parametrize(
        "chunksize,num_chunks", [[0, None], [11, None], [None, 11]]
    )
    def test_map_iter_interrupt_early(self, chunksize, num_chunks, shuffle):
        gc.collect()
        torch.manual_seed(0)
        td = TensorDict(
            {
                "a": torch.arange(100),
                "b": {
                    "c": torch.arange(100, 200),
                    "d": NonTensorStack(*[NonTensorData(str(i)) for i in range(100)]),
                },
            },
            batch_size=[100],
        )
        for _ in td.map_iter(
            self._return_identical,
            shuffle=shuffle,
            num_chunks=num_chunks,
            chunksize=chunksize,
        ):
            return


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
