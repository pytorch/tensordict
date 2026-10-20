# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import argparse
import os
import sys
from datetime import timedelta

import pytest
import torch
from _pytest.fixtures import fixture
from _utils_internal import is_npu_available
from tensordict import LazyStackedTensorDict, MemoryMappedTensor, TensorDict
from tensordict.utils import logger as tdlogger
from torch import distributed as dist, multiprocessing as mp, nn
from torch.distributed._tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    # init_device_mesh,
    Shard,
)
from torch.distributed.fsdp import (
    # FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    # StateDictType,
)

TIMEOUT = 100


@fixture
def set_context():
    try:
        mp.set_start_method("spawn")
    except Exception:
        tdlogger.info("context already set")


@pytest.mark.skipif(
    not torch.cuda.device_count() >= 2, reason="not enough cuda devices"
)
class TestFSDP:
    class MyDModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)
            self.fc2 = nn.Linear(8, 8, bias=False)
            self.relu = nn.ReLU()
            for p in self.parameters():
                p.data.fill_(1.0)

        def forward(self, input):
            return self.relu(self.fc1(input) + self.fc2(input))

    @classmethod
    def make_module(cls, device=None):
        with (
            torch.device(f"cuda:{device}")
            if device is not None
            else torch.device("cuda")
        ):
            my_module = cls.MyDModule()
            my_sharded_module = FSDP(my_module, device_id=device)
        return my_sharded_module

    @classmethod
    def worker(cls, rank, path):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "10017"

        torch.distributed.init_process_group(
            "nccl",
            rank=rank,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        torch.cuda.set_device(rank)
        module = cls.make_module(rank)
        dist.barrier()
        # cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # with FSDP.state_dict_type(module, StateDictType.SHARDED_STATE_DICT): #, cfg):
        #     tdlogger.info(module.state_dict())

        # td = TensorDict(module.state_dict(), []).unflatten_keys(".")
        td = TensorDict.from_module(module, use_state_dict=True)
        if rank == 0:
            td.memmap(path)
        dist.destroy_process_group()

    def test_fsdp_module(self, tmpdir):
        try:
            mp.set_start_method("spawn")
        except Exception:
            tdlogger.info("start method already set to %s", mp.get_start_method())
        proc0 = mp.Process(target=self.worker, args=(0, tmpdir))
        proc1 = mp.Process(target=self.worker, args=(1, tmpdir))
        proc0.start()
        proc1.start()
        proc0.join(timeout=TIMEOUT)
        proc1.join(timeout=TIMEOUT)
        assert (TensorDict.load_memmap(tmpdir) == 1).all()


@pytest.mark.skipif(
    not is_npu_available() or not torch.npu.device_count() > 2,
    reason="not enough npu devices",
)
class TestNPUFSDP:
    class MyDModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)
            self.fc2 = nn.Linear(8, 8, bias=False)
            self.relu = nn.ReLU()
            for p in self.parameters():
                p.data.fill_(1.0)

        def forward(self, input):
            return self.relu(self.fc1(input) + self.fc2(input))

    @classmethod
    def make_module(cls, device=None):
        with (
            torch.device(f"npu:{device}") if device is not None else torch.device("npu")
        ):
            my_module = cls.MyDModule()
            my_sharded_module = FSDP(my_module, device_id=device)
        return my_sharded_module

    @classmethod
    def worker(cls, rank, path):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "10017"

        torch.distributed.init_process_group(
            "hccl",
            rank=rank,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        torch.npu.set_device(rank)
        module = cls.make_module(rank)
        dist.barrier()
        # cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        # with FSDP.state_dict_type(module, StateDictType.SHARDED_STATE_DICT): #, cfg):
        #     tdlogger.info(module.state_dict())

        # td = TensorDict(module.state_dict(), []).unflatten_keys(".")
        td = TensorDict.from_module(module, use_state_dict=True)
        if rank == 0:
            td.memmap(path)
        dist.destroy_process_group()

    def test_fsdp_module(self, tmpdir):
        try:
            mp.set_start_method("spawn")
        except Exception:
            tdlogger.info("start method already set to %s", mp.get_start_method())
        proc0 = mp.Process(target=self.worker, args=(0, tmpdir))
        proc1 = mp.Process(target=self.worker, args=(1, tmpdir))
        proc0.start()
        proc1.start()
        proc0.join(timeout=TIMEOUT)
        proc1.join(timeout=TIMEOUT)
        assert (TensorDict.load_memmap(tmpdir) == 1).all()


class TestDTensor:
    class MyDModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8, bias=False)
            self.fc2 = nn.Linear(8, 8, bias=False)
            self.relu = nn.ReLU()

        def forward(self, input):
            return self.relu(self.fc1(input) + self.fc2(input))

    @classmethod
    def device(cls):
        return "cpu"

    @classmethod
    def _make_tensordict(cls):
        module = cls.MyDModule()
        mesh = DeviceMesh(cls.device(), torch.arange(2))

        def shard_params(mod_name, mod, mesh):
            col_linear_placement = [Shard(0)]
            # shard fc1 and fc2
            if isinstance(mod, nn.Linear):
                for name, param in mod.named_parameters():
                    dist_param = nn.Parameter(
                        distribute_tensor(param, mesh, col_linear_placement)
                    )
                    mod.register_parameter(name, dist_param)

        sharded_module = distribute_module(module, mesh, partition_fn=shard_params)

        return TensorDict.from_module(sharded_module)

    @classmethod
    def worker(cls, rank, queue):
        torch.distributed.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls._make_tensordict()
        if rank == 0:
            tdmemmap = td.memmap()  # noqa: F841
            # for key, val in tdmemmap.items(True, True):
            #     tdlogger.info(key, val)
            queue.put("memmaped")
        else:
            # TODO: we need this bit to call the gather on each worker
            # but we don't want each worker to write a memmap!
            td.apply(lambda t: t.full_tensor())
            queue.put("worker")

    def test_dtensor(self, tmp_path):
        try:
            mp.set_start_method("spawn")
        except Exception:
            tdlogger.info("start method already set to %s", mp.get_start_method())
        server_queue = mp.Queue(1)
        client_queue = mp.Queue(1)
        server_worker = mp.Process(
            target=self.worker,
            args=(
                0,
                server_queue,
            ),
        )
        client_worker = mp.Process(
            target=self.worker,
            args=(
                1,
                client_queue,
            ),
        )

        server_worker.start()
        client_worker.start()
        try:
            assert server_queue.get(timeout=TIMEOUT) == "memmaped"
            assert client_queue.get(timeout=TIMEOUT) == "worker"
        finally:
            server_worker.join()
            client_worker.join()


class TestGather:
    @staticmethod
    def client(memmap_filename):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )

        td = TensorDict(
            {
                ("a", "b"): torch.randn(2),
                "c": torch.randn(2),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(
                    torch.randn(2, 2), filename=memmap_filename
                ),
            },
            [2],
        )
        td.gather_and_stack(0)

    @staticmethod
    def server(queue):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )

        td = (
            TensorDict(
                {
                    ("a", "b"): torch.zeros(2),
                    "c": torch.zeros(2),
                    ("d", "e", "f"): MemoryMappedTensor.from_tensor(torch.zeros(2, 2)),
                },
                [2],
            )
            .expand(1, 2)
            .contiguous()
        )
        td.gather_and_stack(0)
        assert (td != 0).all()
        queue.put("yuppie")

    def test_gather(self, set_context, tmp_path):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(
            target=type(self).client, args=(str(tmp_path / "sub"),)
        )

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


@pytest.mark.skipif(
    sys.version_info.minor <= 7,
    reason="reduce test is incompatible with python 3.7 or lower (cannot pickle the op Enum).",
)
class TestReduce:
    @staticmethod
    def client(memmap_filename, rank, op, async_op, return_premature):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=3,
        )

        td = TensorDict(
            {
                ("a", "b"): torch.ones(2),
                "c": torch.ones(2),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(
                    torch.ones(2, 2), filename=memmap_filename
                ),
            },
            [2],
        )
        td.reduce(0, op=op, async_op=async_op, return_premature=False)

    @staticmethod
    def server(queue, op, async_op, return_premature):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=3,
        )

        td = (
            TensorDict(
                {
                    ("a", "b"): torch.ones(2),
                    "c": torch.ones(2),
                    ("d", "e", "f"): MemoryMappedTensor.from_tensor(torch.ones(2, 2)),
                },
                [2],
            )
            .expand(1, 2)
            .contiguous()
        )
        out = td.reduce(0, op=op, async_op=async_op, return_premature=return_premature)
        if not async_op:
            assert out is None
        elif return_premature:
            for _out in out:
                tdlogger.info("waiting...")
                _out.wait()
                tdlogger.info("done")
        else:
            assert out is None
        if op == dist.ReduceOp.SUM:
            assert (td == 3).all()
        elif op == dist.ReduceOp.PRODUCT:
            assert (td == 1).all()

        queue.put("yuppie")

    @pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.PRODUCT])
    @pytest.mark.parametrize(
        "async_op,return_premature", [[True, True], [False, False], [True, False]]
    )
    def test_gather(self, set_context, tmp_path, op, async_op, return_premature):
        queue = mp.Queue(1)
        main_worker = mp.Process(
            target=type(self).server, args=(queue, op, async_op, return_premature)
        )
        secondary_worker = mp.Process(
            target=type(self).client,
            args=(str(tmp_path / "sub1"), 1, op, async_op, return_premature),
        )
        tertiary_worker = mp.Process(
            target=type(self).client,
            args=(str(tmp_path / "sub"), 2, op, async_op, return_premature),
        )

        main_worker.start()
        secondary_worker.start()
        tertiary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            tertiary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"

    @staticmethod
    def group_worker(queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29530"
        dist.init_process_group(
            "gloo", rank=rank, world_size=3, timeout=timedelta(seconds=30)
        )
        # Rank 2 is outside the group and does not take part in the reduction.
        group = dist.new_group([0, 1], timeout=timedelta(seconds=30))
        if rank in (0, 1):
            td = TensorDict({("a", "b"): torch.ones(2), "c": torch.ones(2)}, [2])
            td.reduce(0, group=group)
            if rank == 0:
                queue.put((td["a", "b"].tolist(), td["c"].tolist()))
        dist.barrier()
        dist.destroy_process_group()

    def test_reduce_group(self, set_context):
        queue = mp.Queue(1)
        workers = [
            mp.Process(target=type(self).group_worker, args=(queue, rank))
            for rank in range(3)
        ]
        for w in workers:
            w.start()
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            for w in workers:
                w.join(timeout=TIMEOUT)
        assert out == ([2.0, 2.0], [2.0, 2.0])

    @staticmethod
    def lazy_worker(queue, rank, world_size, op, use_group):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29531"
        dist.init_process_group(
            "gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
        )
        # With use_group, rank 2 is outside the group and does not take part.
        group = (
            dist.new_group([0, 1], timeout=timedelta(seconds=30)) if use_group else None
        )
        if rank in (0, 1):
            v = float(rank + 1)
            root = LazyStackedTensorDict(
                TensorDict(a=torch.full((3,), v)), TensorDict(a=torch.full((3,), v))
            )
            nested = TensorDict(
                b=torch.full((2,), v),
                s=LazyStackedTensorDict(
                    TensorDict(a=torch.full((3,), v)),
                    TensorDict(a=torch.full((3,), v)),
                ),
                batch_size=[2],
            )
            root.reduce(0, op=op, group=group)
            nested.reduce(0, op=op, group=group)
            if rank == 0:
                queue.put(
                    (
                        root["a"].unique().tolist(),
                        nested["b"].unique().tolist(),
                        nested["s", "a"].unique().tolist(),
                    )
                )
        dist.barrier()
        dist.destroy_process_group()

    @pytest.mark.parametrize("use_group", [False, True])
    @pytest.mark.parametrize(
        "op,expected", [[dist.ReduceOp.SUM, 3.0], [dist.ReduceOp.MAX, 2.0]]
    )
    def test_reduce_lazy_stack(self, set_context, op, expected, use_group):
        world_size = 3 if use_group else 2
        queue = mp.Queue(1)
        workers = [
            mp.Process(
                target=type(self).lazy_worker,
                args=(queue, rank, world_size, op, use_group),
            )
            for rank in range(world_size)
        ]
        for w in workers:
            w.start()
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            for w in workers:
                w.join(timeout=TIMEOUT)
                if w.is_alive():
                    w.terminate()
        assert out == ([expected], [expected], [expected])

    @staticmethod
    def mixed_worker(queue, rank, lazy_rank, return_premature):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29532"
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=30)
        )
        v = float(rank + 1)
        # One rank holds lazy stacks, the other dense tensordicts of the same
        # keys and shapes.
        if rank == lazy_rank:
            root = LazyStackedTensorDict(
                TensorDict(a=torch.full((3,), v)), TensorDict(a=torch.full((3,), v))
            )
        else:
            root = TensorDict(a=torch.full((2, 3), v), batch_size=[2])
        nested = TensorDict(b=torch.full((2,), v), s=root.clone(), batch_size=[2])
        for td in (root, nested):
            works = td.reduce(
                0, async_op=return_premature, return_premature=return_premature
            )
            for work in works or ():
                work.wait()
        if rank == 0:
            queue.put(
                (
                    root["a"].unique().tolist(),
                    nested["b"].unique().tolist(),
                    nested["s", "a"].unique().tolist(),
                )
            )
        dist.barrier()
        dist.destroy_process_group()

    @pytest.mark.parametrize("return_premature", [False, True])
    @pytest.mark.parametrize("lazy_rank", [0, 1])
    def test_reduce_lazy_stack_and_dense(
        self, set_context, lazy_rank, return_premature
    ):
        queue = mp.Queue(1)
        workers = [
            mp.Process(
                target=type(self).mixed_worker,
                args=(queue, rank, lazy_rank, return_premature),
            )
            for rank in range(2)
        ]
        for w in workers:
            w.start()
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            for w in workers:
                w.join(timeout=TIMEOUT)
                if w.is_alive():
                    w.terminate()
        assert out == ([3.0], [3.0], [3.0])

    @staticmethod
    def write_back_worker(queue, rank, mode, expanded):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29533"
        dist.init_process_group(
            "gloo", rank=rank, world_size=2, timeout=timedelta(seconds=30)
        )
        v = float(rank + 1)
        if expanded:
            # Rank 0 cannot write the result back into expanded leaves.
            members = [
                TensorDict(a=torch.full((1,), v).expand(3), b=torch.full((3,), v))
                for _ in range(2)
            ]
        else:
            # Parameters, as from TensorDict.from_modules(..., lazy_stack=True).
            members = [
                TensorDict(
                    a=nn.Parameter(torch.full((3,), v)),
                    b=nn.Parameter(torch.full((3,), v)),
                )
                for _ in range(2)
            ]
        td = TensorDict(
            s=LazyStackedTensorDict(*members), z=torch.full((2,), v), batch_size=[2]
        )
        error = None
        try:
            works = td.reduce(
                0, async_op=mode != "sync", return_premature=mode == "premature"
            )
        except Exception as err:
            error = type(err).__name__
        else:
            # Wait for every work, even after one raises.
            for work in works or ():
                try:
                    work.wait()
                except Exception as err:
                    error = error or type(err).__name__
        result = (
            error,
            td["s", "b"].unique().tolist(),
            td["s", "b"].requires_grad,
            td["z"].unique().tolist(),
        )
        if rank == 0:
            queue.put(result)
        else:
            # The other rank issues and finishes every collective.
            assert error is None, error
        dist.barrier()
        dist.destroy_process_group()

    @pytest.mark.parametrize("mode", ["sync", "async", "premature"])
    def test_reduce_lazy_stack_requires_grad(self, set_context, mode):
        assert self._run_write_back(mode, expanded=False) == (
            None,
            [3.0],
            True,
            [3.0],
        )

    @pytest.mark.parametrize("mode", ["sync", "async", "premature"])
    def test_reduce_lazy_stack_write_back_error(self, set_context, mode):
        error, b, _, z = self._run_write_back(mode, expanded=True)
        # Rank 0 raises once every reduction and write-back has run, so the
        # other leaves are reduced and the other rank does not hang.
        assert error is not None
        assert b == [3.0]
        assert z == [3.0]

    @classmethod
    def _run_write_back(cls, mode, expanded):
        queue = mp.Queue(1)
        workers = [
            mp.Process(target=cls.write_back_worker, args=(queue, rank, mode, expanded))
            for rank in range(2)
        ]
        for w in workers:
            w.start()
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            for w in workers:
                w.join(timeout=TIMEOUT)
                if w.is_alive():
                    w.terminate()
        assert all(w.exitcode == 0 for w in workers)
        return out


# =========================================
# Test td.send
# ------------


class SendBase:
    @staticmethod
    @abc.abstractmethod
    def make_td(ones):
        raise NotImplementedError

    @classmethod
    def client(cls, pseudo_rand, group):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        if group is not None:
            group = dist.new_group(group)
        td = cls.make_td(ones=True)
        td.send(0, pseudo_rand=pseudo_rand, group=group)

    @classmethod
    def server(cls, queue, pseudo_rand, group):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        if group is not None:
            group = dist.new_group(group)
        td = cls.make_td(ones=False)
        td.recv(1, pseudo_rand=pseudo_rand, group=group)
        assert (td == 1).all()
        queue.put("yuppie")

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.parametrize("group", [[0, 1], None])
    @pytest.mark.parametrize("pseudo_rand", [True, False])
    def test_send(self, pseudo_rand, group, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=self.server, args=(queue, pseudo_rand, group))
        secondary_worker = mp.Process(target=self.client, args=(pseudo_rand, group))

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


class TestSend(SendBase):
    """Test send for tensordict as root."""

    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td["_"] = fun(2, 1, 5)
        return td


class TestSendLazyStackRoot(SendBase):
    """Test send for lazy-stack as root."""

    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td["_"] = fun(2, 1, 5)
        td1 = td
        td2 = td.clone()
        td2["c"] = fun(2, 3, 2)
        td2["g"] = fun(2, 1, 5)
        td = LazyStackedTensorDict.maybe_dense_stack([td1, td2], 0)
        return td


class TestSendLazyStackNest(SendBase):
    """Test send for tensordict as root with lazy stacked field."""

    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td["_"] = fun(2, 1, 5)
        td1 = td
        td2 = td.clone()
        td2["c"] = fun(2, 3, 2)
        td2["g"] = fun(2, 1, 5)
        td = TensorDict(
            {"ls": LazyStackedTensorDict.maybe_dense_stack([td1, td2], 0)}, [2, 2]
        )
        return td


# =========================================
# Test td.irecv
# -------------


class iRecvBase:
    @staticmethod
    @abc.abstractmethod
    def make_td(ones):
        raise NotImplementedError

    @classmethod
    def client(cls, pseudo_rand, group):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls.make_td(ones=True)
        if group is not None:
            group = dist.new_group(group)
        td.send(0, pseudo_rand=pseudo_rand, group=group)

    @classmethod
    def server(cls, queue, return_premature, pseudo_rand, group):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls.make_td(ones=False)
        if group is not None:
            group = dist.new_group(group)
        out = td.irecv(
            1, return_premature=return_premature, pseudo_rand=pseudo_rand, group=group
        )
        if return_premature:
            for fut in out:
                fut.wait()
        assert (td == 1).all()
        queue.put("yuppie")

    @pytest.mark.parametrize("group", [None, [0, 1]])
    @pytest.mark.parametrize("pseudo_rand", [True, False])
    @pytest.mark.parametrize("return_premature", [True, False])
    def test_irecv(self, pseudo_rand, return_premature, set_context, group):
        queue = mp.Queue(1)
        main_worker = mp.Process(
            target=type(self).server, args=(queue, return_premature, pseudo_rand, group)
        )
        secondary_worker = mp.Process(
            target=type(self).client, args=(pseudo_rand, group)
        )

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


class TestiRecv(iRecvBase):
    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                "_": fun(2, 1, 5),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        return td


class TestiRecvLazyStackRoot(iRecvBase):
    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td1 = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                "_": fun(2, 1, 5),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td2 = td1.clone()
        td2["c"] = td2["c"].unsqueeze(-1).expand(2, 3, 10).contiguous()
        td2["g"] = td1["c"].clone()
        td = LazyStackedTensorDict.maybe_dense_stack([td1, td2], 0)
        return td


class TestiRecvLazyStackNest(iRecvBase):
    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td1 = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                "_": fun(2, 1, 5),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td2 = td1.clone()
        td2["c"] = td2["c"].unsqueeze(-1).expand(2, 3, 10).contiguous()
        td2["g"] = td1["c"].clone()
        td = LazyStackedTensorDict.maybe_dense_stack([td1, td2], 0)
        td = TensorDict({"td": td}, [2, 2])
        return td


# =========================================
# Test td.isend
# -------------


class iSendBase:
    @staticmethod
    @abc.abstractmethod
    def make_td(ones):
        raise NotImplementedError

    @classmethod
    def client(cls, pseudo_rand, group):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )

        td = cls.make_td(True)
        if group is not None:
            group = dist.new_group(group)
        td.isend(0, pseudo_rand=pseudo_rand, group=group)

    @classmethod
    def server(cls, queue, pseudo_rand, group):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls.make_td(False)
        if group is not None:
            group = dist.new_group(group)
        td.recv(1, pseudo_rand=pseudo_rand, group=group)
        assert (td == 1).all()
        queue.put("yuppie")

    @pytest.mark.parametrize("group", [[0, 1], None])
    @pytest.mark.parametrize("pseudo_rand", [True, False])
    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_isend(self, pseudo_rand, set_context, group):
        queue = mp.Queue(1)
        main_worker = mp.Process(
            target=type(self).server, args=(queue, pseudo_rand, group)
        )
        secondary_worker = mp.Process(
            target=type(self).client, args=(pseudo_rand, group)
        )

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "yuppie"
        except Exception as err:
            # otherwise pytest does not capture it
            raise err
        finally:
            main_worker.join()
            secondary_worker.join()


class TestiSend(iSendBase):
    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                "_": fun(2, 1, 5),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        return td


class TestiSendLazyStackRoot(iSendBase):
    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td1 = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                "_": fun(2, 1, 5),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td2 = td1.clone()
        td2["c"] = td2["c"].unsqueeze(-1).expand(2, 3, 10).contiguous()
        td2["g"] = td1["c"].clone()
        td = LazyStackedTensorDict.maybe_dense_stack([td1, td2], 0)
        return td


class TestiSendLazyStackNest(iSendBase):
    @staticmethod
    def make_td(ones):
        if ones:
            fun = torch.ones
        else:
            fun = torch.zeros
        td1 = TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                "_": fun(2, 1, 5),
                ("d", "e", "f"): MemoryMappedTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td2 = td1.clone()
        td2["c"] = td2["c"].unsqueeze(-1).expand(2, 3, 10).contiguous()
        td2["g"] = td1["c"].clone()
        td = LazyStackedTensorDict.maybe_dense_stack([td1, td2], 0)
        td = TensorDict({"td": td}, [2, 2])
        return td


class TestInitRemote:
    port = "29505"

    @classmethod
    def client(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
        )

        td = TensorDict.from_remote_init(src=0)
        assert set(td.keys()) == {"a", "c", "d"}
        assert (td == 1).all()
        assert td.is_consolidated()
        queue.put("yuppie")

    @classmethod
    def server(cls, queue):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
        )

        td = (
            TensorDict(
                {
                    ("a", "b"): torch.ones(2),
                    "c": torch.ones(2),
                    ("d", "e", "f"): MemoryMappedTensor.from_tensor(torch.ones(2, 2)),
                },
                [2],
            )
            .expand(1, 2)
            .contiguous()
        )
        td.init_remote(dst=1)

    def test_init_remote(self, set_context, tmp_path):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client, args=(queue, 1))

        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestInitRemoteNonTensor:
    port = "29506"

    @classmethod
    def client(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
        )

        td = TensorDict.from_remote_init(src=0)
        assert set(td.keys()) == {"a", "label", "c"}
        assert (td["a"] == 1).all()
        assert (td["c"] == 2).all()
        assert td["label"] == "hello"
        assert td.is_consolidated()
        queue.put("yuppie")

    @classmethod
    def server(cls, queue):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
        )

        td = TensorDict(
            {
                "a": torch.ones(3),
                "label": "hello",
                "c": torch.full((3, 2), 2.0),
            },
            [3],
        )
        td.init_remote(dst=1)

    def test_init_remote_non_tensor(self, set_context, tmp_path):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client, args=(queue, 1))

        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestInitRemoteBroadcast:
    port = "29512"

    @classmethod
    def client(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
        )

        td = TensorDict.from_remote_init(src=0, use_broadcast=True)
        assert set(td.keys()) == {"a", "c", "d"}
        assert (td == 1).all()
        assert td.is_consolidated()
        queue.put("yuppie")

    @classmethod
    def server(cls, queue):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
        )

        td = (
            TensorDict(
                {
                    ("a", "b"): torch.ones(2),
                    "c": torch.ones(2),
                    ("d", "e", "f"): MemoryMappedTensor.from_tensor(torch.ones(2, 2)),
                },
                [2],
            )
            .expand(1, 2)
            .contiguous()
        )
        td.init_remote(use_broadcast=True)

    def test_init_remote_broadcast(self, set_context, tmp_path):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client, args=(queue, 1))

        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestAllGather:
    port = "29510"

    @classmethod
    def worker(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
        )

        td = TensorDict(
            {
                "a": torch.full((3,), float(rank)),
                "b": torch.full((3, 2), float(rank + 10)),
            },
            [3],
        )
        result = td.all_gather()
        assert len(result) == 2
        assert (result[0]["a"] == 0).all()
        assert (result[0]["b"] == 10).all()
        assert (result[1]["a"] == 1).all()
        assert (result[1]["b"] == 11).all()
        if rank == 0:
            queue.put("yuppie")

    def test_all_gather(self, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).worker, args=(queue, 0))
        secondary_worker = mp.Process(target=type(self).worker, args=(queue, 1))

        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestAllReduce:
    port = "29509"

    @classmethod
    def worker(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=3,
        )

        td = TensorDict(
            {
                ("a", "b"): torch.ones(2),
                "c": torch.ones(2, 3),
            },
            [2],
        )
        td.all_reduce(op=dist.ReduceOp.SUM)
        assert (td["a", "b"] == 3).all()
        assert (td["c"] == 3).all()
        if rank == 0:
            queue.put("yuppie")

    @pytest.mark.parametrize("op", [dist.ReduceOp.SUM, dist.ReduceOp.PRODUCT])
    def test_all_reduce(self, set_context, op):
        queue = mp.Queue(1)
        workers = []
        for rank in range(3):
            w = mp.Process(target=type(self).worker, args=(queue, rank))
            workers.append(w)
            w.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            for w in workers:
                w.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestBucketedReduce:
    port = "29587"

    @staticmethod
    def make_td(rank):
        generator = torch.Generator().manual_seed(rank)

        def rand(*shape, dtype=torch.float32):
            return torch.randint(1, 4, shape, generator=generator).to(dtype)

        return TensorDict(
            {
                "a": rand(2, 3),
                ("b", "c"): rand(2, dtype=torch.int64),
                ("b", "d", "e"): rand(2, 5, dtype=torch.float64),
                "f": rand(2, 1, dtype=torch.int32),
                "g": rand(2, 40),
                ("b", "h"): rand(2),
                "s": LazyStackedTensorDict(
                    TensorDict(x=rand(4)), TensorDict(x=rand(4))
                ),
            },
            batch_size=[2],
        )

    @classmethod
    def worker(cls, queue, rank, op):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo", rank=rank, world_size=3, timeout=timedelta(seconds=60)
        )

        def run(td, dst, mode, **kwargs):
            async_op = mode != "sync"
            if dst is None:
                works = td.all_reduce(op=op, async_op=async_op, **kwargs)
            else:
                works = td.reduce(
                    dst,
                    op=op,
                    async_op=async_op,
                    return_premature=mode == "premature",
                    **kwargs,
                )
            for work in works or ():
                work.wait()
            return works

        errors = []
        try:
            for bucket_bytes in (1, 64, 2**20):
                for mode in ("sync", "async", "premature"):
                    # dst None: all_reduce.
                    for dst in (None, 1):
                        case = (bucket_bytes, mode, dst)
                        expected = cls.make_td(rank)
                        run(expected, dst, mode)
                        td = cls.make_td(rank)
                        works = run(td, dst, mode, bucket_bytes=bucket_bytes)
                        if (dst is None or rank == dst) and not (td == expected).all():
                            errors.append((case, "result"))
                        if bucket_bytes == 2**20 and works is not None:
                            # One bucket per dtype.
                            if len(works) != 4:
                                errors.append((case, len(works)))
        except Exception as err:
            errors.append(repr(err))
        queue.put(errors)
        dist.barrier()
        dist.destroy_process_group()

    @pytest.mark.parametrize(
        "op", [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.PRODUCT]
    )
    def test_bucketed_equals_per_leaf(self, set_context, op):
        queue = mp.Queue(3)
        workers = [
            mp.Process(target=type(self).worker, args=(queue, rank, op))
            for rank in range(3)
        ]
        for w in workers:
            w.start()
        try:
            out = [queue.get(timeout=TIMEOUT) for _ in workers]
        finally:
            queue.close()
            for w in workers:
                w.join(timeout=TIMEOUT)
                if w.is_alive():
                    w.terminate()
        assert out == [[], [], []]


class TestBroadcast:
    port = "29508"

    @classmethod
    def worker(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
        )

        if rank == 0:
            td = TensorDict(
                {
                    ("a", "b"): torch.ones(2),
                    "c": torch.ones(2, 3),
                    ("d", "e", "f"): torch.ones(2, 2),
                },
                [2],
            )
        else:
            td = TensorDict({}, [])

        result = td.broadcast(src=0)
        assert set(result.keys()) == {"a", "c", "d"}
        assert (result == 1).all()
        assert result.is_consolidated()
        if rank == 1:
            queue.put("yuppie")

    def test_broadcast(self, set_context, tmp_path):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).worker, args=(queue, 0))
        secondary_worker = mp.Process(target=type(self).worker, args=(queue, 1))

        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestScatter:
    port = "29511"

    @classmethod
    def worker(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
        )

        tds = None
        if rank == 0:
            tds = [
                TensorDict({"a": torch.zeros(3), "b": torch.zeros(3, 2)}, [3]),
                TensorDict({"a": torch.ones(3), "b": torch.ones(3, 2)}, [3]),
            ]
        td = TensorDict({}, []).scatter(src=0, tensordicts=tds)
        assert (td["a"] == float(rank)).all()
        assert (td["b"] == float(rank)).all()
        if rank == 1:
            queue.put("yuppie")

    def test_scatter(self, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).worker, args=(queue, 0))
        secondary_worker = mp.Process(target=type(self).worker, args=(queue, 1))

        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestSendConsolidated:
    port = "29507"

    @classmethod
    def client(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=rank,
            world_size=2,
        )

        # Setup phase: receive schema + first payload
        td_recv = TensorDict.from_remote_init(src=0)
        assert td_recv.is_consolidated()

        # Steady-state: receive updated data into existing buffer
        td_recv.recv(src=0, consolidated=True)
        assert (td_recv["a", "b"] == 2).all()
        assert (td_recv["c"] == 2).all()
        queue.put("yuppie")

    @classmethod
    def server(cls, queue):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
        )

        td = TensorDict(
            {
                ("a", "b"): torch.ones(2),
                "c": torch.ones(2, 3),
            },
            [2],
        )

        # Setup phase: send schema + first payload
        td.init_remote(dst=1)

        # Update values and send again via consolidated path
        td["a", "b"] = torch.full((2,), 2.0)
        td["c"] = torch.full((2, 3), 2.0)
        td.send(dst=1, consolidated=True)

    @classmethod
    def client_async(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29525"
        dist.init_process_group("gloo", rank=rank, world_size=2)

        td_recv = TensorDict.from_remote_init(src=0)
        futures = td_recv.irecv(src=0, consolidated=True, return_premature=True)
        assert len(futures) == 1
        for fut in futures:
            fut.wait()
        assert (td_recv["a", "b"] == 2).all()
        assert (td_recv["c"] == 2).all()
        assert td_recv.irecv(src=0, consolidated=True) is None
        assert (td_recv["a", "b"] == 3).all()
        assert (td_recv["c"] == 3).all()
        queue.put("yuppie")

    @classmethod
    def server_async(cls, queue):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29525"
        dist.init_process_group("gloo", rank=0, world_size=2)

        td = TensorDict({("a", "b"): torch.ones(2), "c": torch.ones(2, 3)}, [2])
        td.init_remote(dst=1)

        # A non-consolidated sender is consolidated on the fly
        td["a", "b"] = torch.full((2,), 2.0)
        td["c"] = torch.full((2, 3), 2.0)
        works = td.isend(dst=1, consolidated=True, return_early=True)
        assert len(works) == 1
        for work in works:
            work.wait()
        # A consolidated sender is sent as is
        td_c = td.consolidate(metadata=True)
        td_c.update_({("a", "b"): torch.full((2,), 3.0), "c": torch.full((2, 3), 3.0)})
        assert td_c.isend(dst=1, consolidated=True) is None

    def test_isend_irecv_consolidated(self, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server_async, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client_async, args=(queue, 1))

        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"

    def test_send_consolidated(self, set_context, tmp_path):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client, args=(queue, 1))

        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


# ========================================================================
# Test group_dst / group_src over raw (unregistered) process groups
# ------------------------------------------------------------------------


def _make_raw_pair_group(rank, port, backend="gloo"):
    """Builds a standalone pair process group over a TCPStore.

    No ``init_process_group`` / ``new_group`` anywhere: the group never goes
    through the c10d registry, so global-rank lookups would fail on it.
    """
    store = dist.TCPStore("localhost", port, 2, rank == 0)
    prefix_store = dist.PrefixStore("raw_pair", store)
    if backend == "gloo":
        return dist.ProcessGroupGloo(prefix_store, rank, 2, timedelta(seconds=TIMEOUT))
    if backend == "nccl":
        return dist.ProcessGroupNCCL(prefix_store, rank, 2)
    raise ValueError(f"Unknown backend {backend}")


class TestRawGroupSendRecv:
    """p2p transfer over standalone (unregistered) groups via group_dst/group_src."""

    port = 29520

    @staticmethod
    def make_td(ones):
        fun = torch.ones if ones else torch.zeros
        return TensorDict(
            {
                ("a", "b"): fun(2),
                "c": fun(2, 3),
                "_": fun(2, 1, 5),
            },
            [2],
        )

    @classmethod
    def client(cls, mode, pseudo_rand):
        pg = _make_raw_pair_group(1, cls.port)
        td = cls.make_td(ones=True)
        if mode == "isend":
            td.isend(group=pg, group_dst=0, pseudo_rand=pseudo_rand)
        elif mode == "consolidated":
            td.send(group=pg, group_dst=0, consolidated=True)
        elif mode == "isend_consolidated":
            td.isend(group=pg, group_dst=0, consolidated=True)
        else:
            td.send(group=pg, group_dst=0, pseudo_rand=pseudo_rand)

    @classmethod
    def server(cls, queue, mode, pseudo_rand, return_premature):
        pg = _make_raw_pair_group(0, cls.port)
        td = cls.make_td(ones=False)
        if mode == "consolidated":
            td = td.consolidate(metadata=True)
            td.recv(group=pg, group_src=1, consolidated=True)
        elif mode == "irecv_consolidated":
            td = td.consolidate(metadata=True)
            out = td.irecv(
                group=pg,
                group_src=1,
                consolidated=True,
                return_premature=return_premature,
            )
            if return_premature:
                assert len(out) == 1
                for fut in out:
                    fut.wait()
        elif mode == "irecv":
            out = td.irecv(
                group=pg,
                group_src=1,
                pseudo_rand=pseudo_rand,
                return_premature=return_premature,
            )
            if return_premature:
                for fut in out:
                    fut.wait()
        else:
            td.recv(group=pg, group_src=1, pseudo_rand=pseudo_rand)
        assert (td == 1).all()
        queue.put("yuppie")

    def _run(self, client_mode, server_mode, pseudo_rand=False, return_premature=False):
        queue = mp.Queue(1)
        main_worker = mp.Process(
            target=type(self).server,
            args=(queue, server_mode, pseudo_rand, return_premature),
        )
        secondary_worker = mp.Process(
            target=type(self).client, args=(client_mode, pseudo_rand)
        )

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "yuppie"
        finally:
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.parametrize("pseudo_rand", [True, False])
    def test_send_recv_raw(self, pseudo_rand, set_context):
        self._run("send", "recv", pseudo_rand=pseudo_rand)

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.parametrize("pseudo_rand", [True, False])
    def test_isend_recv_raw(self, pseudo_rand, set_context):
        self._run("isend", "recv", pseudo_rand=pseudo_rand)

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.parametrize("return_premature", [True, False])
    def test_irecv_raw(self, return_premature, set_context):
        self._run("isend", "irecv", return_premature=return_premature)

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_consolidated_raw(self, set_context):
        self._run("consolidated", "consolidated")

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.parametrize("return_premature", [True, False])
    def test_isend_irecv_consolidated_raw(self, return_premature, set_context):
        self._run(
            "isend_consolidated",
            "irecv_consolidated",
            return_premature=return_premature,
        )


class TestRegisteredGroupSendRecv:
    """group_dst/group_src also work with groups registered via new_group."""

    @classmethod
    def client(cls):
        dist.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:29522",
        )
        group = dist.new_group([0, 1])
        td = TensorDict({("a", "b"): torch.ones(2), "c": torch.ones(2, 3)}, [2])
        td.send(group=group, group_dst=0)

    @classmethod
    def server(cls, queue):
        dist.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:29522",
        )
        group = dist.new_group([0, 1])
        td = TensorDict({("a", "b"): torch.zeros(2), "c": torch.zeros(2, 3)}, [2])
        td.recv(group=group, group_src=1)
        assert (td == 1).all()
        queue.put("yuppie")

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_registered_group(self, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client)

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "yuppie"
        finally:
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)


@pytest.mark.skipif(
    not torch.cuda.device_count() >= 2, reason="not enough cuda devices"
)
class TestRawGroupSendRecvNCCL:
    port = 29523

    @classmethod
    def client(cls):
        torch.cuda.set_device(1)
        pg = _make_raw_pair_group(1, cls.port, backend="nccl")
        td = TensorDict(
            {("a", "b"): torch.ones(2), "c": torch.ones(2, 3)},
            [2],
            device="cuda:1",
        )
        td.send(group=pg, group_dst=0)

    @classmethod
    def server(cls, queue):
        torch.cuda.set_device(0)
        pg = _make_raw_pair_group(0, cls.port, backend="nccl")
        td = TensorDict(
            {("a", "b"): torch.zeros(2), "c": torch.zeros(2, 3)},
            [2],
            device="cuda:0",
        )
        td.recv(group=pg, group_src=1)
        assert (td == 1).all()
        queue.put("yuppie")

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_send_recv_raw_nccl(self, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client)

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=TIMEOUT)
            assert out == "yuppie"
        finally:
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)


class TestGroupPeerValidation:
    """Argument validation for group_dst/group_src (single process, no comms)."""

    @staticmethod
    def make_td():
        return TensorDict({"a": torch.zeros(1)}, [])

    def test_group_peer_requires_group(self):
        td = self.make_td()
        with pytest.raises(ValueError, match="requires `group`"):
            td.send(group_dst=0)
        with pytest.raises(ValueError, match="requires `group`"):
            td.isend(group_dst=0)
        with pytest.raises(ValueError, match="requires `group`"):
            td.recv(group_src=0)
        with pytest.raises(ValueError, match="requires `group`"):
            td.irecv(group_src=0)

    def test_peer_and_group_peer_mutually_exclusive(self):
        store = dist.TCPStore("localhost", 29524, 1, True)
        pg = dist.ProcessGroupGloo(
            dist.PrefixStore("solo", store), 0, 1, timedelta(seconds=10)
        )
        td = self.make_td()
        with pytest.raises(ValueError, match="mutually exclusive"):
            td.send(0, group=pg, group_dst=0)
        with pytest.raises(ValueError, match="mutually exclusive"):
            td.isend(0, group=pg, group_dst=0)
        with pytest.raises(ValueError, match="mutually exclusive"):
            td.recv(0, group=pg, group_src=0)
        with pytest.raises(ValueError, match="mutually exclusive"):
            td.irecv(0, group=pg, group_src=0)

    def test_missing_peer(self):
        td = self.make_td()
        with pytest.raises(ValueError, match="Exactly one"):
            td.send()
        with pytest.raises(ValueError, match="Exactly one"):
            td.isend()
        with pytest.raises(ValueError, match="Exactly one"):
            td.recv()
        with pytest.raises(ValueError, match="Exactly one"):
            td.irecv()


# ========================================================================
# Tensorclass distributed tests
# ========================================================================

from tensordict._testing import MyDistData


class TestTensorclassBroadcast:
    port = "29513"

    @classmethod
    def worker(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group("gloo", rank=rank, world_size=2)

        if rank == 0:
            tc = MyDistData(
                a=torch.ones(3),
                b=torch.ones(3, 2),
                batch_size=[3],
            )
        else:
            tc = MyDistData(
                a=torch.zeros(1),
                b=torch.zeros(1),
                batch_size=[],
            )

        result = tc.broadcast(src=0)
        assert type(result) is MyDistData, f"expected MyDistData, got {type(result)}"
        assert (result.a == 1).all()
        assert (result.b == 1).all()
        assert result.is_consolidated()
        if rank == 1:
            queue.put("yuppie")

    def test_broadcast(self, set_context):
        queue = mp.Queue(1)
        w0 = mp.Process(target=type(self).worker, args=(queue, 0))
        w1 = mp.Process(target=type(self).worker, args=(queue, 1))
        w0.start()
        w1.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            w0.join(timeout=TIMEOUT)
            w1.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestTensorclassAllReduce:
    port = "29514"

    @classmethod
    def worker(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group("gloo", rank=rank, world_size=2)

        tc = MyDistData(
            a=torch.ones(3),
            b=torch.ones(3, 2),
            batch_size=[3],
        )
        tc.all_reduce(op=dist.ReduceOp.SUM)
        assert (tc.a == 2).all()
        assert (tc.b == 2).all()
        if rank == 0:
            queue.put("yuppie")

    def test_all_reduce(self, set_context):
        queue = mp.Queue(1)
        w0 = mp.Process(target=type(self).worker, args=(queue, 0))
        w1 = mp.Process(target=type(self).worker, args=(queue, 1))
        w0.start()
        w1.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            w0.join(timeout=TIMEOUT)
            w1.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestTensorclassAllGather:
    port = "29515"

    @classmethod
    def worker(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group("gloo", rank=rank, world_size=2)

        tc = MyDistData(
            a=torch.full((3,), float(rank)),
            b=torch.full((3, 2), float(rank + 10)),
            batch_size=[3],
        )
        result = tc.all_gather()
        assert len(result) == 2
        assert (result[0]["a"] == 0).all()
        assert (result[0]["b"] == 10).all()
        assert (result[1]["a"] == 1).all()
        assert (result[1]["b"] == 11).all()
        if rank == 0:
            queue.put("yuppie")

    def test_all_gather(self, set_context):
        queue = mp.Queue(1)
        w0 = mp.Process(target=type(self).worker, args=(queue, 0))
        w1 = mp.Process(target=type(self).worker, args=(queue, 1))
        w0.start()
        w1.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            w0.join(timeout=TIMEOUT)
            w1.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestTensorclassScatter:
    port = "29516"

    @classmethod
    def worker(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group("gloo", rank=rank, world_size=2)

        tds = None
        if rank == 0:
            tds = [
                MyDistData(a=torch.zeros(3), b=torch.zeros(3, 2), batch_size=[3]),
                MyDistData(a=torch.ones(3), b=torch.ones(3, 2), batch_size=[3]),
            ]
        tc = MyDistData(a=torch.empty(0), b=torch.empty(0), batch_size=[])
        td = tc.scatter(src=0, tensordicts=tds)
        assert (td["a"] == float(rank)).all()
        assert (td["b"] == float(rank)).all()
        if rank == 1:
            queue.put("yuppie")

    def test_scatter(self, set_context):
        queue = mp.Queue(1)
        w0 = mp.Process(target=type(self).worker, args=(queue, 0))
        w1 = mp.Process(target=type(self).worker, args=(queue, 1))
        w0.start()
        w1.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            w0.join(timeout=TIMEOUT)
            w1.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestTensorclassInitRemote:
    port = "29517"

    @classmethod
    def client(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group("gloo", rank=rank, world_size=2)

        td = TensorDict.from_remote_init(src=0)
        assert type(td) is MyDistData, f"expected MyDistData, got {type(td)}"
        assert set(td.keys()) == {"a", "b"}
        assert (td.a == 1).all()
        assert (td.b == 1).all()
        assert td.is_consolidated()
        queue.put("yuppie")

    @classmethod
    def server(cls, queue):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group("gloo", rank=0, world_size=2)

        tc = MyDistData(a=torch.ones(3), b=torch.ones(3, 2), batch_size=[3])
        tc.init_remote(dst=1)

    def test_init_remote(self, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client, args=(queue, 1))
        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


class TestTensorclassSendConsolidated:
    port = "29518"

    @classmethod
    def client(cls, queue, rank):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group("gloo", rank=rank, world_size=2)

        td_recv = TensorDict.from_remote_init(src=0)
        assert type(td_recv) is MyDistData, f"expected MyDistData, got {type(td_recv)}"
        assert td_recv.is_consolidated()

        td_recv.recv(src=0, consolidated=True)
        assert (td_recv.a == 42).all()
        assert (td_recv.b == 99).all()
        queue.put("yuppie")

    @classmethod
    def server(cls, queue):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = cls.port
        dist.init_process_group("gloo", rank=0, world_size=2)

        tc = MyDistData(a=torch.ones(3), b=torch.ones(3, 2), batch_size=[3])
        tc.init_remote(dst=1)

        tc.a = torch.full((3,), 42.0)
        tc.b = torch.full((3, 2), 99.0)
        tc.send(dst=1, consolidated=True)

    def test_send_consolidated(self, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(target=type(self).client, args=(queue, 1))
        main_worker.start()
        secondary_worker.start()
        out = None
        try:
            out = queue.get(timeout=TIMEOUT)
        finally:
            queue.close()
            main_worker.join(timeout=TIMEOUT)
            secondary_worker.join(timeout=TIMEOUT)
            assert out == "yuppie"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
