# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import abc
import argparse
import logging
import os
import sys

import pytest
import torch
from _pytest.fixtures import fixture
from packaging import version

from packaging.version import parse

from tensordict import MemoryMappedTensor, TensorDict
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
        logging.info("context already set")


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
        with torch.device(f"cuda:{device}") if device is not None else torch.device(
            "cuda"
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
        #     logging.info(module.state_dict())

        # td = TensorDict(module.state_dict(), []).unflatten_keys(".")
        td = TensorDict.from_module(module, use_state_dict=True)
        if rank == 0:
            td.memmap(path)
        dist.destroy_process_group()

    def test_fsdp_module(self, tmpdir):
        try:
            mp.set_start_method("spawn")
        except Exception:
            logging.info("start method already set to", mp.get_start_method())
        proc0 = mp.Process(target=self.worker, args=(0, tmpdir))
        proc1 = mp.Process(target=self.worker, args=(1, tmpdir))
        proc0.start()
        proc1.start()
        proc0.join(timeout=TIMEOUT)
        proc1.join(timeout=TIMEOUT)
        assert (TensorDict.load_memmap(tmpdir) == 1).all()


# not using TorchVersion to make the comparison work with dev
TORCH_VERSION = version.parse(
    ".".join(map(str, version.parse(torch.__version__).release))
)


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.2.0"),
    reason=f"DTensor requires a more recent PyTorch (torch > 2.2.0, got {torch.__version__}).",
)
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
            #     logging.info(key, val)
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
            logging.info("start method already set to", mp.get_start_method())
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
    parse(torch.__version__) < parse("2.0"), reason="Avoid pickle error"
)
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
                logging.info("waiting...")
                _out.wait()
                logging.info("done")
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
        td = torch.stack([td1, td2], 0)
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
        td = TensorDict({"ls": torch.stack([td1, td2], 0)}, [2, 2])
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
        td = torch.stack([td1, td2], 0)
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
        td = torch.stack([td1, td2], 0)
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
        td = torch.stack([td1, td2], 0)
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
        td = torch.stack([td1, td2], 0)
        td = TensorDict({"td": td}, [2, 2])
        return td


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
