import argparse

import pytest
import torch
from _pytest.fixtures import fixture

from tensordict import MemmapTensor, TensorDict
from torch import multiprocessing as mp


@fixture
def set_context():
    try:
        mp.set_start_method("spawn")
    except Exception:
        print("context already set")


class TestGather:
    @staticmethod
    def client(memmap_filename):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10005",
        )

        td = TensorDict(
            {
                ("a", "b"): torch.randn(2),
                "c": torch.randn(2),
                ("d", "e", "f"): MemmapTensor.from_tensor(
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
            init_method="tcp://localhost:10005",
        )

        td = (
            TensorDict(
                {
                    ("a", "b"): torch.zeros(2),
                    "c": torch.zeros(2),
                    ("d", "e", "f"): MemmapTensor.from_tensor(torch.zeros(2, 2)),
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
        master = mp.Process(target=TestGather.server, args=(queue,))
        slave = mp.Process(target=TestGather.client, args=(str(tmp_path / "sub"),))

        master.start()
        slave.start()
        out = queue.get(timeout=10)
        assert out == "yuppie"
        master.join()
        slave.join()


class TestSend:
    @staticmethod
    def client():
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10005",
        )

        td = TensorDict(
            {
                ("a", "b"): torch.randn(2),
                "c": torch.randn(2, 3),
                "_": torch.ones(2, 1, 5),
                ("d", "e", "f"): MemmapTensor.from_tensor(torch.ones(2, 2)),
            },
            [2],
        )
        td.send(0)

    @staticmethod
    def server(queue):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10005",
        )
        td = TensorDict(
            {
                ("a", "b"): torch.zeros(2),
                "c": torch.zeros(2, 3),
                "_": torch.zeros(2, 1, 5),
                ("d", "e", "f"): MemmapTensor.from_tensor(torch.zeros(2, 2)),
            },
            [2],
        )
        td.recv(1)
        assert (td != 0).all()
        queue.put("yuppie")

    def test_send(self, set_context):
        queue = mp.Queue(1)
        master = mp.Process(target=TestSend.server, args=(queue,))
        slave = mp.Process(target=TestSend.client)

        master.start()
        slave.start()
        out = queue.get(timeout=10)
        assert out == "yuppie"
        master.join()
        slave.join()


class TestiSend:
    @staticmethod
    def client():
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10005",
        )

        td = TensorDict(
            {
                ("a", "b"): torch.randn(2),
                "c": torch.randn(2, 3),
                "_": torch.ones(2, 1, 5),
                ("d", "e", "f"): MemmapTensor.from_tensor(torch.randn(2, 2)),
            },
            [2],
        )
        td.isend(0)

    @staticmethod
    def server(queue, return_premature):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10005",
        )
        td = TensorDict(
            {
                ("a", "b"): torch.zeros(2),
                "c": torch.zeros(2, 3),
                "_": torch.zeros(2, 1, 5),
                ("d", "e", "f"): MemmapTensor.from_tensor(torch.zeros(2, 2)),
            },
            [2],
        )
        out = td.irecv(1, return_premature=return_premature)
        if return_premature:
            for fut in out:
                fut.wait()
        assert (td != 0).all()
        queue.put("yuppie")

    @pytest.mark.parametrize("return_premature", [True, False])
    def test_isend(self, return_premature, set_context):
        queue = mp.Queue(1)
        master = mp.Process(target=TestiSend.server, args=(queue, return_premature))
        slave = mp.Process(target=TestiSend.client)

        master.start()
        slave.start()
        out = queue.get(timeout=10)
        assert out == "yuppie"
        master.join()
        slave.join()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
