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
            init_method="tcp://localhost:10017",
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
            init_method="tcp://localhost:10017",
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
        main_worker = mp.Process(target=TestGather.server, args=(queue,))
        secondary_worker = mp.Process(
            target=TestGather.client, args=(str(tmp_path / "sub"),)
        )

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=30)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


@pytest.mark.parametrize("pseudo_rand", [True, False])
class TestSend:
    @staticmethod
    def client(pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )

        td = TensorDict(
            {
                ("a", "b"): torch.randn(2),
                "c": torch.randn(2, 3),
                ("d", "e", "f"): MemmapTensor.from_tensor(torch.ones(2, 2)),
            },
            [2],
        )
        td["_"] = torch.ones(2, 1, 5)
        td.send(0, pseudo_rand=pseudo_rand)

    @staticmethod
    def server(queue, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = TensorDict(
            {
                ("a", "b"): torch.zeros(2),
                "_": torch.zeros(2, 1, 5),
                ("d", "e", "f"): MemmapTensor.from_tensor(torch.zeros(2, 2)),
            },
            [2],
        )
        td["c"] = torch.zeros(2, 3)
        td.recv(1, pseudo_rand=pseudo_rand)
        assert (td != 0).all()
        queue.put("yuppie")

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_send(self, pseudo_rand, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=TestSend.server, args=(queue, pseudo_rand))
        secondary_worker = mp.Process(target=TestSend.client, args=(pseudo_rand,))

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=30)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


@pytest.mark.parametrize("pseudo_rand", [True, False])
class TestiRecv:
    @staticmethod
    def client(pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
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
        td.send(0, pseudo_rand=pseudo_rand)

    @staticmethod
    def server(queue, return_premature, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
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
        out = td.irecv(1, return_premature=return_premature, pseudo_rand=pseudo_rand)
        if return_premature:
            for fut in out:
                fut.wait()
        assert (td != 0).all()
        queue.put("yuppie")

    @pytest.mark.parametrize("return_premature", [True, False])
    def test_isend(self, pseudo_rand, return_premature, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(
            target=TestiRecv.server, args=(queue, return_premature, pseudo_rand)
        )
        secondary_worker = mp.Process(target=TestiRecv.client, args=(pseudo_rand,))

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=30)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


@pytest.mark.parametrize("pseudo_rand", [True, False])
class TestiSend:
    @staticmethod
    def client(pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )

        td = TensorDict(
            {
                ("a", "b"): torch.randn(2),
                "c": torch.randn(2, 3),
                ("d", "e", "f"): MemmapTensor.from_tensor(torch.ones(2, 2)),
            },
            [2],
        )
        td["_"] = torch.ones(2, 1, 5)
        td.isend(0, pseudo_rand=pseudo_rand)

    @staticmethod
    def server(queue, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = TensorDict(
            {
                ("a", "b"): torch.zeros(2),
                "_": torch.zeros(2, 1, 5),
                ("d", "e", "f"): MemmapTensor.from_tensor(torch.zeros(2, 2)),
            },
            [2],
        )
        td["c"] = torch.zeros(2, 3)
        td.recv(1, pseudo_rand=pseudo_rand)
        assert (td != 0).all()
        queue.put("yuppie")

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_send(self, pseudo_rand, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=TestiSend.server, args=(queue, pseudo_rand))
        secondary_worker = mp.Process(target=TestiSend.client, args=(pseudo_rand,))

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=30)
            assert out == "yuppie"
        except Exception as err:
            # otherwise pytest does not capture it
            raise err
        finally:
            main_worker.join()
            secondary_worker.join()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
