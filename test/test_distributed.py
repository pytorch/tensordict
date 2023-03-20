import abc
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
        main_worker = mp.Process(target=type(self).server, args=(queue,))
        secondary_worker = mp.Process(
            target=type(self).client, args=(str(tmp_path / "sub"),)
        )

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=30)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


# =========================================
# Test td.send
# ------------


class SendBase:
    @staticmethod
    @abc.abstractmethod
    def make_td(ones):
        raise NotImplementedError

    @classmethod
    def client(cls, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls.make_td(ones=True)
        td.send(0, pseudo_rand=pseudo_rand)

    @classmethod
    def server(cls, queue, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls.make_td(ones=False)
        td.recv(1, pseudo_rand=pseudo_rand)
        assert (td == 1).all()
        queue.put("yuppie")

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_send(self, pseudo_rand, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue, pseudo_rand))
        secondary_worker = mp.Process(target=type(self).client, args=(pseudo_rand,))

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=30)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td["_"] = fun(2, 1, 5)
        return td


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
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


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
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
    def client(cls, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls.make_td(ones=True)
        td.send(0, pseudo_rand=pseudo_rand)

    @classmethod
    def server(cls, queue, return_premature, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls.make_td(ones=False)
        out = td.irecv(1, return_premature=return_premature, pseudo_rand=pseudo_rand)
        if return_premature:
            for fut in out:
                fut.wait()
        assert (td == 1).all()
        queue.put("yuppie")

    @pytest.mark.parametrize("return_premature", [True, False])
    def test_irecv(self, pseudo_rand, return_premature, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(
            target=type(self).server, args=(queue, return_premature, pseudo_rand)
        )
        secondary_worker = mp.Process(target=type(self).client, args=(pseudo_rand,))

        main_worker.start()
        secondary_worker.start()
        try:
            out = queue.get(timeout=30)
            assert out == "yuppie"
        finally:
            main_worker.join()
            secondary_worker.join()


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        return td


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td2 = td1.clone()
        td2["c"] = td2["c"].unsqueeze(-1).expand(2, 3, 10).contiguous()
        td2["g"] = td1["c"].clone()
        td = torch.stack([td1, td2], 0)
        return td


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
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
    def client(cls, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=1,
            world_size=2,
            init_method="tcp://localhost:10017",
        )

        td = cls.make_td(True)
        td.isend(0, pseudo_rand=pseudo_rand)

    @classmethod
    def server(cls, queue, pseudo_rand):
        torch.distributed.init_process_group(
            "gloo",
            rank=0,
            world_size=2,
            init_method="tcp://localhost:10017",
        )
        td = cls.make_td(False)
        td.recv(1, pseudo_rand=pseudo_rand)
        assert (td == 1).all()
        queue.put("yuppie")

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_isend(self, pseudo_rand, set_context):
        queue = mp.Queue(1)
        main_worker = mp.Process(target=type(self).server, args=(queue, pseudo_rand))
        secondary_worker = mp.Process(target=type(self).client, args=(pseudo_rand,))

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


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        return td


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
            },
            [2],
        )
        td2 = td1.clone()
        td2["c"] = td2["c"].unsqueeze(-1).expand(2, 3, 10).contiguous()
        td2["g"] = td1["c"].clone()
        td = torch.stack([td1, td2], 0)
        return td


@pytest.mark.parametrize("pseudo_rand", [True, False])
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
                ("d", "e", "f"): MemmapTensor.from_tensor(fun(2, 2)),
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
