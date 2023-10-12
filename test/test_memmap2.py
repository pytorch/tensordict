# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os.path
import pickle
import tempfile

import numpy as np
import pytest
import torch
from _utils_internal import get_available_devices

from tensordict.memmap_refact import MemoryMappedTensor as MemmapTensor
from torch import multiprocessing as mp
from contextlib import nullcontext

TIMEOUT = 100


# def test_memmap_type():
#     array = np.random.rand(1)
#     with pytest.raises(
#         TypeError, match="Convert input to torch.Tensor before calling MemmapTensor"
#     ):
#         MemmapTensor.from_tensor(array)


# def test_grad():
#     t = torch.tensor([1.0])
#     MemmapTensor.from_tensor(t)
#     t = t.requires_grad_()
#     with pytest.raises(
#         RuntimeError, match="MemmapTensor is incompatible with tensor.requires_grad."
#     ):
#         MemmapTensor.from_tensor(t)
#     with pytest.raises(
#         RuntimeError, match="MemmapTensor is incompatible with tensor.requires_grad."
#     ):
#         MemmapTensor.from_tensor(t + 1)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.half,
        torch.float,
        torch.double,
        torch.int,
        torch.uint8,
        torch.long,
        torch.bool,
    ],
)
@pytest.mark.parametrize("shape", [[2], [1, 2]])
def test_memmap_data_type(dtype, shape):
    """Test that MemmapTensor can be created with a given data type and shape."""
    t = torch.tensor([1, 0], dtype=dtype).reshape(shape)
    m = MemmapTensor.from_tensor(t)
    assert m.dtype == t.dtype
    assert (m == t).all()
    assert m.shape == t.shape

    assert m.contiguous().dtype == t.dtype
    assert (m.contiguous() == t).all()
    assert m.contiguous().shape == t.shape

    assert m.clone().dtype == t.dtype
    assert (m.clone() == t).all()
    assert m.clone().shape == t.shape


# @pytest.mark.parametrize("transfer_ownership", [True, False])
# def test_memmap_ownership(transfer_ownership):
#     t = torch.tensor([1])
#     m = MemmapTensor.from_tensor(t, transfer_ownership=transfer_ownership)
#     assert not m.file.delete
#     with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
#         pickle.dump(m, tmp)
#         assert m._has_ownership is not m.transfer_ownership
#         m2 = pickle.load(open(tmp.name, "rb"))
#         assert m2._memmap_array is None  # assert data is not actually loaded
#         assert isinstance(m2, MemmapTensor)
#         assert m2.filename == m.filename
#         # assert m2.file.name == m2.filename
#         # assert m2.file._closer.name == m2.filename
#         assert (
#             m._has_ownership is not m2._has_ownership
#         )  # delete attributes must have changed
#         # assert (
#         #     m.file._closer.delete is not m2.file._closer.delete
#         # )  # delete attributes must have changed
#         del m
#         if transfer_ownership:
#             assert os.path.isfile(m2.filename)
#         else:
#             # m2 should point to a non-existing file
#             assert not os.path.isfile(m2.filename)
#             with pytest.raises(FileNotFoundError):
#                 m2.contiguous()
#
#


@pytest.mark.parametrize("index", [None, 0])
def test_memmap_new(index):
    t = torch.tensor([1])
    m = MemmapTensor.from_tensor(t)
    if index is not None:
        m1 = m[index]
    else:
        m1 = m
    m2 = MemmapTensor.from_tensor(m1)
    assert isinstance(m2, MemmapTensor)
    assert m2._filename == m1._filename
    # assert m2.filename == m2.file.name
    # assert m2.filename == m2.file._closer.name
    if index is not None:
        assert m2.contiguous() == t[index]
    m2c = m2.contiguous()
    assert isinstance(m2c, torch.Tensor)
    assert m2c == m1


@pytest.mark.parametrize("device", get_available_devices())
def test_memmap_same_device_as_tensor(device):
    """
    Created MemmapTensor should be on the same device as the input tensor.
    Check if device is correct when .to(device) is called.
    """
    t = torch.tensor([1], device=device)
    m = MemmapTensor.from_tensor(t)
    assert m.device == torch.device(device)
    for other_device in get_available_devices():
        if other_device != device:
            with pytest.raises(
                RuntimeError,
                match="Expected all tensors to be on the same device, "
                + "but found at least two devices",
            ):
                assert torch.all(m + torch.ones([3, 4], device=other_device) == 1)
        m = m.to(other_device)
        assert m.device == torch.device(other_device)


@pytest.mark.parametrize("device", get_available_devices())
def test_memmap_create_on_same_device(device):
    """Test if the device arg for MemmapTensor init is respected."""
    with pytest.raises(ValueError) if device.type != "cpu" else nullcontext():
        MemmapTensor([3, 4], device=device)
    # assert m.device == torch.device(device)


@pytest.mark.parametrize(
    "value", [torch.zeros([3, 4]), MemmapTensor.from_tensor(torch.zeros([3, 4]))]
)
@pytest.mark.parametrize("shape", [[3, 4]])
def test_memmap_zero_value(value, shape):
    """
    Test if all entries are zeros when MemmapTensor is created with size.
    """
    device = "cpu"
    value = value.to(device)
    expected_memmap_tensor = MemmapTensor.from_tensor(value)
    m = MemmapTensor(torch.zeros(tuple(shape), device=device))
    assert m.shape == (3, 4)
    assert torch.all(m == expected_memmap_tensor)
    assert torch.all(m + torch.ones([3, 4], device=device) == 1)


class TestIndexing:
    @staticmethod
    def _recv_and_send(
        queue_out,
        queue_in,
        filename,
        shape,
    ):
        t = queue_in.get(timeout=TIMEOUT)
        assert isinstance(t, MemmapTensor)
        assert t._filename == filename
        assert t.shape == shape
        assert (t == 0).all()
        msg = "done"
        queue_out.put(msg)

        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "modified"
        assert (t == 1).all()
        queue_out.put("done!!")

        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "deleted"
        # assert not os.path.isfile(filename)
        # with pytest.raises(FileNotFoundError, match="No such file or directory"):
        # print(t + 1)
        # queue_out.put("done again")
        # del queue_in, queue_out

    def test_simple_index(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        # int
        assert isinstance(t[0], MemmapTensor)
        assert t[0]._filename == t._filename
        assert t[0].shape == torch.Size([])
        assert t.shape == torch.Size([10])

    def test_range_index(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        # int
        assert isinstance(t[:2], MemmapTensor)
        assert t[:2]._filename == t._filename
        assert t[:2].shape == torch.Size([2])
        assert t.shape == torch.Size([10])

    def test_double_index(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        y = t[:2]
        # int
        assert isinstance(y, MemmapTensor)
        assert y._filename == t._filename
        assert y._handler is t._handler
        assert y.shape == torch.Size([2])
        assert t.shape == torch.Size([10])
        y = y[:1]
        # int
        assert isinstance(y, MemmapTensor)
        assert y._filename == t._filename
        assert y._handler is t._handler
        assert y.shape == torch.Size([1])
        assert t.shape == torch.Size([10])

    # def test_ownership(self):
    #     t = MemmapTensor.from_tensor(torch.zeros(10))
    #     filename = t.filename
    #     y = t[:2][-1:]
    #     del t
    #     # this would fail if t was gone with its file
    #     assert (y * 0 + 1 == 1).all()
    #     del y
    #     # check that file has gone
    #     assert not os.path.isfile(filename)

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_send_across_procs(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        queue_in = mp.Queue(1)
        queue_out = mp.Queue(1)
        filename = t._filename
        p = mp.Process(
            target=TestIndexing._recv_and_send,
            args=(queue_in, queue_out, filename, torch.Size([10])),
        )
        p.start()
        try:
            queue_out.put(t, block=True)
            msg = queue_in.get(timeout=TIMEOUT)
            assert msg == "done"

            t.fill_(1.0)
            queue_out.put("modified", block=True)
            msg = queue_in.get(timeout=TIMEOUT)
            assert msg == "done!!"

            del t
            queue_out.put("deleted")
        finally:
            p.join()

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_send_across_procs_index(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        queue_in = mp.Queue(1)
        queue_out = mp.Queue(1)
        filename = t._filename
        p = mp.Process(
            target=TestIndexing._recv_and_send,
            args=(queue_in, queue_out, filename, torch.Size([3])),
        )
        p.start()
        try:
            queue_out.put(t[:3], block=True)
            msg = queue_in.get(timeout=TIMEOUT)
            assert msg == "done"

            t.fill_(1.0)
            queue_out.put("modified", block=True)
            msg = queue_in.get(timeout=TIMEOUT)
            assert msg == "done!!"

            del t
            queue_out.put("deleted")
        finally:
            p.join()

    def test_iteration(self):
        t = MemmapTensor.from_tensor(torch.rand(10))
        for i, _t in enumerate(t):
            assert _t == t[i]

    def test_iteration_nd(self):
        t = MemmapTensor.from_tensor(torch.rand(10, 5))
        for i, _t in enumerate(t):
            assert (_t == t[i]).all()

    @staticmethod
    def _test_copy_onto_subproc(queue):
        t = MemmapTensor.from_tensor(torch.rand(10, 5))
        idx = torch.tensor([1, 2])
        t_indexed1 = t[idx]
        queue.put(t_indexed1, block=True)
        while queue.full():
            continue

        idx = torch.tensor([3, 4])
        t_indexed2 = t[idx]
        queue.put(t_indexed2, block=True)
        while queue.full():
            continue
        msg = queue.get(timeout=TIMEOUT)
        assert msg == "done"
        assert (t_indexed1 == t_indexed2).all()
        del queue

    def test_copy_onto(self):
        queue = mp.Queue(1)
        p = mp.Process(target=TestIndexing._test_copy_onto_subproc, args=(queue,))
        p.start()
        try:
            t_indexed1 = queue.get(timeout=TIMEOUT)

            # receive 2nd copy
            t_indexed2 = queue.get(timeout=TIMEOUT)
            t_indexed1.copy_(t_indexed2)
            _ = t_indexed2 + 1
            queue.put("done", block=True)
            queue.close()
        finally:
            p.join()


def test_as_tensor():
    num_samples = 300
    rows, cols = 48, 48
    y = MemmapTensor.from_tensor(torch.zeros((), dtype=torch.uint8).expand(num_samples, rows, cols))
    y.copy_(y + torch.randn(num_samples, rows, cols))
    assert isinstance(y, MemmapTensor)
    idx = slice(3)
    assert isinstance(y[idx], MemmapTensor)
    assert (y[idx] == y.clone()[idx]).all()


def test_filename(tmp_path):
    mt = MemmapTensor.from_tensor(torch.zeros((), dtype=torch.float32).expand(10), filename=tmp_path / "test.memmap")
    assert str(mt._filename) == str(tmp_path / "test.memmap")

    mt2 = MemmapTensor.from_tensor(mt)
    assert str(mt2._filename) == str(tmp_path / "test.memmap")
    assert mt2 is mt

    mt3 = MemmapTensor.from_tensor(mt, filename=tmp_path / "test.memmap")
    assert str(mt3._filename) == str(tmp_path / "test.memmap")
    assert mt3 is mt

    mt4 = MemmapTensor.from_tensor(mt, filename=tmp_path / "test2.memmap")
    assert str(mt4._filename) == str(tmp_path / "test2.memmap")
    assert mt4 is not mt

    del mt
    del mt4
    # files should persist
    assert (tmp_path / "test.memmap").exists()
    assert (tmp_path / "test2.memmap").exists()

def test_handler():
    mt = MemmapTensor.from_tensor(torch.zeros((), dtype=torch.float32).expand(10))

    mt2 = MemmapTensor.from_tensor(mt)
    assert mt2._handler is mt._handler



def test_memmap_from_memmap():
    mt = MemmapTensor.from_tensor(torch.zeros(()).expand(4, 3, 2, 1))
    mt2 = MemmapTensor.from_tensor(mt)
    assert mt2.squeeze(-1).shape == torch.Size([4, 3, 2])


def test_memmap_cast():
    # ensure memmap can be cast to tensor and viceversa
    x = torch.zeros(3, 4, 5)
    y = MemmapTensor.from_tensor(torch.ones(3, 4, 5))

    x[:2] = y[:2]
    assert (x[:2] == 1).all()
    y[2:] = x[2:]
    assert (y[2:] == 0).all()


@pytest.fixture
def dummy_memmap():
    return MemmapTensor.from_tensor(torch.randn(10, 11))


class TestOps:
    def test_eq(self, dummy_memmap):
        memmap = dummy_memmap
        assert (memmap == memmap.clone()).all()
        assert (memmap.clone() == memmap).all()

    def test_fill_(self, dummy_memmap):
        memmap = dummy_memmap.fill_(1.0)
        assert (memmap == 1).all()
        assert isinstance(memmap, MemmapTensor)

    def test_copy_(self, dummy_memmap):
        memmap = dummy_memmap.copy_(torch.ones(10, 11))
        assert (memmap == 1).all()
        assert isinstance(memmap, MemmapTensor)
        # check that memmap can be put in a tensor
        assert (torch.ones(10, 11).copy_(memmap) == 1).all()

    def test_or(self):
        memmap = MemmapTensor.from_tensor(torch.ones(10, 11, dtype=torch.bool))
        assert (memmap | (~memmap)).all()

    def test_ne(self):
        memmap = MemmapTensor.from_tensor(torch.ones(10, 11, dtype=torch.bool))
        assert (memmap != ~memmap).all()


# def test_memmap_del(tmpdir):
#     t = torch.tensor([1])
#     m = MemmapTensor.from_tensor(t, filename=tmpdir / "tensor")
#     # filename = m.filename
#     assert os.path.isfile(tmpdir / "tensor")
#     del m
#     assert not os.path.isfile(tmpdir / "tensor")

# @pytest.mark.parametrize("value", [True, False])
# def test_memmap_ownership_2pass(value):
#     t = torch.tensor([1])
#     m1 = MemmapTensor.from_tensor(t, transfer_ownership=value)
#     filename = m1._filename
#     with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp2:
#         pickle.dump(m1, tmp2)
#         # after we dump m1, m1 has lost ownership and waits for m2 to pick it up
#         # if m1 is deleted and m2 is never created, the file is not cleared.
#         if value:
#             assert not m1._has_ownership
#         else:
#             assert m1._has_ownership
#
#         m2 = pickle.load(open(tmp2.name, "rb"))
#         assert m2._filename == m1._filename
#         with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp3:
#             pickle.dump(m2, tmp3)
#             m3 = pickle.load(open(tmp3.name, "rb"))
#             assert m3._filename == m1._filename
#
#     del m1, m2, m3
#     assert not os.path.isfile(filename)


# class TestMP:
#     @staticmethod
#     def getdata(data, queue):
#         queue.put(("has_ownership", data._has_ownership))
#         # queue.put(("transfer_ownership", data.transfer_ownership))
#
#     # @pytest.mark.parametrize("transfer_ownership", [True, False])
#     def test(self, transfer_ownership, tmp_path):
#         m = MemmapTensor(
#             3, filename=tmp_path / "tensor.mp"
#         )
#         queue = mp.Queue(1)
#         p = mp.Process(target=TestMP.getdata, args=(m, queue))
#         p.start()
#         try:
#             msg, val = queue.get()
#             assert msg == "has_ownership"
#             # assert val is transfer_ownership
#             if transfer_ownership:
#                 assert not m._has_ownership
#             else:
#                 assert m._has_ownership
#             msg, val = queue.get()
#             assert msg == "transfer_ownership"
#             # assert val is transfer_ownership
#         finally:
#             p.join()
#             queue.close()
# @pytest.mark.parametrize(
#     "mode", ["r", "r+", "w+", "c", "readonly", "readwrite", "write", "copyonwrite"]
# )
# def test_mode(mode, tmp_path):
#     mt = MemmapTensor(10, dtype=torch.float32, filename=tmp_path / "test.memmap")
#     mt[:] = torch.ones(10) * 1.5
#     del mt
#
#     if mode in ("r", "readonly"):
#         with pytest.raises(ValueError, match=r"Accepted values for mode are"):
#             MemmapTensor(
#                 10, dtype=torch.float32, filename=tmp_path / "test.memmap", mode=mode
#             )
#         return
#     mt = MemmapTensor(
#         10, dtype=torch.float32, filename=tmp_path / "test.memmap", mode=mode
#     )
#     if mode in ("r+", "readwrite", "c", "copyonwrite"):
#         # data in memmap persists
#         assert (mt.as_tensor() == 1.5).all()
#     elif mode in ("w+", "write"):
#         # memmap is initialized to zero
#         assert (mt.as_tensor() == 0).all()
#
#     mt[:] = torch.ones(10) * 2.5
#     assert (mt.as_tensor() == 2.5).all()
#     del mt
#
#     mt2 = MemmapTensor(10, dtype=torch.float32, filename=tmp_path / "test.memmap")
#     if mode in ("c", "copyonwrite"):
#         # tensor was only mutated in memory, not on disk
#         assert (mt2.as_tensor() == 1.5).all()
#     else:
#         assert (mt2.as_tensor() == 2.5).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
