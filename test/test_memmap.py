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
from tensordict import MemmapTensor
from torch import multiprocessing as mp


def test_memmap_type():
    array = np.random.rand(1)
    with pytest.raises(
        TypeError, match="Convert input to torch.Tensor before calling MemmapTensor"
    ):
        MemmapTensor.from_tensor(array)


def test_grad():
    t = torch.tensor([1.0])
    MemmapTensor.from_tensor(t)
    t = t.requires_grad_()
    with pytest.raises(
        RuntimeError, match="MemmapTensor is incompatible with tensor.requires_grad."
    ):
        MemmapTensor.from_tensor(t)
    with pytest.raises(
        RuntimeError, match="MemmapTensor is incompatible with tensor.requires_grad."
    ):
        MemmapTensor.from_tensor(t + 1)


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


def test_memmap_del():
    t = torch.tensor([1])
    m = MemmapTensor.from_tensor(t)
    filename = m.filename
    assert os.path.isfile(filename)
    del m
    with pytest.raises(AssertionError):
        assert os.path.isfile(filename)


@pytest.mark.parametrize("transfer_ownership", [True, False])
def test_memmap_ownership(transfer_ownership):
    t = torch.tensor([1])
    m = MemmapTensor.from_tensor(t, transfer_ownership=transfer_ownership)
    assert not m.file.delete
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp:
        pickle.dump(m, tmp)
        assert m._has_ownership is not m.transfer_ownership
        m2 = pickle.load(open(tmp.name, "rb"))
        assert m2._memmap_array is None  # assert data is not actually loaded
        assert isinstance(m2, MemmapTensor)
        assert m2.filename == m.filename
        # assert m2.file.name == m2.filename
        # assert m2.file._closer.name == m2.filename
        assert (
            m._has_ownership is not m2._has_ownership
        )  # delete attributes must have changed
        # assert (
        #     m.file._closer.delete is not m2.file._closer.delete
        # )  # delete attributes must have changed
        del m
        if transfer_ownership:
            assert os.path.isfile(m2.filename)
        else:
            # m2 should point to a non-existing file
            assert not os.path.isfile(m2.filename)
            with pytest.raises(FileNotFoundError):
                m2.contiguous()


@pytest.mark.parametrize("value", [True, False])
def test_memmap_ownership_2pass(value):
    t = torch.tensor([1])
    m1 = MemmapTensor.from_tensor(t, transfer_ownership=value)
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp2:
        pickle.dump(m1, tmp2)
        m2 = pickle.load(open(tmp2.name, "rb"))
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp3:
            pickle.dump(m2, tmp3)
            m3 = pickle.load(open(tmp3.name, "rb"))
            assert m1._has_ownership + m2._has_ownership + m3._has_ownership == 1

    del m1, m2, m3
    m1 = MemmapTensor.from_tensor(t, transfer_ownership=value)
    with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp2:
        pickle.dump(m1, tmp2)
        m2 = pickle.load(open(tmp2.name, "rb"))
        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp3:
            pickle.dump(m1, tmp3)
            m3 = pickle.load(open(tmp3.name, "rb"))
            assert m1._has_ownership + m2._has_ownership + m3._has_ownership == 1


@pytest.mark.parametrize(
    "index",
    [
        None,
        [
            0,
        ],
    ],
)
def test_memmap_new(index):
    t = torch.tensor([1])
    m = MemmapTensor.from_tensor(t)
    if index is not None:
        m1 = m[index]
    else:
        m1 = m
    m2 = MemmapTensor.from_tensor(m1)
    assert isinstance(m2, MemmapTensor)
    assert m2.filename == m1.filename
    assert m2.filename == m2.file.name
    assert m2.filename == m2.file._closer.name
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
    m = MemmapTensor([3, 4], device=device)
    assert m.device == torch.device(device)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "value", [torch.zeros([3, 4]), MemmapTensor.from_tensor(torch.zeros([3, 4]))]
)
@pytest.mark.parametrize("shape", [[3, 4], [[3, 4]]])
def test_memmap_zero_value(device, value, shape):
    """
    Test if all entries are zeros when MemmapTensor is created with size.
    """
    value = value.to(device)
    expected_memmap_tensor = MemmapTensor.from_tensor(value)
    m = MemmapTensor(*shape, device=device)
    assert m.shape == (3, 4)
    assert torch.all(m == expected_memmap_tensor)
    assert torch.all(m + torch.ones([3, 4], device=device) == 1)


class TestIndexing:
    @staticmethod
    def _recv_and_send(queue, filename, shape):
        t = queue.get(timeout=10.0)
        assert isinstance(t, MemmapTensor)
        assert t.filename == filename
        assert t.shape == shape
        assert (t == 0).all()
        msg = "done"
        queue.put(msg)
        while queue.full():
            continue

        msg = queue.get(timeout=10.0)
        assert msg == "modified"
        assert (t == 1).all()
        queue.put("done!!")

        msg = queue.get(timeout=10.0)
        assert msg == "deleted"
        assert not os.path.isfile(filename)
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            print(t + 1)
        queue.put("done again")
        del queue

    def test_simple_index(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        # int
        assert isinstance(t[0], MemmapTensor)
        assert t[0].filename == t.filename
        assert t[0].shape == torch.Size([])
        assert t.shape == torch.Size([10])

    def test_range_index(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        # int
        assert isinstance(t[:2], MemmapTensor)
        assert t[:2].filename == t.filename
        assert t[:2].shape == torch.Size([2])
        assert t.shape == torch.Size([10])

    def test_double_index(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        y = t[:2][-1:]
        # int
        assert isinstance(y, MemmapTensor)
        assert y.filename == t.filename
        assert y.shape == torch.Size([1])
        assert t.shape == torch.Size([10])

    def test_ownership(self):
        t = MemmapTensor.from_tensor(torch.zeros(10))
        y = t[:2][-1:]
        del t
        with pytest.raises(FileNotFoundError, match="No such file or directory"):
            y + 0

    def test_send_across_procs(self):
        t = MemmapTensor.from_tensor(torch.zeros(10), transfer_ownership=False)
        queue = mp.Queue(1)
        filename = t.filename
        p = mp.Process(
            target=TestIndexing._recv_and_send, args=(queue, filename, torch.Size([10]))
        )
        try:
            p.start()
            queue.put(t, block=True)
            while queue.full():
                continue
            msg = queue.get(timeout=10.0)
            assert msg == "done"

            t.fill_(1.0)
            queue.put("modified", block=True)
            while queue.full():
                continue
            msg = queue.get(timeout=10.0)
            assert msg == "done!!"

            del t
            queue.put("deleted")
            while queue.full():
                continue
            msg = queue.get(timeout=10.0)
            assert msg == "done again"
            p.join()
        except Exception as e:
            p.join()
            raise e

    def test_send_across_procs_index(self):
        t = MemmapTensor.from_tensor(torch.zeros(10), transfer_ownership=False)
        queue = mp.Queue(1)
        filename = t.filename
        p = mp.Process(
            target=TestIndexing._recv_and_send, args=(queue, filename, torch.Size([3]))
        )
        try:
            p.start()
            queue.put(t[:3], block=True)
            while queue.full():
                continue
            msg = queue.get(timeout=10.0)
            assert msg == "done"

            t.fill_(1.0)
            queue.put("modified", block=True)
            while queue.full():
                continue
            msg = queue.get(timeout=10.0)
            assert msg == "done!!"

            del t
            queue.put("deleted")
            while queue.full():
                continue
            msg = queue.get(timeout=10.0)
            assert msg == "done again"
            p.join()
        except Exception as e:
            p.join()
            raise e

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
        queue.put(t[idx], block=True)
        while queue.full():
            continue

        idx = torch.tensor([3, 4])
        queue.put(t[idx], block=True)
        while queue.full():
            continue
        msg = queue.get(timeout=10.0)
        assert msg == "done"
        del queue

    def test_copy_onto(self):
        queue = mp.Queue(1)
        p = mp.Process(target=TestIndexing._test_copy_onto_subproc, args=(queue,))
        p.start()
        try:
            t_indexed1 = queue.get(timeout=10)
            assert (t_indexed1._index[0] == torch.tensor([1, 2])).all()
            # check that file is not opened if we did not access it
            assert t_indexed1._memmap_array is None
            _ = t_indexed1 + 1
            # check that file is now opened
            assert t_indexed1._memmap_array is not None

            # receive 2nd copy
            t_indexed2 = queue.get(timeout=10)
            assert t_indexed2.filename == t_indexed1.filename
            assert (t_indexed2._index[0] == torch.tensor([3, 4])).all()
            # check that file is open only once
            assert t_indexed1._memmap_array is not None
            assert t_indexed2._memmap_array is None
            t_indexed1.copy_(t_indexed2)
            # same assertion: after copying we should only have one file opened
            assert t_indexed1._memmap_array is not None
            assert t_indexed2._memmap_array is None
            _ = t_indexed2 + 1
            # now we should find 2 opened files
            assert t_indexed1._memmap_array is not None
            assert t_indexed2._memmap_array is not None
            queue.put("done", block=True)
            queue.close()
            p.join()
        except Exception as e:
            p.join()
            raise e


def test_as_tensor():
    num_samples = 300
    rows, cols = 48, 48
    idx = torch.randint(num_samples, (128,))
    y = MemmapTensor(num_samples, rows, cols, dtype=torch.uint8)
    y.copy_(y + torch.randn(num_samples, rows, cols))
    assert isinstance(y, MemmapTensor)
    assert isinstance(y[idx], MemmapTensor)
    assert (y[idx] == y.as_tensor()[idx]).all()


def test_filename(tmpdir):
    mt = MemmapTensor(10, dtype=torch.float32, filename=tmpdir / "test.memmap")
    assert mt.filename == str(tmpdir / "test.memmap")

    mt2 = MemmapTensor.from_tensor(mt)
    assert mt2.filename == str(tmpdir / "test.memmap")
    assert mt2 is mt

    mt3 = MemmapTensor.from_tensor(mt, filename=tmpdir / "test.memmap")
    assert mt3.filename == str(tmpdir / "test.memmap")
    assert mt3 is mt

    mt4 = MemmapTensor.from_tensor(mt, filename=tmpdir / "test2.memmap")
    assert mt4.filename == str(tmpdir / "test2.memmap")
    assert mt4 is not mt

    del mt
    # files should persist
    assert (tmpdir / "test.memmap").exists()
    assert (tmpdir / "test2.memmap").exists()


@pytest.mark.parametrize(
    "mode", ["r", "r+", "w+", "c", "readonly", "readwrite", "write", "copyonwrite"]
)
def test_mode(mode, tmpdir):
    mt = MemmapTensor(10, dtype=torch.float32, filename=tmpdir / "test.memmap")
    mt[:] = torch.ones(10) * 1.5
    del mt

    if mode in ("r", "readonly"):
        with pytest.raises(ValueError, match=r"Accepted values for mode are"):
            MemmapTensor(
                10, dtype=torch.float32, filename=tmpdir / "test.memmap", mode=mode
            )
        return
    mt = MemmapTensor(
        10, dtype=torch.float32, filename=tmpdir / "test.memmap", mode=mode
    )
    if mode in ("r+", "readwrite", "c", "copyonwrite"):
        # data in memmap persists
        assert (mt.as_tensor() == 1.5).all()
    elif mode in ("w+", "write"):
        # memmap is initialized to zero
        assert (mt.as_tensor() == 0).all()

    mt[:] = torch.ones(10) * 2.5
    assert (mt.as_tensor() == 2.5).all()
    del mt

    mt2 = MemmapTensor(10, dtype=torch.float32, filename=tmpdir / "test.memmap")
    if mode in ("c", "copyonwrite"):
        # tensor was only mutated in memory, not on disk
        assert (mt2.as_tensor() == 1.5).all()
    else:
        assert (mt2.as_tensor() == 2.5).all()


def test_memmap_from_memmap():
    mt2 = MemmapTensor.from_tensor(MemmapTensor(4, 3, 2, 1))
    assert mt2.squeeze(-1).shape == torch.Size([4, 3, 2])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
