# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from contextlib import nullcontext

import pytest
import torch
from _utils_internal import get_available_devices

from tensordict.memmap import MemoryMappedTensor
from torch import multiprocessing as mp

TIMEOUT = 100


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
    """Test that MemoryMappedTensor can be created with a given data type and shape."""
    t = torch.tensor([1, 0], dtype=dtype).reshape(shape)
    m = MemoryMappedTensor.from_tensor(t)
    assert m.dtype == t.dtype
    assert (m == t).all()
    assert m.shape == t.shape

    assert m.contiguous().dtype == t.dtype
    assert (m.contiguous() == t).all()
    assert m.contiguous().shape == t.shape

    assert m.clone().dtype == t.dtype
    assert (m.clone() == t).all()
    assert m.clone().shape == t.shape


@pytest.mark.parametrize("index", [None, 0])
def test_memmap_new(index):
    t = torch.tensor([1])
    m = MemoryMappedTensor.from_tensor(t)
    if index is not None:
        m1 = m[index]
    else:
        m1 = m
    m2 = MemoryMappedTensor.from_tensor(m1)
    assert isinstance(m2, MemoryMappedTensor)
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
    Created MemoryMappedTensor should be on the same device as the input tensor.
    Check if device is correct when .to(device) is called.
    """
    t = torch.tensor([1], device=device)
    m = MemoryMappedTensor.from_tensor(t)
    assert m.device == torch.device("cpu")


@pytest.mark.parametrize("device", get_available_devices())
def test_memmap_create_on_same_device(device):
    """Test if the device arg for MemoryMappedTensor init is respected."""
    with pytest.raises(ValueError) if device.type != "cpu" else nullcontext():
        MemoryMappedTensor([3, 4], device=device)
    # assert m.device == torch.device(device)


@pytest.mark.parametrize(
    "value", [torch.zeros([3, 4]), MemoryMappedTensor.from_tensor(torch.zeros([3, 4]))]
)
@pytest.mark.parametrize("shape", [[3, 4]])
def test_memmap_zero_value(value, shape):
    """
    Test if all entries are zeros when MemoryMappedTensor is created with size.
    """
    device = "cpu"
    value = value.to(device)
    expected_memmap_tensor = MemoryMappedTensor.from_tensor(value)
    m = MemoryMappedTensor(torch.zeros(tuple(shape), device=device))
    assert m.shape == (3, 4)
    assert torch.all(m == expected_memmap_tensor)
    assert torch.all(m + torch.ones([3, 4], device=device) == 1)


def test_existing(tmp_path):
    tensor = torch.zeros(())
    t1 = MemoryMappedTensor.from_tensor(  # noqa: F841
        tensor, filename=tmp_path / "file.memmap"
    )
    with pytest.raises(RuntimeError, match="already exists"):
        MemoryMappedTensor.from_tensor(tensor, filename=tmp_path / "file.memmap")
    MemoryMappedTensor.from_tensor(
        tensor, filename=tmp_path / "file.memmap", existsok=True
    )


@pytest.mark.parametrize("shape", [(), (3, 4)])
@pytest.mark.parametrize("dtype", [None, torch.float, torch.double, torch.int])
@pytest.mark.parametrize("device", [None] + get_available_devices())
@pytest.mark.parametrize("from_path", [True, False])
class TestConstructors:
    @pytest.mark.parametrize("shape_arg", ["expand", "arg", "kwarg"])
    def test_zeros(self, shape, dtype, device, tmp_path, from_path, shape_arg):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        if device is not None and device.type != "cpu":
            with pytest.raises(RuntimeError):
                MemoryMappedTensor.zeros(
                    shape, dtype=dtype, device=device, filename=filename
                )
            return
        if shape_arg == "expand":
            with pytest.raises(TypeError) if shape == () else nullcontext():
                t = MemoryMappedTensor.zeros(
                    *shape, dtype=dtype, device=device, filename=filename
                )
            if shape == ():
                return
        elif shape_arg == "arg":
            t = MemoryMappedTensor.zeros(
                shape, dtype=dtype, device=device, filename=filename
            )
        elif shape_arg == "kwarg":
            t = MemoryMappedTensor.zeros(
                shape=shape, dtype=dtype, device=device, filename=filename
            )

        assert t.shape == shape
        if dtype is not None:
            assert t.dtype is dtype
        if filename is not None:
            assert t.filename == filename
        assert (t == 0).all()

    @pytest.mark.parametrize("shape_arg", ["expand", "arg", "kwarg"])
    def test_ones(self, shape, dtype, device, tmp_path, from_path, shape_arg):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        if device is not None and device.type != "cpu":
            with pytest.raises(RuntimeError):
                MemoryMappedTensor.ones(
                    shape, dtype=dtype, device=device, filename=filename
                )
            return
        if shape_arg == "expand":
            with pytest.raises(TypeError) if shape == () else nullcontext():
                t = MemoryMappedTensor.ones(
                    *shape, dtype=dtype, device=device, filename=filename
                )
            if shape == ():
                return
        elif shape_arg == "arg":
            t = MemoryMappedTensor.ones(
                shape, dtype=dtype, device=device, filename=filename
            )
        elif shape_arg == "kwarg":
            t = MemoryMappedTensor.ones(
                shape=shape, dtype=dtype, device=device, filename=filename
            )
        assert t.shape == shape
        if dtype is not None:
            assert t.dtype is dtype
        if filename is not None:
            assert t.filename == filename
        assert (t == 1).all()

    @pytest.mark.parametrize("shape_arg", ["expand", "arg", "kwarg"])
    def test_empty(self, shape, dtype, device, tmp_path, from_path, shape_arg):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        if device is not None and device.type != "cpu":
            with pytest.raises(RuntimeError):
                MemoryMappedTensor.empty(
                    shape, dtype=dtype, device=device, filename=filename
                )
            return
        if shape_arg == "expand":
            with pytest.raises(TypeError) if shape == () else nullcontext():
                t = MemoryMappedTensor.empty(
                    *shape, dtype=dtype, device=device, filename=filename
                )
            if shape == ():
                return
        elif shape_arg == "arg":
            t = MemoryMappedTensor.empty(
                shape, dtype=dtype, device=device, filename=filename
            )
        elif shape_arg == "kwarg":
            t = MemoryMappedTensor.empty(
                shape=shape, dtype=dtype, device=device, filename=filename
            )
        assert t.shape == shape
        if dtype is not None:
            assert t.dtype is dtype
        if filename is not None:
            assert t.filename == filename

    @pytest.mark.parametrize("shape_arg", ["expand", "arg", "kwarg"])
    def test_full(self, shape, dtype, device, tmp_path, from_path, shape_arg):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        if device is not None and device.type != "cpu":
            with pytest.raises(RuntimeError):
                MemoryMappedTensor.full(
                    shape, fill_value=2, dtype=dtype, device=device, filename=filename
                )
            return
        if shape_arg == "expand":
            with pytest.raises(TypeError) if shape == () else nullcontext():
                t = MemoryMappedTensor.full(
                    *shape, fill_value=2, dtype=dtype, device=device, filename=filename
                )
            if shape == ():
                return
        elif shape_arg == "arg":
            t = MemoryMappedTensor.full(
                shape, fill_value=2, dtype=dtype, device=device, filename=filename
            )
        elif shape_arg == "kwarg":
            t = MemoryMappedTensor.full(
                shape=shape, fill_value=2, dtype=dtype, device=device, filename=filename
            )
        assert t.shape == shape
        if dtype is not None:
            assert t.dtype is dtype
        if filename is not None:
            assert t.filename == filename
        assert (t == 2).all()

    def test_zeros_like(self, shape, dtype, device, tmp_path, from_path):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        tensor = -torch.ones(shape, dtype=dtype, device=device)
        t = MemoryMappedTensor.zeros_like(tensor, filename=filename)
        assert t.shape == shape
        if dtype is not None:
            assert t.dtype is dtype
        if filename is not None:
            assert t.filename == filename
        assert (t == 0).all()

    def test_ones_like(self, shape, dtype, device, tmp_path, from_path):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        tensor = -torch.ones(shape, dtype=dtype, device=device)
        t = MemoryMappedTensor.ones_like(tensor, filename=filename)
        assert t.shape == shape
        if dtype is not None:
            assert t.dtype is dtype
        if filename is not None:
            assert t.filename == filename
        assert (t == 1).all()

    def test_full_like(self, shape, dtype, device, tmp_path, from_path):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        tensor = -torch.ones(shape, dtype=dtype, device=device)
        t = MemoryMappedTensor.full_like(tensor, 2, filename=filename)
        assert t.shape == shape
        if dtype is not None:
            assert t.dtype is dtype
        if filename is not None:
            assert t.filename == filename
        assert (t == 2).all()

    def test_from_filename(self, shape, dtype, device, tmp_path, from_path):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        if dtype is None:
            dtype = torch.float32
        tensor = -torch.randint(10, shape, dtype=dtype, device=device)
        t = MemoryMappedTensor.full_like(tensor, 2, filename=filename)
        if filename is not None:
            t2 = MemoryMappedTensor.from_filename(filename, dtype=dtype, shape=shape)
        else:
            t2 = MemoryMappedTensor.from_handler(
                t._handler, dtype=dtype, shape=shape, index=None
            )
        torch.testing.assert_close(t, t2)


class TestIndexing:
    @staticmethod
    def _recv_and_send(
        queue_out,
        queue_in,
        filename,
        shape,
    ):
        t = queue_in.get(timeout=TIMEOUT)
        assert isinstance(t, MemoryMappedTensor)
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

    def test_simple_index(self):
        t = MemoryMappedTensor.from_tensor(torch.zeros(10))
        # int
        assert isinstance(t[0], MemoryMappedTensor)
        assert t[0]._filename == t._filename
        assert t[0].shape == torch.Size([])
        assert t.shape == torch.Size([10])

    def test_range_index(self):
        t = MemoryMappedTensor.from_tensor(torch.zeros(10))
        # int
        assert isinstance(t[:2], MemoryMappedTensor)
        assert t[:2]._filename == t._filename
        assert t[:2].shape == torch.Size([2])
        assert t.shape == torch.Size([10])

    def test_double_index(self):
        t = MemoryMappedTensor.from_tensor(torch.zeros(10))
        y = t[:2]
        # int
        assert isinstance(y, MemoryMappedTensor)
        assert y._filename == t._filename
        assert y._handler is t._handler
        assert y.shape == torch.Size([2])
        assert t.shape == torch.Size([10])
        y = y[:1]
        # int
        assert isinstance(y, MemoryMappedTensor)
        assert y._filename == t._filename
        assert y._handler is t._handler
        assert y.shape == torch.Size([1])
        assert t.shape == torch.Size([10])

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_send_across_procs(self, tmp_path):
        t = MemoryMappedTensor.from_tensor(
            torch.zeros(10), filename=tmp_path / "tensor.memmap"
        )
        queue_in = mp.Queue(1)
        queue_out = mp.Queue(1)
        filename = t._filename
        assert filename is not None
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
    def test_send_across_procs_index(self, tmp_path):
        t = MemoryMappedTensor.from_tensor(
            torch.zeros(10), filename=tmp_path / "tensor.memmap"
        )
        queue_in = mp.Queue(1)
        queue_out = mp.Queue(1)
        filename = t._filename
        assert filename is not None
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
        t = MemoryMappedTensor.from_tensor(torch.rand(10))
        for i, _t in enumerate(t):
            assert _t == t[i]

    def test_iteration_nd(self):
        t = MemoryMappedTensor.from_tensor(torch.rand(10, 5))
        for i, _t in enumerate(t):
            assert (_t == t[i]).all()

    @staticmethod
    def _test_copy_onto_subproc(queue):
        t = MemoryMappedTensor.from_tensor(torch.rand(10, 5))
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
    y = MemoryMappedTensor.from_tensor(
        torch.zeros((), dtype=torch.uint8).expand(num_samples, rows, cols)
    )
    y.copy_(y + torch.randn(num_samples, rows, cols))
    assert isinstance(y, MemoryMappedTensor)
    idx = slice(3)
    assert isinstance(y[idx], MemoryMappedTensor)
    assert (y[idx] == y.clone()[idx]).all()


def test_filename(tmp_path):
    mt = MemoryMappedTensor.from_tensor(
        torch.zeros((), dtype=torch.float32).expand(10),
        filename=tmp_path / "test.memmap",
    )
    assert str(mt._filename) == str(tmp_path / "test.memmap")

    # memmap -> memmap keeps the filename
    mt2 = MemoryMappedTensor.from_tensor(mt, filename=mt.filename)
    assert str(mt2._filename) == str(tmp_path / "test.memmap")
    assert mt2 is mt

    # memmap -> memmap keeps id if no filename
    mt0 = MemoryMappedTensor.from_tensor(
        torch.zeros((), dtype=torch.float32).expand(10)
    )
    mt3 = MemoryMappedTensor.from_tensor(mt0)
    assert mt3 is mt0

    # memmap -> memmap with a new filename
    with pytest.raises(RuntimeError, match="copy_existing"):
        MemoryMappedTensor.from_tensor(mt, filename=tmp_path / "test2.memmap")
    mt4 = MemoryMappedTensor.from_tensor(
        mt, filename=tmp_path / "test2.memmap", copy_existing=True
    )
    assert str(mt4._filename) == str(tmp_path / "test2.memmap")
    assert mt4 is not mt

    del mt
    del mt4
    # files should persist
    assert (tmp_path / "test.memmap").exists()
    assert (tmp_path / "test2.memmap").exists()


def test_handler():
    mt = MemoryMappedTensor.from_tensor(torch.zeros((), dtype=torch.float32).expand(10))

    mt2 = MemoryMappedTensor.from_tensor(mt)
    assert mt2._handler is mt._handler


def test_memmap_from_memmap():
    mt = MemoryMappedTensor.from_tensor(torch.zeros(()).expand(4, 3, 2, 1))
    mt2 = MemoryMappedTensor.from_tensor(mt)
    assert mt2.squeeze(-1).shape == torch.Size([4, 3, 2])


def test_memmap_cast():
    # ensure memmap can be cast to tensor and viceversa
    x = torch.zeros(3, 4, 5)
    y = MemoryMappedTensor.from_tensor(torch.ones(3, 4, 5))

    x[:2] = y[:2]
    assert (x[:2] == 1).all()
    y[2:] = x[2:]
    assert (y[2:] == 0).all()


@pytest.fixture
def dummy_memmap():
    return MemoryMappedTensor.from_tensor(torch.randn(10, 11))


class TestOps:
    def test_eq(self, dummy_memmap):
        memmap = dummy_memmap
        assert (memmap == memmap.clone()).all()
        assert (memmap.clone() == memmap).all()

    def test_fill_(self, dummy_memmap):
        memmap = dummy_memmap.fill_(1.0)
        assert (memmap == 1).all()
        assert isinstance(memmap, MemoryMappedTensor)

    def test_copy_(self, dummy_memmap):
        memmap = dummy_memmap.copy_(torch.ones(10, 11))
        assert (memmap == 1).all()
        assert isinstance(memmap, MemoryMappedTensor)
        # check that memmap can be put in a tensor
        assert (torch.ones(10, 11).copy_(memmap) == 1).all()

    def test_or(self):
        memmap = MemoryMappedTensor.from_tensor(torch.ones(10, 11, dtype=torch.bool))
        assert (memmap | (~memmap)).all()

    def test_ne(self):
        memmap = MemoryMappedTensor.from_tensor(torch.ones(10, 11, dtype=torch.bool))
        assert (memmap != ~memmap).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
