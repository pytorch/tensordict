# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import os
import stat
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from _utils_internal import get_available_devices
from tensordict import TensorDict

from tensordict.memmap import _is_writable, MemoryMappedTensor
from torch import multiprocessing as mp

TIMEOUT = 100

HAS_NESTED_TENSOR = (
    getattr(torch, "_nested_compute_contiguous_strides_offsets", None) is not None
)


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
            assert t.filename == str(Path(filename).absolute())
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
            assert t.filename == str(Path(filename).absolute())
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
            assert t.filename == str(Path(filename).absolute())

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
            assert t.filename == str(Path(filename).absolute())
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
            assert t.filename == str(Path(filename).absolute())
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
            assert t.filename == str(Path(filename).absolute())
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
            assert t.filename == str(Path(filename).absolute())
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


class TestNestedTensor:
    shape = torch.tensor([[2, 3], [2, 4], [3, 2]])

    @pytest.mark.skipif(not HAS_NESTED_TENSOR, reason="Nested tensor incomplete")
    def test_with_filename(self, tmpdir):
        filename = tmpdir + "/test_file2.memmap"
        tensor = MemoryMappedTensor.empty(
            self.shape, filename=filename, dtype=torch.int
        )
        assert isinstance(tensor, MemoryMappedTensor)
        assert tensor.dtype == torch.int
        tensor.fill_(2)
        assert (tensor[0] == 2).all()
        assert tensor.filename is not None

        filename = tmpdir + "/test_file0.memmap"
        tensor = MemoryMappedTensor.zeros(
            self.shape, filename=filename, dtype=torch.bool
        )
        assert isinstance(tensor, MemoryMappedTensor)
        assert tensor.dtype == torch.bool
        assert tensor.filename is not None

        filename = tmpdir + "/test_file1.memmap"
        tensor = MemoryMappedTensor.ones(self.shape, filename=filename, dtype=torch.int)
        assert type(tensor) is MemoryMappedTensor
        assert tensor.dtype == torch.int
        assert (tensor[0] == 1).all()
        assert tensor.filename is not None

        filename = tmpdir + "/test_file3.memmap"
        tensor = torch.nested.nested_tensor(
            [torch.zeros(shape.tolist()) + i for i, shape in enumerate(self.shape)]
        )
        memmap_tensor = MemoryMappedTensor.from_tensor(tensor, filename=filename)
        assert type(memmap_tensor) is MemoryMappedTensor
        for t1, t2 in zip(tensor, memmap_tensor):
            assert t1.dtype == t2.dtype
            assert (t1 == t2).all()

        memmap_tensor2 = MemoryMappedTensor.from_filename(
            filename, dtype=memmap_tensor.dtype, shape=self.shape
        )
        assert type(memmap_tensor2) is MemoryMappedTensor
        for t1, t2 in zip(memmap_tensor2, memmap_tensor):
            assert t1.dtype == t2.dtype
            assert (t1 == t2).all()

    @pytest.mark.skipif(not HAS_NESTED_TENSOR, reason="Nested tensor incomplete")
    def test_with_handler(self):
        tensor = MemoryMappedTensor.empty(self.shape, dtype=torch.int)
        assert isinstance(tensor, MemoryMappedTensor)
        assert tensor.dtype == torch.int
        tensor.fill_(2)
        assert (tensor[0] == 2).all()
        assert tensor._handler is not None

        tensor = MemoryMappedTensor.zeros(self.shape, dtype=torch.bool)
        assert isinstance(tensor, MemoryMappedTensor)
        assert tensor.dtype == torch.bool
        assert tensor._handler is not None

        tensor = MemoryMappedTensor.ones(self.shape, dtype=torch.int)
        assert type(tensor) is MemoryMappedTensor
        assert tensor.dtype == torch.int
        assert (tensor[0] == 1).all()
        assert tensor._handler is not None

        tensor = torch.nested.nested_tensor(
            [torch.zeros(shape.tolist()) + i for i, shape in enumerate(self.shape)]
        )
        memmap_tensor = MemoryMappedTensor.from_tensor(tensor)
        assert type(memmap_tensor) is MemoryMappedTensor
        for t1, t2 in zip(tensor, memmap_tensor):
            assert t1.dtype == t2.dtype
            assert (t1 == t2).all()

        memmap_tensor2 = MemoryMappedTensor.from_handler(
            memmap_tensor._handler, dtype=memmap_tensor.dtype, shape=self.shape
        )
        assert type(memmap_tensor2) is MemoryMappedTensor
        for t1, t2 in zip(memmap_tensor2, memmap_tensor):
            assert t1.dtype == t2.dtype
            assert (t1 == t2).all()

    @pytest.mark.skipif(not HAS_NESTED_TENSOR, reason="Nested tensor incomplete")
    @pytest.mark.parametrize("with_filename", [False, True])
    def test_from_storage(self, with_filename, tmpdir):
        if with_filename:
            filename = Path(tmpdir) / "file.memmap"
            filename = str(filename)
        else:
            filename = None
        a = MemoryMappedTensor.from_tensor(
            torch.arange(10, dtype=torch.float64), filename=filename
        )
        assert type(a) is MemoryMappedTensor
        shape = torch.tensor([[2, 2], [2, 3]])
        b = MemoryMappedTensor.from_storage(
            a.untyped_storage(), filename=filename, shape=shape, dtype=a.dtype
        )
        assert type(b) is MemoryMappedTensor
        assert (b._nested_tensor_size() == shape).all()
        assert (b[0] == torch.arange(4).view(2, 2)).all()
        assert (b[1] == torch.arange(4, 10).view(2, 3)).all()

    @pytest.mark.skipif(not HAS_NESTED_TENSOR, reason="Nested tensor incomplete")
    def test_save_td_with_nested(self, tmpdir):
        td = TensorDict(
            {
                "a": torch.nested.nested_tensor(
                    [
                        torch.arange(12, dtype=torch.float64).view(3, 4),
                        torch.arange(15, dtype=torch.float64).view(3, 5),
                    ]
                )
            },
            batch_size=[2, 3],
        )
        tdsave = td.clone()
        td.memmap(tmpdir)
        del td
        gc.collect()
        td = TensorDict.load(tmpdir)
        for i in range(2):
            for j in range(3):
                assert (td[i, j] == tdsave[i, j]).all()


class TestReadWrite:
    @pytest.mark.skipif(os.getuid() == 0, reason="root can write to read-only files")
    def test_read_only(self, tmpdir):
        tmpdir = Path(tmpdir)
        file_path = tmpdir / "elt.mmap"
        mmap = MemoryMappedTensor.from_filename(
            filename=file_path, shape=[2, 3], dtype=torch.float64
        )
        mmap.copy_(torch.arange(6).view(2, 3))

        file_path = str(file_path.absolute())

        assert _is_writable(file_path)
        # Modify the permissions field to set the desired permissions
        new_permissions = stat.S_IREAD  # | stat.S_IWRITE | stat.S_IEXEC

        # change permission
        os.chmod(file_path, new_permissions)

        # Get the current file status
        assert not _is_writable(file_path)

        del mmap

        # load file
        mmap = MemoryMappedTensor.from_filename(
            filename=file_path, shape=[2, 3], dtype=torch.float64
        )
        assert (mmap.reshape(-1) == torch.arange(6)).all()

    @pytest.mark.skipif(not HAS_NESTED_TENSOR, reason="Nested tensor incomplete")
    @pytest.mark.skipif(os.getuid() == 0, reason="root can write to read-only files")
    def test_read_only_nested(self, tmpdir):
        tmpdir = Path(tmpdir)
        file_path = tmpdir / "elt.mmap"
        data = MemoryMappedTensor.from_tensor(torch.arange(26), filename=file_path)
        mmap = MemoryMappedTensor.from_storage(
            data.untyped_storage(),
            filename=file_path,
            shape=torch.tensor([[2, 3], [4, 5]]),
            dtype=data.dtype,
        )

        file_path = str(file_path.absolute())
        assert _is_writable(file_path)

        # Modify the permissions field to set the desired permissions
        new_permissions = stat.S_IREAD  # | stat.S_IWRITE | stat.S_IEXEC

        # change permission
        os.chmod(file_path, new_permissions)

        # Get the current file status
        assert not _is_writable(file_path)

        # load file
        mmap1 = MemoryMappedTensor.from_filename(
            filename=file_path, shape=torch.tensor([[2, 3], [4, 5]]), dtype=data.dtype
        )
        assert (mmap1[0].view(-1) == torch.arange(6)).all()
        assert (mmap1[1].view(-1) == torch.arange(6, 26)).all()
        # test filename
        assert mmap1.filename == mmap.filename
        assert mmap1.filename == data.filename
        assert mmap1.filename == data.untyped_storage().filename
        # assert mmap1.untyped_storage().filename == data.untyped_storage().filename

        # os.chmod(str(file_path), 0o444)
        # data.fill_(0)
        # os.chmod(str(file_path), 0o444)
        #
        # assert (mmap1[0].view(-1) == 0).all()
        # assert (mmap1[1].view(-1) == 0).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
