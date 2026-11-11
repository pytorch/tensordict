# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import io
import os
import pickle
import stat
import sys
import time
from contextlib import nullcontext
from multiprocessing.reduction import ForkingPickler
from pathlib import Path

import pytest
import torch
from _utils_internal import get_available_devices
from tensordict import (
    LazyStackedTensorDict,
    NonTensorData,
    NonTensorStack,
    tensorclass,
    TensorDict,
)
from tensordict._base import serialization as td_serialization
from tensordict.memmap import _is_writable, MemoryMappedTensor
from torch import multiprocessing as mp

TIMEOUT = 100
# Windows has no os.getuid, and no root that can write to read-only files
_IS_ROOT = hasattr(os, "getuid") and os.getuid() == 0


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
        torch.complex64,
        torch.complex128,
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


def test_existing_symlink_is_rejected(tmp_path):
    target = tmp_path / "target"
    target.write_bytes(b"unchanged")
    link = tmp_path / "file.memmap"
    link.symlink_to(target)

    with pytest.raises(RuntimeError, match="through symlink"):
        MemoryMappedTensor.from_tensor(
            torch.zeros(()),
            filename=link,
            existsok=True,
        )

    assert target.read_bytes() == b"unchanged"


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
            t2 = MemoryMappedTensor._from_handler(
                t._handler, dtype=dtype, shape=shape, index=None
            )
        torch.testing.assert_close(t, t2)

    def test_from_storage(self, shape, dtype, device, tmp_path, from_path):
        if from_path:
            filename = tmp_path / "file.memmap"
        else:
            filename = None
        if dtype is None:
            dtype = torch.float32
        tensor = -torch.randint(10, shape, dtype=dtype, device=device)
        t = MemoryMappedTensor.from_tensor(tensor, filename=filename)
        t2 = MemoryMappedTensor.from_storage(
            t.untyped_storage(), shape=shape, dtype=dtype
        )
        assert type(t2) is MemoryMappedTensor
        assert t2.shape == shape
        assert t2.dtype is dtype
        assert t2.data_ptr() == t.data_ptr()
        if filename is None:
            # a storage without a file still cannot be pickled
            return
        assert t2.filename == t.filename
        # pickles by its file, as t does, also when indexed
        index = (slice(1, None),) * len(shape)
        for view in (t2, t2[index]):
            loaded = pickle.loads(pickle.dumps(view))
            assert loaded.filename == t.filename
            torch.testing.assert_close(loaded, view)
        loaded.fill_(3)
        assert (t[index] == 3).all()


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

    @staticmethod
    def _check_view(parent, view, expected):
        # the view is on the file or the shared memory of its parent
        assert type(view) is MemoryMappedTensor
        assert view._filename == parent._filename
        assert view._handler is parent._handler
        assert view._mode == parent._mode
        assert torch.as_tensor(view._parent_shape).equal(
            torch.as_tensor(parent._parent_shape)
        )
        assert view.shape == expected.shape
        assert (view == expected).all()
        if parent._filename is None:
            with pytest.raises(RuntimeError, match="no file associated"):
                view.filename
            if os.name == "nt":
                # the handler pickles through a fd
                return
            loaded = pickle.loads(ForkingPickler.dumps(view))
        else:
            assert view.filename == parent.filename
            # the form that the filename setter stores
            assert view._filename == str(Path(view._filename).absolute())
            loaded = pickle.loads(pickle.dumps(view))
        assert type(loaded) is MemoryMappedTensor
        assert loaded._filename == parent._filename
        assert loaded._mode == parent._mode
        assert loaded.shape == expected.shape
        assert (loaded == expected).all()

    @pytest.mark.parametrize(
        "make_view",
        [
            lambda t: t[1],
            lambda t: t[1:3],
            lambda t: t[1, :, 1],
            lambda t: t[..., 0],
            lambda t: t[None],
            lambda t: t[::2],
            lambda t: t[1:][0][1:],
            lambda t: t.unbind(0)[2],
            lambda t: t.unbind(1)[1],
            lambda t: t.chunk(2, dim=1)[1],
            lambda t: t.split(1, dim=2)[1],
            lambda t: t[1:].unbind(1)[0],
            lambda t: t[:, 1:].chunk(2)[1],
            lambda t: t.split(3)[0][2],
        ],
        ids=[
            "int",
            "slice",
            "tuple",
            "ellipsis",
            "none",
            "step",
            "view-of-view",
            "unbind",
            "unbind-dim1",
            "chunk",
            "split",
            "slice-unbind",
            "slice-chunk",
            "split-int",
        ],
    )
    @pytest.mark.parametrize(
        "source",
        ["file", "anonymous", "r", "r+", "unpickled", "td", "loaded-td", "atomic-td"],
    )
    def test_view_keeps_file(self, make_view, source, tmp_path):
        # #3570: a view takes the file name of its parent as it is, and it has
        # the file, the handler and the mode that it had through the setter
        data = torch.arange(24).view(4, 3, 2)
        if source == "anonymous":
            t = MemoryMappedTensor.from_tensor(data)
        elif source in ("td", "loaded-td", "atomic-td"):
            if source == "atomic-td" and sys.platform == "win32":
                pytest.skip("atomic=True is not supported on Windows")
            td = TensorDict(a=data, batch_size=[4])
            td = td.memmap(tmp_path / "td", atomic=source == "atomic-td")
            if source == "loaded-td":
                td = TensorDict.load_memmap(tmp_path / "td")
            t = td["a"]
        else:
            filename = tmp_path / "tensor.memmap"
            t = MemoryMappedTensor.from_tensor(data, filename=filename)
            if source == "unpickled":
                t = pickle.loads(pickle.dumps(t))
            elif source != "file":
                t = MemoryMappedTensor.from_filename(
                    filename, data.dtype, data.shape, mode=source
                )
        assert (t._filename is None) == (source == "anonymous")
        self._check_view(t, make_view(t), make_view(data))

    @pytest.mark.parametrize("with_filename", [False, True])
    def test_view_keeps_file_nested(self, with_filename, tmp_path):
        # #3570: the same for the views of a nested tensor
        tensors = [torch.arange(6).view(2, 3), torch.arange(8).view(2, 4)]
        filename = tmp_path / "nested.memmap" if with_filename else None
        t = MemoryMappedTensor.from_tensor(
            torch.nested.nested_tensor(tensors), filename=filename
        )
        assert (t._filename is None) == (not with_filename)
        self._check_view(t, t[1], tensors[1])
        self._check_view(t, t[1][1:], tensors[1][1:])
        self._check_view(t, t.unbind(0)[0][:, 1], tensors[0][:, 1])
        self._check_view(t, t[..., 0][1], tensors[1][:, 0])

    def test_view_keeps_file_after_chdir(self, tmp_path, monkeypatch):
        # #3570: the file name of a view is absolute, like its parent's: it
        # does not follow the working directory
        (tmp_path / "saved").mkdir()
        (tmp_path / "other").mkdir()
        monkeypatch.chdir(tmp_path / "saved")
        data = torch.arange(6).view(2, 3)
        t = MemoryMappedTensor.from_tensor(data, filename="tensor.memmap")
        td = TensorDict(a=data, batch_size=[2]).memmap("td")
        filename = t.filename
        td_filename = td["a"].filename
        assert os.path.isabs(filename)
        assert os.path.isabs(td_filename)
        assert os.path.samefile(filename, tmp_path / "saved" / "tensor.memmap")
        assert os.path.samefile(td_filename, tmp_path / "saved" / "td" / "a.memmap")
        monkeypatch.chdir(tmp_path / "other")
        views = (t[1], t[:1], t[1:][0], *t.unbind(0), *t.chunk(2), *t.split(1), *t)
        for view in views:
            assert view.filename == filename
        assert pickle.loads(pickle.dumps(t[1])).filename == filename
        assert (pickle.loads(pickle.dumps(t[1])) == data[1]).all()
        for sub in (td[1], td[:1], *td.unbind(0)):
            assert sub["a"].filename == td_filename
        assert not os.listdir(tmp_path / "other")

    def test_view_does_not_call_pathlib(self, tmp_path, monkeypatch):
        # #3570: a view has the file name of its parent, which is absolute
        # already, so only opening a file goes through pathlib
        data = torch.arange(24).view(4, 3, 2)
        t = MemoryMappedTensor.from_tensor(data, filename=tmp_path / "tensor.memmap")
        td = TensorDict(a=data, b=data + 1, batch_size=[4]).memmap(tmp_path / "td")
        calls = []

        def counting_path(*args):
            calls.append(args)
            return Path(*args)

        # tensordict.memmap is also a function of the package
        monkeypatch.setattr(sys.modules["tensordict.memmap"], "Path", counting_path)
        views = [t[1], t[1:3], t[1][0], t[..., 0], *t]
        views += [*t.unbind(1), *t.chunk(2), *t.split(1, dim=2)]
        views += [td[1]["a"], td[1:3]["b"], *(sub["a"] for sub in td.unbind(0))]
        for view in views:
            assert type(view) is MemoryMappedTensor
            assert view.filename in (t.filename, td["a"].filename, td["b"].filename)
        assert not calls
        loaded = MemoryMappedTensor.from_filename(t.filename, data.dtype, data.shape)
        assert loaded.filename == t.filename
        assert calls

    @pytest.mark.parametrize(
        "make_view",
        [
            lambda t: t[1][2],
            lambda t: t[1:][0],
            lambda t: t[:, 1:][:, 1],
            lambda t: t[..., ::2][1, :, 1:],
            lambda t: t[1:][0][1:][::2],
            lambda t: t[None][0, -1],
            lambda t: t[None],
            lambda t: t[1].unbind(0)[2],
            lambda t: t[:, 1:].unbind(1)[1],
            lambda t: t[1:].chunk(2, dim=2)[1],
            lambda t: t[:, 1:].split(1, dim=1)[1],
        ],
        ids=[
            "int-int",
            "slice-int",
            "slices",
            "step",
            "four-indices",
            "none-int",
            "none",
            "unbind",
            "unbind-dim1",
            "chunk",
            "split",
        ],
    )
    @pytest.mark.parametrize("with_filename", [False, True])
    def test_pickle_view_of_view(self, make_view, with_filename, tmp_path):
        if not with_filename and os.name == "nt":
            pytest.skip("the handler pickles through a fd")
        data = torch.arange(24).view(2, 3, 4)
        filename = tmp_path / "tensor.memmap" if with_filename else None
        t = MemoryMappedTensor.from_tensor(data, filename=filename)
        view = make_view(t)
        hows = ["forking", "pickle", "torch.save"] if with_filename else ["forking"]
        for how in hows:
            loaded = self._pickle_and_load(view, how)
            assert isinstance(loaded, MemoryMappedTensor)
            assert loaded.shape == view.shape
            assert (loaded == make_view(data)).all()
        # it is a view of the same memory
        loaded.fill_(-1)
        assert (view == -1).all()
        assert (t == -1).sum() == view.numel()

    @staticmethod
    def _pickle_and_load(tensor, how):
        if how == "forking":
            return pickle.loads(ForkingPickler.dumps(tensor))
        if how == "pickle":
            return pickle.loads(pickle.dumps(tensor))
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        return torch.load(buffer, weights_only=False)

    def test_pickle_td_view_of_view(self, tmp_path):
        td = TensorDict(a=torch.arange(12).view(3, 4), batch_size=[3, 4])
        sub = td.memmap(tmp_path)[1:][1:]
        loaded = pickle.loads(pickle.dumps(sub))
        assert loaded.batch_size == sub.batch_size
        assert loaded["a"].shape == sub["a"].shape
        assert (loaded["a"] == td["a"][2:]).all()

    @pytest.mark.parametrize("with_filename", [False, True])
    def test_pickle_empty_view_of_view(self, with_filename, tmp_path):
        if not with_filename and os.name == "nt":
            pytest.skip("the handler pickles through a fd")
        filename = tmp_path / "tensor.memmap" if with_filename else None
        t = MemoryMappedTensor.from_tensor(
            torch.arange(12).view(3, 4), filename=filename
        )
        hows = ["forking", "pickle", "torch.save"] if with_filename else ["forking"]
        # no index selects the first two from t: they slice a dim that None
        # added to length 0
        for view in (t[2][None][3:], t[None][1:], t[1:][5:]):
            for how in hows:
                loaded = self._pickle_and_load(view, how)
                assert isinstance(loaded, MemoryMappedTensor)
                assert loaded.shape == view.shape
                assert loaded.dtype == view.dtype

    def test_pickle_td_empty_view_of_view(self, tmp_path):
        td = TensorDict(a=torch.arange(12).view(3, 4), batch_size=[3])
        sub = td.memmap(tmp_path / "td")[2][None][3:]
        loaded = pickle.loads(pickle.dumps(sub))
        assert loaded.batch_size == torch.Size([0])
        assert loaded["a"].shape == torch.Size([0, 4])
        # torch gives the storage of a tensor with no elements the data_ptr 0,
        # so indexing it with a tensor gives a copy that is wrapped as a view
        td = TensorDict(
            obs=torch.arange(12.0).view(4, 3), info=torch.zeros(4, 0), batch_size=[4]
        ).memmap(tmp_path / "empty_leaf")
        sample = td[torch.tensor([0, 2])]
        for view in (sample[1], sample.unbind(0)[1]):
            loaded = pickle.loads(pickle.dumps(view))
            assert loaded["info"].shape == torch.Size([0])
            assert (loaded["obs"] == td["obs"][2]).all()
        loaded = pickle.loads(pickle.dumps(pickle.loads(pickle.dumps(sample))[0]))
        assert loaded["info"].shape == torch.Size([0])

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

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    def test_send_across_procs_split(self, tmp_path):
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
            queue_out.put(t.split((3, 3, 4), 0)[0], block=True)
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
    def test_send_across_procs_chunk(self, tmp_path):
        t = MemoryMappedTensor.from_tensor(
            torch.zeros(9), filename=tmp_path / "tensor.memmap"
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
            queue_out.put(t.chunk(3, 0)[0], block=True)
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
    def _test_copy_onto_subproc(queue_out, queue_in):
        t = MemoryMappedTensor.from_tensor(torch.rand(10, 5))
        idx = torch.tensor([1, 2])
        t_indexed1 = t[idx]
        queue_out.put(t_indexed1, block=True)

        idx = torch.tensor([3, 4])
        t_indexed2 = t[idx]
        queue_out.put(t_indexed2, block=True)
        msg = queue_in.get(timeout=TIMEOUT)
        assert msg == "done"
        assert (t_indexed1 == t_indexed2).all()
        del queue_out, queue_in

    def test_copy_onto(self):
        queue_in = mp.Queue(1)
        queue_out = mp.Queue(1)
        p = mp.Process(
            target=TestIndexing._test_copy_onto_subproc, args=(queue_in, queue_out)
        )
        p.start()
        try:
            t_indexed1 = queue_in.get(timeout=TIMEOUT)

            # receive 2nd copy
            t_indexed2 = queue_in.get(timeout=TIMEOUT)
            t_indexed1.copy_(t_indexed2)
            _ = t_indexed2 + 1
            queue_out.put("done", block=True)
            queue_out.close()
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


def test_constructor_filename_source_and_index(tmp_path):
    # https://github.com/pytorch/tensordict/issues/2970
    filename = str(tmp_path / "x.memmap")
    MemoryMappedTensor.from_tensor(torch.arange(6.0), filename=filename)

    # a str source maps the file, as from_filename does
    x = MemoryMappedTensor(filename, dtype=torch.float32, shape=(6,))
    assert type(x) is MemoryMappedTensor
    assert x.filename == filename
    torch.testing.assert_close(x, torch.arange(6.0))
    x[0] = 9.0
    y = MemoryMappedTensor.from_filename(filename, torch.float32, (6,))
    assert y[0] == 9.0
    loaded = pickle.loads(pickle.dumps(x))
    torch.testing.assert_close(loaded, y)

    # index on every route
    x = MemoryMappedTensor(filename, dtype=torch.float32, shape=(2, 3), index=1)
    torch.testing.assert_close(x, torch.tensor([3.0, 4.0, 5.0]))
    x = MemoryMappedTensor(
        torch.arange(6.0), filename=tmp_path / "y.memmap", index=slice(2, 4)
    )
    torch.testing.assert_close(x, torch.tensor([2.0, 3.0]))
    x = MemoryMappedTensor(
        y.untyped_storage(), dtype=torch.float32, shape=(6,), index=slice(1, 3)
    )
    torch.testing.assert_close(x, torch.tensor([1.0, 2.0]))
    x = MemoryMappedTensor(torch.arange(6.0), index=-1)
    assert type(x) is MemoryMappedTensor
    assert x == 5.0

    with pytest.raises(TypeError, match="Duplicated filename argument"):
        MemoryMappedTensor(
            filename, dtype=torch.float32, shape=(6,), filename=tmp_path / "z"
        )
    with pytest.raises(TypeError, match="dtype and shape are required"):
        MemoryMappedTensor(filename)


def test_handler():
    mt = MemoryMappedTensor.from_tensor(torch.zeros((), dtype=torch.float32).expand(10))

    mt2 = MemoryMappedTensor.from_tensor(mt)
    assert mt2._handler is mt._handler


def test_index_and_parent_shape_are_deprecated():
    mt = MemoryMappedTensor.from_tensor(torch.zeros(4, 3))[1:]
    with pytest.warns(
        DeprecationWarning,
        match=r"^MemoryMappedTensor\.index is deprecated and will be removed in "
        r"TensorDict 0\.17\.$",
    ) as record:
        assert mt.index == slice(1, None)
    assert record[0].filename == __file__
    with pytest.warns(
        DeprecationWarning,
        match=r"^MemoryMappedTensor\.parent_shape is deprecated and will be removed "
        r"in TensorDict 0\.17\.$",
    ) as record:
        assert mt.parent_shape == torch.Size([4, 3])
    assert record[0].filename == __file__


def test_from_handler_is_deprecated():
    mt = MemoryMappedTensor.from_tensor(torch.arange(6.0).view(2, 3))
    with pytest.warns(
        DeprecationWarning,
        match=r"^MemoryMappedTensor\.from_handler\(\) is deprecated and will be "
        r"removed in TensorDict 0\.17\.$",
    ) as record:
        mt2 = MemoryMappedTensor.from_handler(mt._handler, torch.float32, [2, 3], 1)
    assert record[0].filename == __file__
    assert isinstance(mt2, MemoryMappedTensor)
    assert mt2._handler is mt._handler
    assert (mt2 == mt[1]).all()


class _OldHandlerPickle:
    # Pickles like a MemoryMappedTensor did with TensorDict 0.14.
    def __init__(self, args):
        self.args = args

    def __reduce__(self):
        return MemoryMappedTensor.from_handler, self.args


@pytest.mark.skipif(os.name == "nt", reason="the handler pickles through a fd")
def test_pickle_handler():
    mt = MemoryMappedTensor.from_tensor(torch.arange(6.0).view(2, 3))[1]
    func, args = mt.__reduce__()
    assert func == MemoryMappedTensor._from_handler
    loaded = pickle.loads(ForkingPickler.dumps(mt))
    assert isinstance(loaded, MemoryMappedTensor)
    assert loaded._index == 1
    assert loaded._parent_shape == torch.Size([2, 3])
    assert (loaded == mt).all()
    # pickles made with TensorDict 0.14 call the deprecated from_handler
    payload = ForkingPickler.dumps(_OldHandlerPickle(args))
    with pytest.warns(DeprecationWarning, match="from_handler"):
        loaded = pickle.loads(payload)
    assert isinstance(loaded, MemoryMappedTensor)
    assert (loaded == mt).all()


@pytest.mark.skipif(os.name == "nt", reason="the handler pickles through a fd")
def test_pickle_handler_complex():
    tensor = torch.complex(torch.arange(6.0), -torch.arange(6.0)).view(2, 3)
    mt = MemoryMappedTensor.from_tensor(tensor)[1]
    loaded = pickle.loads(ForkingPickler.dumps(mt))
    assert loaded.dtype == torch.complex64
    assert (loaded == tensor[1]).all()


@pytest.mark.skipif(sys.platform != "linux", reason="counts /proc/self/fd")
def test_pickle_handler_closes_fd():
    # #3565: a tensor without a file that is received from another process
    # closes the descriptor it received when it is dropped
    def round_trip():
        td = TensorDict(obs=torch.zeros(8, 3), reward=torch.zeros(8, 1), batch_size=[8])
        td.memmap_()
        loaded = pickle.loads(ForkingPickler.dumps(td))
        assert (loaded["obs"] == 0).all()

    def open_fds():
        return len(os.listdir("/proc/self/fd"))

    # the first round trip starts the thread that sends the descriptors
    round_trip()
    gc.collect()
    start = open_fds()
    for _ in range(50):
        round_trip()
    gc.collect()
    # That thread closes its own copy of a descriptor after the receiver has
    # it: wait for it. A leak is 100 descriptors.
    for _ in range(200):
        if open_fds() <= start:
            break
        time.sleep(0.01)
    assert open_fds() <= start


def _recv_write_and_send_on(queue_out, queue_in):
    t = queue_in.get(timeout=TIMEOUT)
    assert isinstance(t, MemoryMappedTensor)
    assert (t == 0).all()
    queue_out.put("received")
    assert queue_in.get(timeout=TIMEOUT) == "written"
    assert (t == 1).all()
    t[1] = 2
    # the second hop, of a view: this process drops its tensors once they are sent
    queue_out.put(t[1:])
    del t
    gc.collect()
    assert queue_in.get(timeout=TIMEOUT) == "done"


@pytest.mark.skipif(os.name == "nt", reason="the handler pickles through a fd")
def test_pickle_handler_shares_memory_across_procs():
    # #3565: a received tensor reads and writes the sender's memory, also after
    # it is sent on and the tensors that it came from are dropped
    t = MemoryMappedTensor.from_tensor(torch.zeros(10))
    queue_in = mp.Queue(1)
    queue_out = mp.Queue(1)
    p = mp.Process(target=_recv_write_and_send_on, args=(queue_in, queue_out))
    p.start()
    try:
        queue_out.put(t, block=True)
        assert queue_in.get(timeout=TIMEOUT) == "received"
        t.fill_(1.0)
        queue_out.put("written", block=True)
        t2 = queue_in.get(timeout=TIMEOUT)
        queue_out.put("done", block=True)
    finally:
        p.join()
    assert p.exitcode == 0
    assert isinstance(t2, MemoryMappedTensor)
    assert t2.tolist() == [2.0] + [1.0] * 8
    assert t[1] == 2
    t2[1] = 3
    assert t[2] == 3
    # a third hop, after the process and the tensor of the second hop are gone
    t3 = pickle.loads(ForkingPickler.dumps(t2))
    del t2
    gc.collect()
    assert isinstance(t3, MemoryMappedTensor)
    t[3] = 4
    assert t3.tolist() == [2.0, 3.0, 4.0] + [1.0] * 6
    t3[3] = 5
    assert t.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0] + [1.0] * 5


# #3566: an anonymous memory map of a tensor with no elements raised
# "ValueError: cannot mmap an empty file"
@pytest.mark.parametrize("shape", [(0,), (3, 0), (0, 2), (2, 0, 3)])
@pytest.mark.parametrize(
    "dtype", [None, torch.float64, torch.int64, torch.bool, torch.complex64]
)
def test_empty_anonymous_constructors(shape, dtype):
    source = torch.zeros(shape, dtype=dtype)
    tensors = [
        MemoryMappedTensor.zeros(shape, dtype=dtype),
        MemoryMappedTensor.zeros(*shape, dtype=dtype),
        MemoryMappedTensor.zeros(shape=shape, dtype=dtype),
        MemoryMappedTensor.ones(shape, dtype=dtype),
        MemoryMappedTensor.empty(shape, dtype=dtype),
        MemoryMappedTensor.empty(*shape, dtype=dtype),
        MemoryMappedTensor.full(shape, fill_value=1, dtype=dtype),
        MemoryMappedTensor.from_tensor(source),
        MemoryMappedTensor.from_tensor(source, copy_data=False),
        MemoryMappedTensor.from_tensor(torch.zeros((), dtype=dtype), shape=shape),
        MemoryMappedTensor.zeros_like(source),
        MemoryMappedTensor.ones_like(source),
        MemoryMappedTensor.empty_like(source),
        MemoryMappedTensor.full_like(source, 1),
    ]
    for tensor in tensors:
        assert type(tensor) is MemoryMappedTensor
        assert tensor.shape == source.shape
        assert tensor.dtype == source.dtype
        assert tensor._handler is not None
        with pytest.raises(RuntimeError, match="has no file associated"):
            tensor.filename
        assert MemoryMappedTensor.from_tensor(tensor) is tensor
        assert tensor.clone().shape == source.shape
        tensor.fill_(1)
        tensor.copy_(source)
        assert torch.cat([tensor, source]).shape[0] == 2 * shape[0]


# #3566: memmap(), memmap_() and memmap_like() without a prefix raised when a
# leaf had no elements; they now give what the calls with a prefix give
@pytest.mark.parametrize("method", ["memmap", "memmap_", "memmap_like"])
@pytest.mark.parametrize("num_threads", [0, 2])
@pytest.mark.parametrize("batch", [4, 0])
def test_empty_anonymous_td(method, num_threads, batch, tmp_path):
    td = TensorDict(
        obs=torch.arange(batch * 3.0).view(batch, 3),
        empty=torch.zeros(batch, 0, dtype=torch.int64),
        nested={"empty": torch.zeros(batch, 0, 2)},
        batch_size=[batch],
    )
    anon = getattr(td.clone(), method)(num_threads=num_threads)
    with_file = getattr(td.clone(), method)(tmp_path / "td", num_threads=num_threads)
    assert anon.is_memmap()
    assert anon.batch_size == with_file.batch_size
    assert set(anon.keys(True, True)) == set(with_file.keys(True, True))
    for key in td.keys(True, True):
        assert type(anon[key]) is MemoryMappedTensor
        assert anon[key].shape == with_file[key].shape == td[key].shape
        assert anon[key].dtype == with_file[key].dtype == td[key].dtype
        assert anon[key]._handler is not None
    if method != "memmap_like":
        assert (anon["obs"] == td["obs"]).all()


# #3566: the leaf with no elements behaves like the one of a tensordict that is
# memory-mapped in a directory
@pytest.mark.parametrize("with_prefix", [False, True])
def test_empty_anonymous_td_ops(with_prefix, tmp_path):
    td = TensorDict(
        obs=torch.arange(12.0).view(4, 3), empty=torch.zeros(4, 0), batch_size=[4]
    )
    mm = td.memmap(tmp_path / "td" if with_prefix else None)
    for index, shape in ((slice(1, 3), (2, 0)), (2, (0,)), (None, (1, 4, 0))):
        assert type(mm[index]["empty"]) is MemoryMappedTensor
        assert mm[index]["empty"].shape == shape
    for index, rows in (
        (torch.tensor([0, 2]), 2),
        (torch.tensor([True, False, True, True]), 3),
    ):
        assert mm[index]["empty"].shape == (rows, 0)
        assert (mm[index]["obs"] == td["obs"][index]).all()
    assert mm["empty"][:, None].shape == (4, 1, 0)
    assert [t.shape for t in mm["empty"].unbind(0)] == [(0,)] * 4
    clone = mm.clone()
    assert clone["empty"].shape == (4, 0)
    assert not clone.is_memmap()
    assert torch.cat([mm, td])["empty"].shape == (8, 0)
    assert torch.stack([mm, td])["empty"].shape == (2, 4, 0)
    empty = mm["empty"]
    mm.update_(td + 1)
    mm[1:1] = td[1:1]
    mm[:2] = td[2:]
    mm[torch.tensor([3])] = td[:1]
    assert mm["empty"] is empty
    assert (
        mm["obs"] == td["obs"][[2, 3, 2, 0]] + torch.tensor([[0], [0], [1], [0]])
    ).all()
    if with_prefix:
        assert mm.memmap_refresh_()["empty"].shape == (4, 0)


# #3566: TorchRL's LazyMemmapStorage without a scratch_dir allocates its storage
# with memmap_like() and writes rows into it
def test_empty_anonymous_memmap_like_write():
    data = TensorDict(
        obs=torch.arange(6.0).view(2, 3), empty=torch.zeros(2, 0), batch_size=[2]
    )
    storage = data[0].clone().expand(10).memmap_like()
    assert storage.is_memmap()
    assert storage["empty"].shape == (10, 0)
    storage[torch.tensor([0, 1])] = data
    storage[2:4] = data
    storage[4] = data[1]
    sample = storage[torch.tensor([1, 0, 3, 4])]
    assert sample["empty"].shape == (4, 0)
    assert (sample["obs"] == data["obs"][[1, 0, 1, 1]]).all()
    # with an empty batch size, as the storage of a buffer that holds nothing
    storage = data[:0].memmap_like()
    assert storage.batch_size == (0,)
    assert storage["obs"].shape == (0, 3)
    storage[:] = data[:0]
    storage[torch.tensor([], dtype=torch.long)] = data[:0]
    assert storage[:0]["empty"].shape == (0, 0)


def _empty_anonymous_child(queue_in, queue_out):
    td = queue_in.get(timeout=TIMEOUT)
    td["obs"].fill_(1.0)
    queue_out.put(
        [
            (type(td[key]).__name__, tuple(td[key].shape), td[key].dtype)
            for key in ("obs", "empty")
        ]
        + [(type(leaf).__name__, tuple(leaf.shape)) for leaf in td[1:3][0].values()]
    )


# #3566: a memory map with no elements reaches another process like the other
# leaves of its tensordict
@pytest.mark.flaky(reruns=5, reruns_delay=5)
@pytest.mark.skipif(os.name == "nt", reason="the handler pickles through a fd")
@pytest.mark.parametrize("start_method", ["fork", "spawn"])
def test_empty_anonymous_send_across_procs(start_method):
    if start_method == "fork" and sys.platform != "linux":
        pytest.skip("fork is only safe on Linux")
    td = TensorDict(
        obs=torch.zeros(4, 3),
        empty=torch.zeros(4, 0, dtype=torch.int64),
        batch_size=[4],
    ).memmap()
    ctx = mp.get_context(start_method)
    queue_in = ctx.Queue(1)
    queue_out = ctx.Queue(1)
    p = ctx.Process(target=_empty_anonymous_child, args=(queue_out, queue_in))
    p.start()
    try:
        queue_out.put(td, block=True)
        assert queue_in.get(timeout=TIMEOUT) == [
            ("MemoryMappedTensor", (4, 3), torch.float32),
            ("MemoryMappedTensor", (4, 0), torch.int64),
            ("MemoryMappedTensor", (3,)),
            ("MemoryMappedTensor", (0,)),
        ]
        # the child wrote to the shared memory of the other leaf
        assert (td["obs"] == 1).all()
    finally:
        p.join()


# #3566: pickling a memory map with no elements, and its views
@pytest.mark.skipif(os.name == "nt", reason="the handler pickles through a fd")
def test_empty_anonymous_pickle():
    mt = MemoryMappedTensor.zeros((4, 0, 2), dtype=torch.int32)
    for view in (mt, mt[1:3], mt[2], mt[None], mt[1:3][0], mt[:, None][2:], mt[..., 0]):
        loaded = pickle.loads(ForkingPickler.dumps(view))
        assert type(loaded) is MemoryMappedTensor
        assert loaded.shape == view.shape
        assert loaded.dtype == torch.int32
        assert loaded._handler is not None
    # a handler that does not hold a tensor with no elements still raises
    mt = MemoryMappedTensor.zeros(4, 3)
    with pytest.raises(RuntimeError):
        MemoryMappedTensor._from_handler(mt._handler, torch.float32, (0, 3))


# #3566: an invalid call for a tensor with no elements keeps the error it had:
# the one of mapping 0 bytes
@pytest.mark.parametrize(
    "call",
    [
        lambda: MemoryMappedTensor.from_tensor(torch.zeros(3), shape=(2, 0)),
        lambda: MemoryMappedTensor.from_tensor(torch.tensor(3.0), shape=(-1, 0)),
        lambda: MemoryMappedTensor.from_tensor(torch.tensor(3.0), shape=(True, 0)),
        lambda: MemoryMappedTensor.from_tensor(torch.zeros(3, 0).to_sparse()),
        lambda: MemoryMappedTensor.from_tensor(torch.zeros(3, 0, device="meta")),
        lambda: MemoryMappedTensor.from_tensor(
            torch.quantize_per_tensor(torch.zeros(0), 1.0, 0, torch.qint8)
        ),
        lambda: MemoryMappedTensor.zeros((-1, 0)),
        lambda: MemoryMappedTensor.full((-1, 0), fill_value=1),
    ],
    ids=[
        "not_broadcastable",
        "ambiguous_shape",
        "bool_in_shape",
        "sparse",
        "meta",
        "quantized",
        "zeros_ambiguous_shape",
        "full_ambiguous_shape",
    ],
)
def test_empty_anonymous_invalid_call(call):
    handler_type = type(MemoryMappedTensor.zeros(1)._handler)
    with pytest.raises(Exception) as expected:
        handler_type(0)
    with pytest.raises(type(expected.value)) as raised:
        call()
    assert str(raised.value) == str(expected.value)


# #3566: a handler of one element with a shape or an index that does not fit a
# tensor with no elements keeps the error of reshaping one element
@pytest.mark.parametrize(
    "shape,index", [((0,), 0), ((0, 3), 0), ((0, 3), "a"), ((-1, 0), None)]
)
def test_empty_anonymous_from_handler_invalid(shape, index):
    handler = MemoryMappedTensor.zeros(1)._handler
    with pytest.raises(RuntimeError) as expected:
        torch.zeros(1).reshape(shape)
    with pytest.raises(RuntimeError) as raised:
        MemoryMappedTensor._from_handler(handler, torch.float32, shape, index)
    assert str(raised.value) == str(expected.value)


# #3566: what must not change. A memory map with elements maps exactly its
# elements, holds the same file descriptors and pickles as before, and a
# file-backed memory map with no elements still works.
@pytest.mark.skipif(os.name == "nt", reason="the handler pickles through a fd")
def test_anonymous_with_elements_unchanged(tmp_path):
    mt = MemoryMappedTensor.from_tensor(torch.arange(6.0).view(2, 3))
    assert mt._handler.size == 24
    assert len(mt._handler.buffer) == 24
    func, args = mt.__reduce__()
    assert func == MemoryMappedTensor._from_handler
    assert args == (mt._handler, torch.float32, torch.Size([2, 3]), None)
    loaded = pickle.loads(ForkingPickler.dumps(mt))
    assert loaded._handler.size == 24
    assert loaded.shape == (2, 3)
    loaded[0, 0] = 10.0
    assert mt[0, 0] == 10.0
    empty_view = pickle.loads(ForkingPickler.dumps(mt[1:][3:]))
    assert empty_view.shape == (0, 3)

    filename = tmp_path / "empty.memmap"
    mt = MemoryMappedTensor.zeros((3, 0), dtype=torch.int64, filename=filename)
    assert type(mt) is MemoryMappedTensor
    assert mt.shape == (3, 0)
    assert mt.dtype == torch.int64
    assert mt._handler is None
    assert mt.filename == str(filename)
    assert pickle.loads(pickle.dumps(mt)).shape == (3, 0)


# #3566: a memory map with no elements holds the file descriptors of one with
# elements, and releases them with the tensor
@pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"), reason="reads the open files in /proc"
)
@pytest.mark.parametrize("shape", [(4, 3), (4, 0)])
def test_anonymous_file_descriptors(shape):
    def count():
        # the open files of the handlers of this process
        gc.collect()
        targets = []
        for fd in os.listdir("/proc/self/fd"):
            try:
                targets.append(os.readlink(f"/proc/self/fd/{fd}"))
            except OSError:
                # the file descriptor of the listing itself
                continue
        return sum(f"pym-{os.getpid()}-" in target for target in targets)

    before = count()
    tensors = [MemoryMappedTensor.zeros(shape) for _ in range(8)]
    # the handler holds one, and its mmap a duplicate
    assert count() == before + 16
    del tensors
    assert count() == before


def test_pickle_filename(tmp_path):
    mt = MemoryMappedTensor.from_tensor(
        torch.arange(6.0).view(2, 3), filename=tmp_path / "tensor.memmap"
    )[1:]
    state = mt.__getstate__()
    assert set(state) == {"filename", "dtype", "shape", "index", "mode"}
    assert state["index"] == slice(1, None)
    assert state["shape"] == torch.Size([2, 3])
    loaded = pickle.loads(pickle.dumps(mt))
    assert loaded._index == slice(1, None)
    assert loaded._parent_shape == torch.Size([2, 3])
    assert (loaded == mt).all()


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

        memmap_tensor2 = MemoryMappedTensor._from_handler(
            memmap_tensor._handler, dtype=memmap_tensor.dtype, shape=self.shape
        )
        assert type(memmap_tensor2) is MemoryMappedTensor
        for t1, t2 in zip(memmap_tensor2, memmap_tensor):
            assert t1.dtype == t2.dtype
            assert (t1 == t2).all()

    @pytest.mark.parametrize("fn", ["empty", "zeros", "ones"])
    @pytest.mark.parametrize("with_filename", [False, True])
    def test_nested_default_dtype(self, fn, with_filename, tmpdir):
        filename = str(Path(tmpdir) / "file.memmap") if with_filename else None
        tensor = getattr(MemoryMappedTensor, fn)(self.shape, filename=filename)
        assert type(tensor) is MemoryMappedTensor
        assert tensor.is_nested
        assert tensor.dtype == torch.get_default_dtype()
        assert (tensor._nested_tensor_size() == self.shape).all()
        if fn != "empty":
            expected = 0 if fn == "zeros" else 1
            assert all((t == expected).all() for t in tensor.unbind(0))

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

    def test_pickle_view_of_view(self, tmpdir):
        tensors = [torch.arange(6).view(2, 3), torch.arange(8).view(2, 4)]
        mt = MemoryMappedTensor.from_tensor(
            torch.nested.nested_tensor(tensors), filename=Path(tmpdir) / "nt.memmap"
        )
        for view, expected in [
            (mt[1][1:], tensors[1][1:]),
            (mt.unbind(0)[0][:, 1], tensors[0][:, 1]),
            (mt[..., 0][1], tensors[1][:, 0]),
        ]:
            loaded = pickle.loads(pickle.dumps(view))
            assert loaded.shape == expected.shape
            assert (loaded == expected).all()
        # a view that is still nested
        loaded = pickle.loads(pickle.dumps(mt[..., 0][..., None]))
        for got, tensor in zip(loaded.unbind(0), tensors, strict=True):
            assert got.shape == tensor[:, :1].shape
            assert (got == tensor[:, :1]).all()

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

    @pytest.mark.parametrize("num_threads", [0, 2])
    def test_memmap_like_td_with_nested(self, tmpdir, num_threads):
        nt = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])
        td = TensorDict({"n": nt}, batch_size=[2])
        out = td.memmap_like(tmpdir, num_threads=num_threads)
        assert out["n"].is_nested
        assert (out["n"]._nested_tensor_size() == nt._nested_tensor_size()).all()
        assert all((t == 0).all() for t in out["n"].unbind(0))
        assert all((t == 1).all() for t in td["n"].unbind(0))
        loaded = TensorDict.load_memmap(tmpdir)
        assert loaded["n"].is_nested
        assert (loaded["n"]._nested_tensor_size() == nt._nested_tensor_size()).all()

    @pytest.mark.parametrize("existing", [False, True])
    def test_memmap_like_archive_with_nested_raises(self, tmpdir, existing):
        nt = torch.nested.nested_tensor([torch.ones(2), torch.ones(3)])
        td = TensorDict({"n": nt}, batch_size=[2])
        path = Path(tmpdir) / "data.tdz"
        if existing:
            path.write_bytes(b"old")
        with pytest.raises(NotImplementedError, match="archive mode"):
            td.memmap_like(path)
        if existing:
            assert path.read_bytes() == b"old"
        else:
            assert not path.exists()
        assert os.listdir(tmpdir) == (["data.tdz"] if existing else [])

    @pytest.mark.parametrize("num_threads", [0, 2])
    @pytest.mark.parametrize("with_prefix", [False, True])
    def test_memmap_td_jagged_raises(self, num_threads, with_prefix, tmpdir):
        # https://github.com/pytorch/tensordict/issues/1924
        lazy = TensorDict.from_list(
            [{"tokens": torch.arange(n)} for n in (3, 5, 2)], lazy=True
        )
        td = lazy.densify(layout=torch.jagged)
        prefix = tmpdir if with_prefix else None
        with pytest.raises(NotImplementedError, match="jagged.*'tokens'"):
            td.memmap(prefix, num_threads=num_threads)

    @pytest.mark.parametrize("num_threads", [0, 2])
    @pytest.mark.parametrize("with_prefix", [False, True])
    def test_memmap_td_strided_nested(self, num_threads, with_prefix, tmpdir):
        # https://github.com/pytorch/tensordict/issues/1924
        lazy = TensorDict.from_list(
            [{"tokens": torch.arange(n)} for n in (3, 5, 2)], lazy=True
        )
        td = lazy.densify(layout=torch.strided)
        prefix = tmpdir if with_prefix else None
        tdmemmap = td.memmap(prefix, num_threads=num_threads)
        if with_prefix:
            del tdmemmap
            gc.collect()
            tdmemmap = TensorDict.load_memmap(tmpdir)
        assert tdmemmap["tokens"].is_nested
        for t1, t2 in zip(td["tokens"].unbind(0), tdmemmap["tokens"].unbind(0)):
            assert (t1 == t2).all()

    @pytest.mark.parametrize("with_filename", [False, True])
    def test_from_tensor_jagged_raises(self, with_filename, tmpdir):
        # https://github.com/pytorch/tensordict/issues/1924
        tensor = torch.nested.nested_tensor(
            [torch.arange(n) for n in (3, 5, 2)], layout=torch.jagged
        )
        filename = str(Path(tmpdir) / "file.memmap") if with_filename else None
        with pytest.raises(NotImplementedError, match="jagged"):
            MemoryMappedTensor.from_tensor(tensor, filename=filename)

    @pytest.mark.parametrize("num_threads", [0, 2])
    @pytest.mark.parametrize("prefix", [None, "dir", "data.tdz"])
    def test_memmap_like_td_jagged_raises(self, num_threads, prefix, tmpdir):
        # https://github.com/pytorch/tensordict/issues/1924
        lazy = TensorDict.from_list(
            [{"tokens": torch.arange(n)} for n in (3, 5, 2)], lazy=True
        )
        td = lazy.densify(layout=torch.jagged)
        if prefix is not None:
            prefix = Path(tmpdir) / prefix
        with pytest.raises(NotImplementedError, match="jagged.*'tokens'"):
            td.memmap_like(prefix, num_threads=num_threads)
        assert os.listdir(tmpdir) == []

    @pytest.mark.parametrize("fn", ["empty", "zeros", "ones"])
    def test_complex_with_handler(self, fn):
        tensor = getattr(MemoryMappedTensor, fn)(self.shape, dtype=torch.complex64)
        assert type(tensor) is MemoryMappedTensor
        assert tensor.dtype == torch.complex64
        assert tensor._handler is not None
        assert (tensor._nested_tensor_size() == self.shape).all()


class TestReadWrite:
    @pytest.mark.skipif(_IS_ROOT, reason="root can write to read-only files")
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
        with pytest.raises(PermissionError, match="not writable"):
            MemoryMappedTensor.from_filename(
                filename=file_path, shape=[2, 3], dtype=torch.float64, mode="r+"
            )

    def test_mode(self, tmpdir):
        file_path = Path(tmpdir) / "elt.mmap"
        MemoryMappedTensor.from_tensor(torch.zeros(4), filename=file_path)

        def load(mode=None):
            return MemoryMappedTensor.from_filename(
                filename=file_path, shape=[4], dtype=torch.float32, mode=mode
            )

        # "r" maps the file copy-on-write: in-place writes stay in memory
        cow = load("r")
        cow.fill_(1)
        assert (load() == 0).all()
        # "r+" maps it shared: in-place writes reach the file
        load("r+").fill_(2)
        assert (load() == 2).all()
        assert (cow == 1).all()
        # the mode survives pickling, also for indexed views
        for tensor in (cow, cow[1:]):
            pickle.loads(pickle.dumps(tensor)).fill_(3)
        assert (load() == 2).all()
        with pytest.raises(ValueError, match="mode must be"):
            load("w")

    @pytest.mark.skipif(_IS_ROOT, reason="root can write to read-only files")
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


@tensorclass
class _AtomicTC:
    x: torch.Tensor


_posix_only = pytest.mark.skipif(
    sys.platform == "win32", reason="atomic=True is not supported on Windows"
)


class TestAtomicSave:
    @staticmethod
    def _td(value):
        return TensorDict(
            a=torch.full((3,), value),
            nested=TensorDict(b=torch.full((3, 2), value), batch_size=[3]),
            batch_size=[3],
        )

    @_posix_only
    @pytest.mark.parametrize("method", ["memmap", "memmap_", "save"])
    @pytest.mark.parametrize("num_threads", [0, 2])
    @pytest.mark.parametrize("existing", [False, True])
    def test_atomic(self, tmp_path, method, num_threads, existing):
        path = tmp_path / "ckpt"
        if existing:
            old = self._td(0.0)
            old["stale"] = torch.zeros(3)
            old.memmap(path)
        td = self._td(1.0)
        out = getattr(td, method)(path, num_threads=num_threads, atomic=True)
        assert (out is td) == (method == "memmap_")
        # the directory was replaced, and no temporary directory is left
        assert os.listdir(tmp_path) == ["ckpt"]
        assert not (path / "stale.memmap").exists()
        assert out.is_locked
        assert out.saved_path == path
        assert out["nested"].saved_path == path / "nested"
        assert out["a"].filename == str(path / "a.memmap")
        assert out["nested", "b"].filename == str(path / "nested" / "b.memmap")
        # writes reach the files at path
        out["nested", "b"].fill_(2.0)
        loaded = TensorDict.load_memmap(path)
        assert (loaded["a"] == 1).all()
        assert (loaded["nested", "b"] == 2).all()

    @_posix_only
    @pytest.mark.parametrize("kind", ["tensorclass", "lazy_stack"])
    @pytest.mark.parametrize("existing", [False, True])
    def test_atomic_containers(self, tmp_path, kind, existing):
        def make(value):
            if kind == "tensorclass":
                return _AtomicTC(x=torch.full((3,), value), batch_size=[3])
            return LazyStackedTensorDict(
                *[TensorDict(a=torch.full((3,), value)) for _ in range(2)],
                stack_dim=0,
            )

        path = tmp_path / "ckpt"
        if existing:
            make(0.0).memmap(path)
        out = make(1.0).memmap(path, atomic=True)
        assert os.listdir(tmp_path) == ["ckpt"]
        if kind == "tensorclass":
            assert out.x.filename == str(path / "_tensordict" / "x.memmap")
            out.x.fill_(2.0)
            assert (TensorDict.load_memmap(path).x == 2).all()
        else:
            for i, td in enumerate(out.tensordicts):
                assert td.saved_path == path / str(i)
                assert td["a"].filename == str(path / str(i) / "a.memmap")
                td["a"].fill_(2.0)
            assert (TensorDict.load_memmap(path)["a"] == 2).all()

    @_posix_only
    @pytest.mark.parametrize("method", ["memmap", "memmap_"])
    def test_atomic_non_tensor(self, tmp_path, method):
        path = tmp_path / "ckpt"
        td = TensorDict(
            a=torch.zeros(2),
            s=NonTensorData("old", batch_size=[2]),
            stack=NonTensorStack("x", "y"),
            batch_size=[2],
        )
        out = getattr(td, method)(path, atomic=True)
        # the non-tensor leaves write their files back to path
        out.update_({"s": "new"})
        out.get("stack")[0] = "z"
        assert os.listdir(tmp_path) == ["ckpt"]
        loaded = TensorDict.load_memmap(path)
        assert loaded.get("s").data == "new"
        assert loaded.get("stack").tolist() == ["z", "y"]

    @_posix_only
    def test_atomic_symlink_and_file(self, tmp_path):
        real = tmp_path / "real"
        self._td(0.0).memmap(real)
        link = tmp_path / "link"
        link.symlink_to(real)
        self._td(1.0).memmap(link, atomic=True)
        # the directory the link points to is replaced, and the link is kept
        assert link.is_symlink()
        assert sorted(os.listdir(tmp_path)) == ["link", "real"]
        assert (TensorDict.load_memmap(real)["a"] == 1).all()
        (tmp_path / "file").write_text("keep")
        with pytest.raises(NotADirectoryError):
            self._td(1.0).memmap(tmp_path / "file", atomic=True)
        assert (tmp_path / "file").read_text() == "keep"

    def test_atomic_windows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(td_serialization, "_IS_WINDOWS", True)
        for method in ("memmap", "memmap_", "save"):
            with pytest.raises(NotImplementedError, match="Windows"):
                getattr(self._td(1.0), method)(tmp_path / "ckpt", atomic=True)
        assert os.listdir(tmp_path) == []

    @_posix_only
    @pytest.mark.parametrize("method", ["memmap", "memmap_"])
    @pytest.mark.parametrize("existing", [False, True])
    def test_atomic_failure(self, tmp_path, monkeypatch, method, existing):
        path = tmp_path / "ckpt"
        if existing:
            self._td(0.0).memmap(path)
            before = sorted(p.relative_to(path) for p in path.rglob("*"))
        td = self._td(1.0)
        td["nested", "boom"] = torch.zeros(3)
        from_tensor = MemoryMappedTensor.from_tensor.__func__

        def failing_from_tensor(cls, input, *, filename=None, **kwargs):
            if filename is not None and Path(filename).name == "boom.memmap":
                raise OSError("simulated failure")
            return from_tensor(cls, input, filename=filename, **kwargs)

        monkeypatch.setattr(
            MemoryMappedTensor, "from_tensor", classmethod(failing_from_tensor)
        )
        with pytest.raises(OSError, match="simulated failure"):
            getattr(td, method)(path, atomic=True)
        if existing:
            # the old directory is untouched
            assert os.listdir(tmp_path) == ["ckpt"]
            assert sorted(p.relative_to(path) for p in path.rglob("*")) == before
            assert (TensorDict.load_memmap(path)["nested", "b"] == 0).all()
        else:
            assert os.listdir(tmp_path) == []

    @_posix_only
    def test_atomic_errors(self, tmp_path):
        td = self._td(1.0)
        with pytest.raises(ValueError, match="needs a prefix"):
            td.memmap(atomic=True)
        with pytest.raises(NotImplementedError, match="return_early"):
            td.memmap(tmp_path / "ckpt", num_threads=2, return_early=True, atomic=True)
        td.memmap(tmp_path / "ckpt")
        with pytest.raises(RuntimeError, match="existsok=False"):
            td.memmap(tmp_path / "ckpt", existsok=False, atomic=True)
        # a directory that holds no saved tensordict is not replaced
        (tmp_path / "other").mkdir()
        (tmp_path / "other" / "notes.txt").write_text("keep")
        with pytest.raises(RuntimeError, match="no meta.json"):
            td.memmap(tmp_path / "other", atomic=True)
        assert os.listdir(tmp_path / "other") == ["notes.txt"]
        assert sorted(os.listdir(tmp_path)) == ["ckpt", "other"]


class TestAtomicSaveClaims:
    # Behaviour that the description of #3141 promises and TestAtomicSave
    # does not cover.
    @staticmethod
    def _td(value):
        return TensorDict(
            a=torch.full((3,), value),
            nested=TensorDict(b=torch.full((3, 2), value), batch_size=[3]),
            batch_size=[3],
        )

    @pytest.mark.parametrize("method", ["memmap", "save"])
    @pytest.mark.parametrize("windows", [False, True])
    def test_atomic_archive_has_no_effect(self, tmp_path, monkeypatch, method, windows):
        # "For an archive, atomic is accepted and has no effect, since the
        # archive is already written this way." The Windows error is for
        # directories only.
        monkeypatch.setattr(td_serialization, "_IS_WINDOWS", windows)
        path = tmp_path / "data.tdz"
        self._td(0.0).save(path)
        getattr(self._td(1.0), method)(path, atomic=True)
        assert os.listdir(tmp_path) == ["data.tdz"]
        loaded = TensorDict.load_memmap(path)
        assert (loaded == self._td(1.0)).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
