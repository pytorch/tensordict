# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import atexit
import json
import os
import pathlib
import shutil
import tempfile
import types
import warnings
from functools import wraps
from unittest import mock

import torch
from filelock import FileLock
from tensordict import (
    MetaData,
    NonTensorData,
    PersistentTensorDict,
    tensorclass,
    TensorDict,
    TypedTensorDict,
    UnbatchedTensor,
)
from tensordict._lazy import LazyStackedTensorDict
from tensordict.base import is_tensor_collection
from tensordict.nn.params import TensorDictParams
from tensordict.persistent import _has_h5 as _has_h5py
from tensordict.utils import _check_recursive_properties, set_lazy_legacy


def legacy_lazy_mode():
    """Return ``set_lazy_legacy(True)``, without the warning that it is deprecated.

    The legacy lazy mode stays tested until it is removed.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The legacy lazy mode", DeprecationWarning)
        return set_lazy_legacy(True)


def streaming_dataset(**kwargs):
    """Builds a ``streaming.StreamingDataset``, one process at a time.

    A ``StreamingDataset`` claims the first free shared-memory prefix
    (``/000000_locals``, ...) by looking for it and then creating it. Two
    pytest-xdist workers that build one at the same moment pick the same
    prefix, and the second fails with ``FileExistsError``. A lock file in the
    temporary directory makes the workers build theirs one after the other.

    The shared memory a ``StreamingDataset`` creates or attaches to is closed
    only by the ``atexit`` handlers that ``streaming`` registers, and building
    one attaches to the shared memory of every live prefix. Each dataset of a
    test session would leak two file descriptors per block and per earlier
    prefix, enough to exhaust the default limit of 1024. The handlers of the
    previous dataset run here instead, so a worker keeps one dataset's blocks.
    """
    import streaming.base.shared.memory as streaming_memory
    from streaming import StreamingDataset

    with FileLock(
        pathlib.Path(tempfile.gettempdir()) / "tensordict_test_streaming.lock"
    ):
        _release_streaming_shm()
        with mock.patch.object(
            streaming_memory,
            "atexit",
            types.SimpleNamespace(register=_STREAMING_CLEANUPS.append),
        ):
            return StreamingDataset(**kwargs)


_STREAMING_CLEANUPS = []


@atexit.register
def _release_streaming_shm():
    while _STREAMING_CLEANUPS:
        _STREAMING_CLEANUPS.pop()()


def is_npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()


def get_available_devices():
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
            if i == 1:
                break
    if is_npu_available():
        warnings.warn(
            "torch_npu is an experimental feature and not currently included in tensordict CI/CD."
        )
        n_npu = torch.npu.device_count()
        if n_npu > 0:
            for i in range(n_npu):
                devices += [torch.device(f"npu:{i}")]
                if i == 1:
                    break
    # TODO: MPS would be worth considering but it's a lot of work
    # for example, many ops are tested with various dtypes but MPS struggles with
    # float64. Shared mem can also cause trouble.
    # if torch.backends.mps.is_available():
    #     for i in range(torch.mps.device_count()):
    #         devices += [torch.device(f"mps:{i}")]
    #         if i == 1:
    #             break
    return devices


_UPDATE_KNOWN_FAILURES = bool(os.environ.get("TENSORDICT_UPDATE_KNOWN_FAILURES"))


def _describe(err):
    return f"raises {type(err).__name__}: {str(err).splitlines()[0][:80] if str(err) else ''}"


def check_known_failures(key, failures, path, known_failures, hint=None):
    """Compare ``failures`` ({case: reason}) with the known failures of ``key`` in ``path``.

    ``known_failures`` is the content of ``path`` when the tests started. With
    ``TENSORDICT_UPDATE_KNOWN_FAILURES`` set, the list of ``key`` in ``path`` is
    rewritten instead. ``hint`` ends the message of a failure.
    """
    if _UPDATE_KNOWN_FAILURES:
        known = json.loads(path.read_text())
        known[key] = sorted(failures)
        path.write_text(json.dumps(dict(sorted(known.items())), indent=1) + "\n")
        return
    known = set(known_failures.get(key, ()))
    new = {case: reason for case, reason in failures.items() if case not in known}
    fixed = sorted(known.difference(failures))
    message = []
    if new:
        message.append(
            f"{len(new)} cases fail and are not known failures:\n"
            + "\n".join(f"  {case}: {reason}" for case, reason in sorted(new.items()))
        )
    if fixed:
        message.append(
            f"{len(fixed)} known failures pass now; remove them from "
            f"{path.name} under {key!r}:\n" + "\n".join(f"  {case}" for case in fixed)
        )
    if message and hint:
        message.append(hint)
    assert not message, "\n".join(message)


@tensorclass
class MyClass:
    X: torch.Tensor
    y: "MyClass"
    z: str


class TestTensorDictsBase:
    TYPES_DEVICES = []
    TYPES_DEVICES_NOLAZY = []

    @classmethod
    def td(cls, device):
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    for device in get_available_devices():
        TYPES_DEVICES += [["td", device]]
        TYPES_DEVICES_NOLAZY += [["td", device]]

    @classmethod
    def nested_td(cls, device):
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
                "my_nested_td": TensorDict(
                    {"inner": torch.randn(4, 3, 2, 1, 2)}, [4, 3, 2, 1]
                ),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    for device in get_available_devices():
        TYPES_DEVICES += [["nested_td", device]]
        TYPES_DEVICES_NOLAZY += [["nested_td", device]]

    @classmethod
    def nested_tensorclass(cls, device):

        nested_class = MyClass(
            X=torch.randn(4, 3, 2, 1),
            y=MyClass(
                X=torch.randn(
                    4,
                    3,
                    2,
                    1,
                ),
                y=None,
                z=None,
                batch_size=[4, 3, 2, 1],
            ),
            z="z",
            batch_size=[4, 3, 2, 1],
        )
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
                "my_nested_tc": nested_class,
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    for device in get_available_devices():
        TYPES_DEVICES += [["nested_tensorclass", device]]
        TYPES_DEVICES_NOLAZY += [["nested_tensorclass", device]]

    @classmethod
    def nested_stacked_td(cls, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
                "my_nested_td": TensorDict(
                    {"inner": torch.randn(4, 3, 2, 1, 2)}, [4, 3, 2, 1]
                ),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )
        # we need to clone to avoid passing a views other tensors
        return LazyStackedTensorDict(
            *[_td.clone() for _td in td.unbind(1)], stack_dim=1
        )

    for device in get_available_devices():
        TYPES_DEVICES += [["nested_stacked_td", device]]
        TYPES_DEVICES_NOLAZY += [["nested_stacked_td", device]]

    @classmethod
    def stacked_td(cls, device):
        td1 = TensorDict(
            source={
                "a": torch.randn(4, 3, 1, 5),
                "b": torch.randn(4, 3, 1, 10),
                "c": torch.randint(10, (4, 3, 1, 3)),
            },
            batch_size=[4, 3, 1],
            device=device,
        )
        td2 = TensorDict(
            source={
                "a": torch.randn(4, 3, 1, 5),
                "b": torch.randn(4, 3, 1, 10),
                "c": torch.randint(10, (4, 3, 1, 3)),
            },
            batch_size=[4, 3, 1],
            device=device,
        )
        return LazyStackedTensorDict(td1, td2, stack_dim=2)

    for device in get_available_devices():
        TYPES_DEVICES += [["stacked_td", device]]

    @classmethod
    def idx_td(cls, device):
        td = TensorDict(
            source={
                "a": torch.randn(2, 4, 3, 2, 1, 5),
                "b": torch.randn(2, 4, 3, 2, 1, 10),
                "c": torch.randint(10, (2, 4, 3, 2, 1, 3)),
            },
            batch_size=[2, 4, 3, 2, 1],
            device=device,
        )
        return td[1]

    for device in get_available_devices():
        TYPES_DEVICES += [["idx_td", device]]

    @classmethod
    def sub_td(cls, device):
        td = TensorDict(
            source={
                "a": torch.randn(2, 4, 3, 2, 1, 5),
                "b": torch.randn(2, 4, 3, 2, 1, 10),
                "c": torch.randint(10, (2, 4, 3, 2, 1, 3)),
            },
            batch_size=[2, 4, 3, 2, 1],
            device=device,
        )
        return td._get_sub_tensordict(1)

    for device in get_available_devices():
        TYPES_DEVICES += [["sub_td", device]]

    @classmethod
    def sub_td2(cls, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 2, 3, 2, 1, 5),
                "b": torch.randn(4, 2, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 2, 3, 2, 1, 3)),
            },
            batch_size=[4, 2, 3, 2, 1],
            device=device,
        )
        return td._get_sub_tensordict((slice(None), 1))

    for device in get_available_devices():
        TYPES_DEVICES += [["sub_td2", device]]

    temp_path_memmap = tempfile.TemporaryDirectory()

    @classmethod
    def memmap_td(cls, device):
        path = pathlib.Path(cls.temp_path_memmap.name)
        shutil.rmtree(path)
        path.mkdir()
        return cls.td(device).memmap_(path)

    TYPES_DEVICES += [["memmap_td", torch.device("cpu")]]
    TYPES_DEVICES_NOLAZY += [["memmap_td", torch.device("cpu")]]

    @classmethod
    @legacy_lazy_mode()
    def permute_td(cls, device):
        return TensorDict(
            source={
                "a": torch.randn(3, 1, 4, 2, 5),
                "b": torch.randn(3, 1, 4, 2, 10),
                "c": torch.randint(10, (3, 1, 4, 2, 3)),
            },
            batch_size=[3, 1, 4, 2],
            device=device,
        ).permute(2, 0, 3, 1)

    for device in get_available_devices():
        TYPES_DEVICES += [["permute_td", device]]

    @classmethod
    @legacy_lazy_mode()
    def unsqueezed_td(cls, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 5),
                "b": torch.randn(4, 3, 2, 10),
                "c": torch.randint(10, (4, 3, 2, 3)),
            },
            batch_size=[4, 3, 2],
            device=device,
        )
        return td.unsqueeze(-1)

    for device in get_available_devices():
        TYPES_DEVICES += [["unsqueezed_td", device]]

    @classmethod
    @legacy_lazy_mode()
    def squeezed_td(cls, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 3, 1, 2, 1, 5),
                "b": torch.randn(4, 3, 1, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 1, 2, 1, 3)),
            },
            batch_size=[4, 3, 1, 2, 1],
            device=device,
        )
        return td.squeeze(2)

    for device in get_available_devices():
        TYPES_DEVICES += [["squeezed_td", device]]

    @classmethod
    def td_reset_bs(cls, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
            },
            batch_size=[4, 3, 2],
            device=device,
        )
        td.batch_size = torch.Size([4, 3, 2, 1])
        return td

    for device in get_available_devices():
        TYPES_DEVICES += [["td_reset_bs", device]]
        TYPES_DEVICES_NOLAZY += [["td_reset_bs", device]]

    @classmethod
    def td_h5(
        cls,
        device,
    ):
        file = tempfile.NamedTemporaryFile(delete=False)
        filename = file.name
        file.close()  # Close file handle before h5py opens it
        nested_td = cls.nested_td(device)
        td_h5 = PersistentTensorDict.from_dict(
            nested_td, filename=filename, device=device
        )
        assert td_h5.batch_size == nested_td.batch_size
        return td_h5

    if _has_h5py:
        for device in get_available_devices():
            TYPES_DEVICES += [["td_h5", device]]
            TYPES_DEVICES_NOLAZY += [["td_h5", device]]

    @classmethod
    def td_params(cls, device):
        return TensorDictParams(cls.td(device))

    for device in get_available_devices():
        TYPES_DEVICES += [["td_params", device]]
        TYPES_DEVICES_NOLAZY += [["td_params", device]]

    @classmethod
    def td_with_non_tensor(cls, device):
        td = cls.td(device)
        return td.set_non_tensor(
            ("data", "non_tensor"),
            # this is allowed since nested NonTensorData are automatically unwrapped
            NonTensorData(
                "some text data",
                batch_size=td.batch_size,
                device=td.device,
                names=td.names if td._has_names() else None,
            ),
        )

    for device in get_available_devices():
        TYPES_DEVICES += [["td_with_non_tensor", device]]
        TYPES_DEVICES_NOLAZY += [["td_with_non_tensor", device]]

    @classmethod
    def td_with_non_tensor_and_metadata(cls, device):
        td = cls.td(device)
        td.set(("data", "non_tensor"), NonTensorData("a string!"))
        td.set(("data", "metadata"), MetaData[str]("a metadata!"))
        return td

    for device in get_available_devices():
        TYPES_DEVICES += [["td_with_non_tensor_and_metadata", device]]
        TYPES_DEVICES_NOLAZY += [["td_with_non_tensor_and_metadata", device]]

    @classmethod
    def td_with_unbatched(cls, device):
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
                "unbatched": UnbatchedTensor(torch.randn(7, 11)),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    for device in get_available_devices():
        TYPES_DEVICES += [["td_with_unbatched", device]]
        TYPES_DEVICES_NOLAZY += [["td_with_unbatched", device]]

    @classmethod
    def typed_td(cls, device):
        return _TypedTD(
            a=torch.randn(4, 3, 2, 1, 5),
            b=torch.randn(4, 3, 2, 1, 10),
            c=torch.randint(10, (4, 3, 2, 1, 3)),
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    for device in get_available_devices():
        TYPES_DEVICES += [["typed_td", device]]
        TYPES_DEVICES_NOLAZY += [["typed_td", device]]

    @classmethod
    def nested_typed_td(cls, device):
        return _NestedTypedTD(
            a=torch.randn(4, 3, 2, 1, 5),
            b=torch.randn(4, 3, 2, 1, 10),
            c=torch.randint(10, (4, 3, 2, 1, 3)),
            my_nested_td=TensorDict(
                {"inner": torch.randn(4, 3, 2, 1, 2)}, [4, 3, 2, 1]
            ),
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    for device in get_available_devices():
        TYPES_DEVICES += [["nested_typed_td", device]]
        TYPES_DEVICES_NOLAZY += [["nested_typed_td", device]]

    # Not in TYPES_DEVICES: test_conformance.py runs it.
    @classmethod
    def typed_lazy(cls, device):
        return _TypedTD.from_tensordict(cls.stacked_td(device))


class _TypedTD(TypedTensorDict):
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor


class _NestedTypedTD(TypedTensorDict):
    a: torch.Tensor
    b: torch.Tensor
    c: torch.Tensor
    my_nested_td: TensorDict


def _wrap_checked_tensordict_factory(name):
    factory = getattr(TestTensorDictsBase, name)

    @classmethod
    @wraps(factory.__func__)
    def checked_factory(cls, device, _factory=factory):
        td = _factory.__func__(cls, device)
        _check_recursive_properties(td)
        return td

    return checked_factory


for _td_name in dict.fromkeys(
    td_name for td_name, _ in TestTensorDictsBase.TYPES_DEVICES
):
    setattr(
        TestTensorDictsBase,
        _td_name,
        _wrap_checked_tensordict_factory(_td_name),
    )


def expand_list(list_of_tensors, *dims):
    n = len(list_of_tensors)
    td = TensorDict({str(i): tensor for i, tensor in enumerate(list_of_tensors)}, [])
    td = td.expand(dims).contiguous()
    return [td[str(i)] for i in range(n)]


def decompose(td):
    if isinstance(td, LazyStackedTensorDict):
        for inner_td in td.tensordicts:
            yield from decompose(inner_td)
    else:
        for v in td.values():
            if is_tensor_collection(v):
                yield from decompose(v)
            else:
                yield v


class DummyPicklableClass:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(other, DummyPicklableClass):
            return self.value == other.value
        return self.value == other
