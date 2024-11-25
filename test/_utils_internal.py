# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import pathlib
import shutil
import tempfile

import numpy as np
import torch

from tensordict import NonTensorData, PersistentTensorDict, tensorclass, TensorDict
from tensordict._lazy import LazyStackedTensorDict
from tensordict._torch_func import _stack as stack_td
from tensordict.base import is_tensor_collection
from tensordict.nn.params import TensorDictParams
from tensordict.persistent import _has_h5 as _has_h5py
from tensordict.utils import set_lazy_legacy


def prod(sequence):
    if hasattr(math, "prod"):
        return math.prod(sequence)
    else:
        return int(np.prod(sequence))


def get_available_devices():
    devices = [torch.device("cpu")]
    n_cuda = torch.cuda.device_count()
    if n_cuda > 0:
        for i in range(n_cuda):
            devices += [torch.device(f"cuda:{i}")]
            if i == 1:
                break
    # if torch.backends.mps.is_available():
    #     for i in range(torch.mps.device_count()):
    #         devices += [torch.device(f"mps:{i}")]
    #         if i == 1:
    #             break
    return devices


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
    @set_lazy_legacy(True)
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
        return torch.stack([_td.clone() for _td in td.unbind(1)], 1)

    for device in get_available_devices():
        TYPES_DEVICES += [["nested_stacked_td", device]]
        TYPES_DEVICES_NOLAZY += [["nested_stacked_td", device]]

    @classmethod
    @set_lazy_legacy(True)
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
        return stack_td([td1, td2], 2)

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
    @set_lazy_legacy(True)
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
    @set_lazy_legacy(True)
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
    @set_lazy_legacy(True)
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
        file = tempfile.NamedTemporaryFile()
        filename = file.name
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
