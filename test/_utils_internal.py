# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import tempfile

import numpy as np
import torch

from tensordict import PersistentTensorDict, tensorclass, TensorDict
from tensordict.nn.params import TensorDictParams
from tensordict.tensordict import (
    _stack as stack_td,
    is_tensor_collection,
    LazyStackedTensorDict,
)
import multiprocessing as mp

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
    return devices


@tensorclass
class MyClass:
    X: torch.Tensor
    y: "MyClass"
    z: str


class TestTensorDictsBase:
    def td(self, device):
        return TensorDict(
            source={
                "a": torch.randn(4, 3, 2, 1, 5),
                "b": torch.randn(4, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 3, 2, 1, 3)),
            },
            batch_size=[4, 3, 2, 1],
            device=device,
        )

    def nested_td(self, device):
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

    def nested_tensorclass(self, device):

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

    def nested_stacked_td(self, device):
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
        return torch.stack(list(td.unbind(1)), 1)

    def stacked_td(self, device):
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

    def idx_td(self, device):
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

    def sub_td(self, device):
        td = TensorDict(
            source={
                "a": torch.randn(2, 4, 3, 2, 1, 5),
                "b": torch.randn(2, 4, 3, 2, 1, 10),
                "c": torch.randint(10, (2, 4, 3, 2, 1, 3)),
            },
            batch_size=[2, 4, 3, 2, 1],
            device=device,
        )
        return td.get_sub_tensordict(1)

    def sub_td2(self, device):
        td = TensorDict(
            source={
                "a": torch.randn(4, 2, 3, 2, 1, 5),
                "b": torch.randn(4, 2, 3, 2, 1, 10),
                "c": torch.randint(10, (4, 2, 3, 2, 1, 3)),
            },
            batch_size=[4, 2, 3, 2, 1],
            device=device,
        )
        return td.get_sub_tensordict((slice(None), 1))

    def memmap_td(self, device):
        return self.td(device).memmap_()

    def permute_td(self, device):
        return TensorDict(
            source={
                "a": torch.randn(3, 1, 4, 2, 5),
                "b": torch.randn(3, 1, 4, 2, 10),
                "c": torch.randint(10, (3, 1, 4, 2, 3)),
            },
            batch_size=[3, 1, 4, 2],
            device=device,
        ).permute(2, 0, 3, 1)

    def unsqueezed_td(self, device):
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

    def squeezed_td(self, device):
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

    def td_reset_bs(self, device):
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

    def td_h5(
        self,
        device,
    ):
        file = tempfile.NamedTemporaryFile()
        filename = file.name
        td_h5 = PersistentTensorDict.from_dict(
            self.nested_td(device), filename=filename, device=device
        )
        return td_h5

    def td_params(self, device):
        return TensorDictParams(self.td(device))


def expand_list(list_of_tensors, *dims):
    n = len(list_of_tensors)
    td = TensorDict({str(i): tensor for i, tensor in enumerate(list_of_tensors)}, [])
    td = td.expand(*dims).contiguous()
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
