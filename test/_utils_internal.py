# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import numpy as np
import torch.cuda
from tensordict import SavedTensorDict, TensorDict
from tensordict.tensordict import _stack as stack_td


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

    def saved_td(self, device):
        return SavedTensorDict(source=self.td(device))

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


def expand_list(list_of_tensors, *dims):
    n = len(list_of_tensors)
    td = TensorDict({str(i): tensor for i, tensor in enumerate(list_of_tensors)}, [])
    td = td.expand(*dims).contiguous()
    return [td[str(i)] for i in range(n)]
