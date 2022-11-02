# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import numpy as np
import torch.cuda
from tensordict import TensorDict


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


def expand_list(list_of_tensors, *dims):
    n = len(list_of_tensors)
    td = TensorDict({str(i): tensor for i, tensor in enumerate(list_of_tensors)}, [])
    td = td.expand(*dims).contiguous()
    return [td[str(i)] for i in range(n)]
