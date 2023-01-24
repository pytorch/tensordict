import numpy as np
import torch

from tensordict import MemmapTensor


class DynamicShapeTensor:
    def __init__(
        self,
        shape: torch.Size,  # -1 indicates variable size
        total_elts: int,  # size of the storage
        total_index: int,  # number of indexable items along stack dim
        stack_dim: int = 0,
    ):
        if stack_dim != 0:
            raise NotImplementedError(stack_dim)
        self.offsets = MemmapTensor(total_index, dtype=torch.int64).as_tensor()
        self.total_elts = total_elts
        assert sum(s == -1 for s in shape) == 1
        assert sum(s > 0 for s in shape) == len(shape) - 1
        self.shape = shape
        self.shapes = MemmapTensor(total_index, dtype=torch.int64).as_tensor()
        self.storage = MemmapTensor(total_elts).as_tensor()
        self.variable_shape = sum(np.cumprod([s > 0 for s in shape]))
        self.total_index = total_index

    def __setitem__(self, idx: int, value):
        assert isinstance(idx, int)
        assert value.ndim == len(self.shape)
        assert (s1 == s2 for (s1, s2) in zip(value.shape, self.shape) if s2 > 0)
        prev_offset = self.offsets[idx]
        offset = value.numel()
        if idx < 0:
            idx = self.total_index + idx
        if idx > 0:
            start = self.offsets[idx - 1].item()
            stop = self.offsets[idx] = start + offset
        else:
            start = 0
            stop = self.offsets[idx] = offset
        if idx < self.total_index - 1 and self.offsets[idx + 1].item() > 0:
            self.offsets[idx:] += offset - prev_offset
            if self.offsets[-1] > self.total_elts:
                raise RuntimeError
        print(start, stop)
        self.storage[start:stop] = value.view(-1)
        shape = value.shape
        out_shape = list(self.shape)
        out_shape[self.variable_shape] = shape[self.variable_shape]
        print(shape, out_shape)
        self.shapes[idx] = shape[self.variable_shape]

    def __getitem__(self, item):
        if item < 0:
            item = self.total_index + item
        start = self.offsets[item - 1].item() if item > 0 else 0
        stop = self.offsets[item].item()
        if stop == 0:
            raise RuntimeError
        shape = list(self.shape)
        shape[self.variable_shape] = self.shapes[item].item()
        data = self.storage[start:stop].view(shape)
        return data


if __name__ == "__main__":
    N = 10
    data = []
    d = DynamicShapeTensor([3, 4, -1], 3 * 4 * 12 * N, N, 0)
    for i in range(N):
        data.append(torch.randn(3, 4, torch.randint(1, 12, (1,)).item()))
        d[i] = data[-1]
    for i in range(N):
        assert (d[i] == data[i]).all()
