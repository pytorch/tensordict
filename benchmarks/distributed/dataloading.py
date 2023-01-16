import argparse
import collections
import os
import time
from functools import wraps
from pathlib import Path

import torch

import torch.distributed.rpc as rpc
import tqdm

from tensordict import MemmapTensor
from tensordict.prototype import tensorclass
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.prototype import transforms

parser = argparse.ArgumentParser(
    description="RPC Replay Buffer Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--rank",
    type=int,
    help="Node Rank [0 = Replay Buffer, 1 = Dummy Trainer, 2+ = Dummy Data Collector]",
)
parser.add_argument(
    "--world_size",
    type=int,
    default=2,
)

torch.cuda.set_device(0)

RETRY_LIMIT = 2
NUM_WORKERS = 16
RETRY_DELAY_SECS = 3
FRACTION = 1
DATA_NODE = "Data"
TRAINER_NODE = "Trainer"
BATCH_SIZE = 128


def accept_remote_rref_invocation(func):
    """Object method decorator that allows a method to be invoked remotely by passing the `rpc.RRef` associated with the remote object construction as first argument in place of the object reference."""

    @wraps(func)
    def unpack_rref_and_invoke_function(self, *args, **kwargs):
        if isinstance(self, torch._C._distributed_rpc.PyRRef):
            self = self.local_value()
        return func(self, *args, **kwargs)

    return unpack_rref_and_invoke_function


def accept_remote_rref_udf_invocation(decorated_class):
    """Class decorator that applies `accept_remote_rref_invocation` to all public methods."""
    # ignores private methods
    for name in dir(decorated_class):
        method = getattr(decorated_class, name)
        if callable(method) and not name.startswith("_"):
            setattr(decorated_class, name, accept_remote_rref_invocation(method))
    return decorated_class


@tensorclass
class ImageNetData:
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset):
        data = cls(
            images=MemmapTensor(
                len(dataset),
                *dataset[0][0].squeeze().shape,
                dtype=torch.uint8,
            ),
            targets=MemmapTensor(len(dataset), dtype=torch.int64),
            batch_size=[len(dataset)],
        )
        data = data.memmap_()

        batch = 64
        dl = DataLoader(dataset, batch_size=batch, num_workers=NUM_WORKERS)
        i = 0
        pbar = tqdm.tqdm(total=len(dataset))
        for image, target in dl:
            _batch = image.shape[0]
            pbar.update(_batch)
            data[i : i + _batch] = cls(
                images=image, targets=target, batch_size=[_batch]
            )
            i += _batch

        return data


class InvAffine(nn.Module):
    """A custom normalization layer."""

    def __init__(self, loc, scale):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, x):
        return (x - self.loc) / self.scale


##############################################################################
# Next two transformations that can be used to randomly crop and flip the images.


class RandomHFlip(nn.Module):
    def forward(self, x: torch.Tensor):
        idx = (
            torch.zeros(*x.shape[:-3], 1, 1, 1, device=x.device, dtype=torch.bool)
            .bernoulli_()
            .expand_as(x)
        )
        return x.masked_fill(idx, 0.0) + x.masked_fill(~idx, 0.0).flip(-1)


class RandomCrop(nn.Module):
    def __init__(self, w, h):
        super(RandomCrop, self).__init__()
        self.w = w
        self.h = h

    def forward(self, x):
        batch = x.shape[:-3]
        index0 = torch.randint(x.shape[-2] - self.h, (*batch, 1), device=x.device)
        index0 = index0 + torch.arange(self.h, device=x.device)
        index0 = (
            index0.unsqueeze(1).unsqueeze(-1).expand(*batch, 3, self.h, x.shape[-1])
        )
        index1 = torch.randint(x.shape[-1] - self.w, (*batch, 1), device=x.device)
        index1 = index1 + torch.arange(self.w, device=x.device)
        index1 = index1.unsqueeze(1).unsqueeze(-2).expand(*batch, 3, self.h, self.w)
        return x.gather(-2, index0).gather(-1, index1)


class Collate(nn.Module):
    def __init__(self, transform=None, device=None):
        super().__init__()
        self.transform = transform
        self.device = device

    def __call__(self, x: ImageNetData):
        # move data to RAM
        out = x.apply(lambda _tensor: _tensor.contiguous())
        if self.device:
            # move data to gpu
            out = out.to(self.device)
        if self.transform:
            # apply transforms on gpu
            out.images = self.transform(out.images)
        return out


@accept_remote_rref_udf_invocation
class DummyTrainerNode:
    def __init__(self, world_size: int) -> None:
        print("DummyTrainerNode")
        self.id = rpc.get_worker_info().id
        self._prepare()
        self.datanodes = []
        self.world_size = world_size

    def _prepare(self):
        print("preparing data")
        data_dir = Path("/datasets01_ontap/imagenet_full_size/061417/")
        train_data_raw = datasets.ImageFolder(
            root=data_dir / "train",
            transform=transforms.Compose(
                [transforms.Resize((256, 256)), transforms.PILToTensor()]
            ),
        )
        train_data_raw.samples = train_data_raw.samples[
            : len(train_data_raw) // FRACTION
        ]

        # val_transform = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406],
        #                              [0.229, 0.224, 0.225]),
        #     ]
        # )

        # val_data = datasets.ImageFolder(root=data_dir / "val", transform=val_transform)
        # val_data.samples = val_data.samples[: len(val_data) // FRACTION]

        train_data_tc = ImageNetData.from_dataset(train_data_raw)
        # val_data_tc = ImageNetData.from_dataset(val_data)
        self.data = train_data_tc

    def train(self) -> None:
        print("train")
        len_data = self.data.shape[0]
        pbar = tqdm.tqdm(total=len_data)

        _prefetch_queue = collections.deque()
        _last = 0
        _next = BATCH_SIZE
        for i in range(self.world_size - 1):
            _prefetch_queue.append(
                rpc.rpc_async(
                    self.datanodes[i].owner(),
                    DataNode.sample,
                    args=(self.datanodes[i], range(_last, _next)),
                )
            )
            _last = _next
            _next = min(len_data - 1, _next + BATCH_SIZE)
            pbar.update(BATCH_SIZE)

        t0 = time.time()
        iteration = 0
        total = 0
        while total < len_data:
            batch = _prefetch_queue.popleft().wait()
            i = iteration % (self.world_size - 1)
            iteration += 1
            _prefetch_queue.append(
                rpc.rpc_async(
                    self.datanodes[i].owner(),
                    DataNode.sample,
                    args=(self.datanodes[i], range(_last, _next)),
                )
            )
            _last = _next
            _next = min(len_data - 1, _next + BATCH_SIZE)
            pbar.update(BATCH_SIZE)
            total += batch.shape[0]
        t = time.time() - t0
        print(f"time spent: {t:4.4f}s, Rate: {total/t} fps")

    def create_data(self, node) -> rpc.RRef:
        print(f"Creating DataNode object on remote node {node}")
        while True:
            try:
                data_info = rpc.get_worker_info(f"{DATA_NODE}_{node}")
                data_rref = rpc.remote(
                    data_info,
                    DataNode,
                    args=(
                        node,
                        BATCH_SIZE,
                    ),
                )
                print(f"Connected to data node {data_info}")
                time.sleep(5)
                self.datanodes.append(data_rref)
                return
            except Exception as e:
                print(f"Failed to connect to data node: {e}")
                time.sleep(RETRY_DELAY_SECS)

    def get_data(self):
        return self.data


@accept_remote_rref_udf_invocation
class DataNode:
    def __init__(self, rank, batch_size: int = BATCH_SIZE):
        print("Creating DataNode object")
        self.rank = rank
        self.id = rpc.get_worker_info().id
        train_ref = rpc.get_worker_info(f"{TRAINER_NODE}")
        self.train_ref = rpc.remote(
            train_ref,
            get_trainer,
        )
        self.batch_size = batch_size
        device = f"cuda:{rank}"
        self.collate_transform = nn.Sequential(
            InvAffine(
                loc=torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
                * 255,
                scale=torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
                * 255,
            ),
            RandomCrop(224, 224),
            RandomHFlip(),
        )
        self.collate = Collate(self.collate_transform, device=device)
        self._init()
        self.count = 0
        print("done!")

    def _init(self):
        self.data: ImageNetData = rpc.rpc_sync(
            self.train_ref.owner(),
            DummyTrainerNode.get_data,
            args=(self.train_ref,),
        )

    def sample(self, idx):
        self.count += 1
        if self.count % 100 == 0:
            print(self.count)
        out = self.collate(self.data[idx])
        return out


global trainer


def get_trainer():
    return trainer


if __name__ == "__main__":
    args = parser.parse_args()
    rank = args.rank
    world_size = args.world_size

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    str_init_method = "tcp://localhost:10003"

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=str_init_method,
        _transports=["uv"],
    )

    if rank == 0:
        for dest_rank in range(1, args.world_size):
            options.set_device_map(f"{DATA_NODE}_{dest_rank}", {i: i for i in range(8)})
        rpc.init_rpc(
            TRAINER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        torch.randn(1, device="cuda:0")
        trainer = DummyTrainerNode(args.world_size)
        for dest_rank in range(1, args.world_size):
            trainer.create_data(dest_rank)
        trainer.train()
        breakpoint()

    else:
        rpc.init_rpc(
            f"{DATA_NODE}_{rank}",
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        torch.randn(1, device="cuda:0")
        # device = f"cuda:{rank}"
        print(f"Initialised Data node {rank}")
        breakpoint()
    rpc.shutdown()
