# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distributed data-loading benchmark with tensorclass.

Requires ImageNet, multiple GPUs, and several optional packages
(tenacity, torchvision, wandb).  Skipped automatically when any
requirement is missing.

Manual run::

    python -m pytest benchmarks/distributed/test_dataloading.py -v -s \
        --world_size=5
"""

import collections
import importlib
import logging
import os
import time
from functools import wraps
from pathlib import Path

import pytest
import torch

from tensordict import MemoryMappedTensor, tensorclass
from torch import multiprocessing as mp, nn

_has_tenacity = importlib.util.find_spec("tenacity") is not None
_has_torchvision = importlib.util.find_spec("torchvision") is not None
_has_rpc = hasattr(torch.distributed, "rpc")
_has_cuda = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not (_has_tenacity and _has_torchvision and _has_rpc and _has_cuda),
    reason=(
        "Requires tenacity, torchvision, torch.distributed.rpc and CUDA. "
        "Also needs ImageNet on disk."
    ),
)

RETRY_LIMIT = 2
NUM_WORKERS = 8
RETRY_DELAY_SECS = 3
DATA_NODE = "Data"
TRAINER_NODE = "Trainer"
BATCH_SIZE = 128
NUM_COLLECTION = 5


# ---------------------------------------------------------------------------
# tensorclass
# ---------------------------------------------------------------------------


@tensorclass
class ImageNetData:
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset):
        import tqdm
        from torch.utils.data import DataLoader

        data = cls(
            images=MemoryMappedTensor.empty(
                (len(dataset), *dataset[0][0].squeeze().shape),
                dtype=torch.uint8,
            ),
            targets=MemoryMappedTensor.empty(len(dataset), dtype=torch.int64),
            batch_size=[len(dataset)],
        )
        data.memmap_()
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

    @classmethod
    def load(cls, dataset, path):
        import torchsnapshot

        data = cls(
            images=MemoryMappedTensor.empty(
                (len(dataset), *dataset[0][0].squeeze().shape),
                dtype=torch.uint8,
            ),
            targets=MemoryMappedTensor(len(dataset), dtype=torch.int64),
            batch_size=[len(dataset)],
        )
        data.memmap_()
        t0 = time.time()
        logging.info("loading...")
        snapshot = torchsnapshot.Snapshot(path=path)
        sd = dict(data.state_dict())
        app_state = {"state": torchsnapshot.StateDict(data=sd)}
        snapshot.restore(app_state=app_state)
        logging.info(f"done! Took: {time.time() - t0:4.4f}s")
        return data

    def save(self, path):
        import torchsnapshot

        sd = dict(self.state_dict())
        app_state = {"state": torchsnapshot.StateDict(data=sd)}
        torchsnapshot.Snapshot.take(app_state=app_state, path=path)


# ---------------------------------------------------------------------------
# RPC helpers
# ---------------------------------------------------------------------------


def accept_remote_rref_invocation(func):
    @wraps(func)
    def unpack_rref_and_invoke_function(self, *args, **kwargs):
        if isinstance(self, torch._C._distributed_rpc.PyRRef):
            self = self.local_value()
        return func(self, *args, **kwargs)

    return unpack_rref_and_invoke_function


def accept_remote_rref_udf_invocation(decorated_class):
    for name in dir(decorated_class):
        method = getattr(decorated_class, name)
        if callable(method) and not name.startswith("_"):
            setattr(decorated_class, name, accept_remote_rref_invocation(method))
    return decorated_class


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


class InvAffine(nn.Module):
    def __init__(self, loc, scale):
        super().__init__()
        self.loc = loc
        self.scale = scale

    def forward(self, x):
        return (x - self.loc) / self.scale


class RandomHFlip(nn.Module):
    def forward(self, x: torch.Tensor):
        idx = (
            torch.zeros([*x.shape[:-3], 1, 1, 1], device=x.device, dtype=torch.bool)
            .bernoulli_()
            .expand_as(x)
        )
        return x.masked_fill(idx, 0.0) + x.masked_fill(~idx, 0.0).flip(-1)


class RandomCrop(nn.Module):
    def __init__(self, w, h):
        super().__init__()
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

    @torch.inference_mode()
    def __call__(self, x: ImageNetData):
        out = x.apply(lambda _tensor: _tensor.as_tensor()).pin_memory().to(self.device)
        if self.transform:
            out.images = self.transform(out.images)
        return out


# ---------------------------------------------------------------------------
# Trainer / Data nodes
# ---------------------------------------------------------------------------


@accept_remote_rref_udf_invocation
class DummyTrainerNode:
    def __init__(self, world_size, single_gpu, local_transform=True):
        from torch.distributed import rpc

        self.id = rpc.get_worker_info().id
        self.datanodes = []
        self.world_size = world_size
        self.single_gpu = single_gpu
        self.local_transform = local_transform
        if not local_transform:
            loc = (
                torch.tensor([0.485, 0.456, 0.406], device="cuda:0").view(3, 1, 1) * 255
            )
            scale = (
                torch.tensor([0.229, 0.224, 0.225], device="cuda:0").view(3, 1, 1) * 255
            )
            self.collate_transform = nn.Sequential(
                InvAffine(loc=loc, scale=scale),
                RandomCrop(224, 224),
                RandomHFlip(),
            )

    def init(self, train_data_tc):
        from torch.distributed import rpc

        self.data = train_data_tc
        for i in range(self.world_size - 1):
            rpc.rpc_sync(
                self.datanodes[i].owner(),
                DataNode.set_data,
                args=(self.datanodes[i], self.data),
            )

    def train(self):
        import tqdm
        from torch.distributed import rpc

        logging.info("train")
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

        iteration = 0
        while len(_prefetch_queue):
            batch = _prefetch_queue.popleft().wait()
            if not self.local_transform:
                batch.images = self.collate_transform(batch.images)
            i = iteration % (self.world_size - 1)
            iteration += 1
            if iteration == self.world_size:
                t0 = time.time()
                total = 0
            if _next != _last:
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
            if iteration >= self.world_size:
                total += batch.shape[0]
        t = time.time() - t0
        logging.info(f"time spent: {t:4.4f}s, Rate: {total / t} fps")
        return {"time": t, "rate": total / t}

    def create_data_nodes(self, data):
        for dest_rank in range(1, self.world_size):
            self.create_data_node(dest_rank, local_transform=self.local_transform)
        self.init(data)

    def create_data_node(self, node, local_transform):
        import tenacity
        from torch.distributed import rpc

        @tenacity.retry(
            stop=tenacity.stop_after_attempt(RETRY_LIMIT),
            wait=tenacity.wait_fixed(RETRY_DELAY_SECS),
            reraise=True,
        )
        def _create(node, local_transform):
            logging.info(f"Creating DataNode object on remote node {node}")
            data_info = rpc.get_worker_info(f"{DATA_NODE}_{node}")
            data_rref = rpc.remote(
                data_info,
                DataNode,
                args=(node, BATCH_SIZE, self.single_gpu, local_transform),
            )
            logging.info(f"Connected to data node {data_info}")
            time.sleep(5)
            self.datanodes.append(data_rref)

        _create(node, local_transform)

    def get_data(self):
        return self.data


@accept_remote_rref_udf_invocation
class DataNode:
    def __init__(
        self, rank, batch_size=BATCH_SIZE, single_gpu=False, make_transform=True
    ):
        logging.info("Creating DataNode object")
        self.rank = rank
        from torch.distributed import rpc

        self.id = rpc.get_worker_info().id
        self.single_gpu = single_gpu
        self.batch_size = batch_size
        device = "cuda:1" if self.single_gpu else f"cuda:{rank}"
        self.make_transform = make_transform
        if self.make_transform:
            loc = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1) * 255
            scale = (
                torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1) * 255
            )
            self.collate_transform = nn.Sequential(
                InvAffine(loc=loc, scale=scale),
                RandomCrop(224, 224),
                RandomHFlip(),
            )
        else:
            self.collate_transform = None
        self.collate = Collate(self.collate_transform, device=device)
        self.initialized = False
        self.count = 0
        logging.info("done!")

    def set_data(self, data):
        logging.info("initializing")
        self.initialized = True
        self.data: ImageNetData = data

    def sample(self, idx):
        self.count += 1
        return self.collate(self.data[idx])


# ---------------------------------------------------------------------------
# RPC bootstrap
# ---------------------------------------------------------------------------


def init_rpc(rank, name, world_size, single_gpu):
    from torch.distributed import rpc

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
        for dest_rank in range(1, world_size):
            options.set_device_map(
                f"{DATA_NODE}_{dest_rank}",
                {0: dest_rank if not single_gpu else 1},
            )
    rpc.init_rpc(
        name,
        rank=rank,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
    )
    logging.info(f"Initialised {name}")


def _worker(rank, world_size, train_data_tc, single_gpu, trainer_transform):
    torch.randn(1, device="cuda:0")
    if rank == 0:
        trainer = DummyTrainerNode(
            world_size,
            single_gpu,
            local_transform=not trainer_transform,
        )
        trainer.create_data_nodes(train_data_tc)
        min_time = float("inf")
        rate = 0.0
        for _ in range(NUM_COLLECTION):
            stats = trainer.train()
            if stats["time"] < min_time:
                min_time = stats["time"]
                rate = stats["rate"]
        logging.info(f"FINAL: time spent: {min_time:4.4f}s, Rate: {rate} fps")


# ---------------------------------------------------------------------------
# pytest entry point
# ---------------------------------------------------------------------------

_IMAGENET_DIR = Path("/datasets01_ontap/imagenet_full_size/061417/")


def pytest_addoption(parser):
    parser.addoption("--world_size", type=int, default=2)
    parser.addoption("--single_gpu", action="store_true", default=False)
    parser.addoption("--trainer_transform", action="store_true", default=False)
    parser.addoption("--fraction", type=int, default=1)


@pytest.mark.skipif(
    not _IMAGENET_DIR.exists(),
    reason=f"ImageNet not found at {_IMAGENET_DIR}",
)
@pytest.mark.skipif(
    not _has_cuda or torch.cuda.device_count() < 2,
    reason="Requires at least 2 CUDA devices",
)
def test_distributed_dataloading(pytestconfig):
    from torchvision import datasets
    from torchvision.prototype import transforms

    world_size = pytestconfig.getoption("--world_size", default=2)
    single_gpu = pytestconfig.getoption("--single_gpu", default=False)
    trainer_transform = pytestconfig.getoption("--trainer_transform", default=False)
    fraction = pytestconfig.getoption("--fraction", default=1)

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    train_data_raw = datasets.ImageFolder(
        root=_IMAGENET_DIR / "train",
        transform=transforms.Compose(
            [transforms.Resize((256, 256)), transforms.PILToTensor()]
        ),
    )
    train_data_raw.samples = train_data_raw.samples[: len(train_data_raw) // fraction]
    train_data_tc = ImageNetData.from_dataset(train_data_raw)
    names = [TRAINER_NODE, *[f"{DATA_NODE}_{r}" for r in range(1, world_size)]]

    with mp.Pool(world_size) as pool:
        pool.starmap(
            init_rpc,
            ((rank, name, world_size, single_gpu) for rank, name in enumerate(names)),
        )
        pool.starmap(
            _worker,
            (
                (rank, world_size, train_data_tc, single_gpu, trainer_transform)
                for rank, _ in enumerate(names)
            ),
        )
        from torch.distributed import rpc

        pool.apply_async(rpc.shutdown)
