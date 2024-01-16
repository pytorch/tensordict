# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed data-loading with tensorclass
=========================================
"""

##############################################################################
# This file provides an example of distributed dataloading with tensorclass.
# It can be run by simply calling `python dataloading.py --world_size=5` (optionally
# adding the wandb entity and key if logging is required).
#
# The longest part of this script is by far the data construction.

import argparse
import collections

import logging
import os
import time
from functools import wraps
from pathlib import Path

import tenacity
import torch
import tqdm

from tensordict import MemoryMappedTensor
from tensordict.prototype import tensorclass
from torch import multiprocessing as mp, nn
from torch.distributed import rpc
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.prototype import transforms

parser = argparse.ArgumentParser(
    description="RPC Replay Buffer Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--world_size",
    type=int,
    default=2,
)
parser.add_argument(
    "--single_gpu",
    action="store_true",
    help="if True, a single GPU is used for all the data collection nodes.",
)
parser.add_argument(
    "--trainer_transform",
    action="store_true",
    help="if True, the transforms are applied on the trainer node. "
    "Otherwise, they are done by the workers on the assigned GPU.",
)
parser.add_argument(
    "--wandb_entity",
    type=str,
    default="",
)
parser.add_argument(
    "--wandb_key",
    type=str,
    default="",
)

parser.add_argument(
    "--save_path",
    type=str,
    default="",
)
parser.add_argument(
    "--load_path",
    type=str,
    default="",
)
parser.add_argument(
    "--fraction",
    type=int,
    default=1,
)

torch.cuda.set_device(0)

RETRY_LIMIT = 2
NUM_WORKERS = 8
RETRY_DELAY_SECS = 3

DATA_NODE = "Data"
TRAINER_NODE = "Trainer"
BATCH_SIZE = 128
NUM_COLLECTION = 5


##############################################################################
# Out tensorclass: contains images and target.
#


@tensorclass
class ImageNetData:
    images: torch.Tensor
    targets: torch.Tensor

    @classmethod
    def from_dataset(cls, dataset):
        data = cls(
            images=MemoryMappedTensor.empty(
                (
                    len(dataset),
                    *dataset[0][0].squeeze().shape,
                ),
                dtype=torch.uint8,
            ),
            targets=MemoryMappedTensor.empty(len(dataset), dtype=torch.int64),
            batch_size=[len(dataset)],
        )
        # locks the tensorclass and ensures that is_memmap will return True.
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
                (
                    len(dataset),
                    *dataset[0][0].squeeze().shape,
                ),
                dtype=torch.uint8,
            ),
            targets=MemoryMappedTensor(len(dataset), dtype=torch.int64),
            batch_size=[len(dataset)],
        )
        # locks the tensorclass and ensures that is_memmap will return True.
        data.memmap_()
        t0 = time.time()
        logging.info("loading...", end="\t")
        snapshot = torchsnapshot.Snapshot(path=path)
        sd = dict(data.state_dict())
        app_state = {"state": torchsnapshot.StateDict(data=sd)}
        snapshot.restore(app_state=app_state)
        logging.info(f"done! Took: {time.time()-t0:4.4f}s")
        return data

    def save(self, path):
        import torchsnapshot

        sd = dict(self.state_dict())
        app_state = {"state": torchsnapshot.StateDict(data=sd)}
        torchsnapshot.Snapshot.take(app_state=app_state, path=path)


##############################################################################
# Utils: allow to query a method of an object through RPC.
#


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


##############################################################################
# On-device, random transforms. Notice that these transforms can easily be
# plugged onto a model as the first layer, making them fully compatible
# with production pipelines.
#


class InvAffine(nn.Module):
    """A custom normalization layer."""

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

    @torch.inference_mode()
    def __call__(self, x: ImageNetData):
        # move data to cuda: we first assign the devie to the memmap tensors,
        # then call contiguous() to effectively move them to the right device
        out = x.apply(lambda _tensor: _tensor.as_tensor()).pin_memory().to(self.device)
        if self.transform:
            # apply transforms on gpu
            out.images = self.transform(out.images)
        return out


##############################################################################
# "Trainer" node (even though there is not real training).
# The train method will query data from each data node and collect them on the
# trainer node.
#


@accept_remote_rref_udf_invocation
class DummyTrainerNode:
    def __init__(
        self, world_size: int, single_gpu: bool, local_transform: bool = True
    ) -> None:
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
                InvAffine(
                    loc=loc,
                    scale=scale,
                ),
                RandomCrop(224, 224),
                RandomHFlip(),
            )

    def init(self, train_data_tc):
        self.data = train_data_tc
        for i in range(self.world_size - 1):
            rpc.rpc_sync(
                self.datanodes[i].owner(),
                DataNode.set_data,
                args=(self.datanodes[i], self.data),
            )

    def train(self) -> None:
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
        logging.info(f"time spent: {t:4.4f}s, Rate: {total/t} fps")
        return {"time": t, "rate": total / t}

    def create_data_nodes(self, data):
        for dest_rank in range(1, self.world_size):
            self.create_data_node(dest_rank, local_transform=self.local_transform)
        self.init(data)

    @tenacity.retry(
        stop=tenacity.stop_after_attempt(RETRY_LIMIT),
        wait=tenacity.wait_fixed(RETRY_DELAY_SECS),
        reraise=True,
    )
    def create_data_node(self, node, local_transform) -> rpc.RRef:
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

    def get_data(self):
        return self.data


##############################################################################
# Data nodes.
# The data nodes hold a reference to the data stored on disk, execute indexing
# and return the transformed tensorclass to the training node.
#


@accept_remote_rref_udf_invocation
class DataNode:
    def __init__(
        self,
        rank,
        batch_size: int = BATCH_SIZE,
        single_gpu: bool = False,
        make_transform: bool = True,
    ):
        logging.info("Creating DataNode object")
        self.rank = rank
        self.id = rpc.get_worker_info().id
        self.single_gpu = single_gpu
        # train_ref = rpc.get_worker_info(f"{TRAINER_NODE}")
        # self.train_ref = rpc.remote(
        #     train_ref,
        #     get_trainer,
        # )
        self.batch_size = batch_size
        if self.single_gpu:
            device = "cuda:1"
        else:
            device = f"cuda:{rank}"
        self.make_transform = make_transform
        if self.make_transform:
            loc = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1) * 255
            scale = (
                torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1) * 255
            )
            self.collate_transform = nn.Sequential(
                InvAffine(
                    loc=loc,
                    scale=scale,
                ),
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
        # if not self.initialized:
        #     self._init()
        self.count += 1
        return self.collate(self.data[idx])


##############################################################################
# Some RPC functions responsible for the control flow.
#


def init_rpc(
    rank,
    name,
    world_size,
    single_gpu,
):
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
        # All data must be sent to cuda:0
        for dest_rank in range(1, world_size):
            options.set_device_map(
                f"{DATA_NODE}_{dest_rank}", {0: dest_rank if not single_gpu else 1}
            )

    rpc.init_rpc(
        name,
        rank=rank,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
    )

    logging.info(f"Initialised {name}")


def shutdown():
    rpc.shutdown()


def func(rank, world_size, args, train_data_tc, single_gpu, trainer_transform):
    global trainer
    # access GPU
    torch.randn(1, device="cuda:0")
    if rank == 0:
        trainer = DummyTrainerNode(
            world_size, single_gpu, local_transform=not trainer_transform
        )
        trainer.create_data_nodes(train_data_tc)

        import wandb

        if not args.wandb_key:
            logging.info("no wandb key provided, using it offline")
            mode = "offline"
        else:
            mode = "online"
            wandb.login(key=str(args.wandb_key))
            if not args.wandb_entity:
                raise ValueError("Please indicate the wandb entity.")
        with wandb.init(
            project="dataloading",
            name=f"distributed_w-{args.world_size}_f-{args.fraction}_g-{args.single_gpu}_t-{args.trainer_transform}",
            entity=args.wandb_entity,
            mode=mode,
        ):
            min_time = 10**10
            for i in range(NUM_COLLECTION):
                stats = trainer.train()
                if stats["time"] < min_time:
                    min_time = stats["time"]
                    rate = stats["rate"]
                wandb.log(stats, step=i)
            wandb.log({"min time": min_time, "max_rate": rate})
            logging.info(f"FINAL: time spent: {min_time:4.4f}s, Rate: {rate} fps")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except Exception as err:
        logging.info(f"Could not start mp with spawn method. Error: {err}")

    args = parser.parse_args()
    world_size = args.world_size
    single_gpu = args.single_gpu
    trainer_transform = args.trainer_transform
    save_path = args.save_path
    load_path = args.load_path
    if save_path and load_path:
        raise ValueError("Cannot specify a save_path and a load_path at the same time.")

    names = [TRAINER_NODE, *[f"{DATA_NODE}_{rank}" for rank in range(1, world_size)]]

    logging.info("preparing data")
    data_dir = Path("/datasets01_ontap/imagenet_full_size/061417/")
    train_data_raw = datasets.ImageFolder(
        root=data_dir / "train",
        transform=transforms.Compose(
            [transforms.Resize((256, 256)), transforms.PILToTensor()]
        ),
    )
    train_data_raw.samples = train_data_raw.samples[
        : len(train_data_raw) // args.fraction
    ]

    if load_path:
        logging.info("loading...", end="\t")
        train_data_tc = ImageNetData.load(train_data_raw, load_path)
        logging.info("done")
    else:
        train_data_tc = ImageNetData.from_dataset(train_data_raw)
        if save_path:
            logging.info("saving...", end="\t")
            train_data_tc.save(save_path)
            logging.info("done")

    with mp.Pool(world_size) as pool:
        pool.starmap(
            init_rpc,
            (
                (
                    rank,
                    name,
                    world_size,
                    single_gpu,
                )
                for rank, name in enumerate(names)
            ),
        )
        pool.starmap(
            func,
            (
                (rank, world_size, args, train_data_tc, single_gpu, trainer_transform)
                for rank, name in enumerate(names)
            ),
        )
        pool.apply_async(shutdown)
