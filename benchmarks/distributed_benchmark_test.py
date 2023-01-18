import argparse
import os
import time
import pytest

import torch
from tensordict import MemmapTensor, TensorDict
from torch.distributed import rpc

MAIN_NODE = "Main"
WORKER_NODE = "worker"


def test_distributed(benchmark):
    benchmark.pedantic(exec_distributed_test, iterations=1)


class CloudpickleWrapper(object):
    def __init__(self, fn):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob: bytes):
        import pickle

        self.fn = pickle.loads(ob)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

def exec_distributed_test():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29549"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    str_init_method = "tcp://localhost:10001"
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16, init_method=str_init_method
    )
    rank = None
    if rank == 0:
        rpc.init_rpc(
            MAIN_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )

        # create a tensordict is 1Gb big, stored on disk, assuming that both nodes have access to /tmp/
        tensordict = TensorDict(
            {
                "memmap": MemmapTensor(
                    1000, 640, 640, 3, dtype=torch.uint8, prefix="/tmp/"
                )
            },
            [1000],
        )
        assert tensordict.is_memmap()

        while True:
            try:
                worker_info = rpc.get_worker_info("worker")
                break
            except RuntimeError:
                time.sleep(0.1)
                print("-", end="")
        print("")

        def fill_tensordict(tensordict, idx):
            tensordict[idx] = TensorDict(
                {"memmap": torch.ones(5, 640, 640, 3, dtype=torch.uint8)}, [5]
            )
            return tensordict

        fill_tensordict_cp = CloudpickleWrapper(fill_tensordict)
        idx = [0, 1, 2, 3, 999]
        out = rpc.rpc_sync(
            worker_info,
            fill_tensordict_cp,
            args=(tensordict, idx),
        )

        idx = [4, 5, 6, 7, 998]
        t0 = time.time()
        out = rpc.rpc_sync(
            worker_info,
            fill_tensordict_cp,
            args=(tensordict, idx),
        )

        rpc.shutdown()
    elif rank == 1:
        rpc.init_rpc(
            WORKER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )

        breakpoint()
        rpc.shutdown()
