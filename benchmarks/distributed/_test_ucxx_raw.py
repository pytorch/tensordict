"""Minimal UCXX raw test — bypass TensorDictPipe."""
import asyncio
import os
import socket
import subprocess
import time

import numpy as np

os.environ["UCX_TLS"] = "all"
os.environ["UCX_NET_DEVICES"] = "all"
os.environ["UCX_WARN_UNUSED_ENV_VARS"] = "n"

import ucxx

rank = int(os.environ.get("SLURM_PROCID", 0))
nodelist = os.environ.get("SLURM_NODELIST", "")
r = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                   capture_output=True, text=True)
nodes = r.stdout.strip().split("\n")
master_ip = socket.gethostbyname(nodes[0])
peer_ip = socket.gethostbyname(nodes[1 if rank == 0 else 0])
my_ip = socket.gethostbyname(socket.gethostname())
PORT = 55555

print(f"[rank {rank}] my_ip={my_ip}, peer_ip={peer_ip}", flush=True)

async def run():
    if rank == 0:
        print(f"[rank 0] listening on port {PORT}...", flush=True)
        ep_future = asyncio.get_event_loop().create_future()
        def cb(ep):
            if not ep_future.done():
                ep_future.set_result(ep)
        listener = ucxx.create_listener(cb, port=PORT)
        ep = await ep_future
        print(f"[rank 0] connected!", flush=True)

        # Receive a small array
        buf = np.empty(10, dtype=np.float32)
        await ep.recv(buf)
        print(f"[rank 0] received: {buf[:5]}", flush=True)

        # Receive a larger array (1MB)
        buf_large = np.empty(262144, dtype=np.float32)
        await ep.recv(buf_large)
        print(f"[rank 0] received large: sum={buf_large.sum():.2f}", flush=True)

        # Receive a very large array (100MB)
        buf_xl = np.empty(26214400, dtype=np.float32)
        await ep.recv(buf_xl)
        print(f"[rank 0] received XL: sum={buf_xl.sum():.2f}", flush=True)

        listener.close()
        print(f"[rank 0] done", flush=True)
    else:
        await asyncio.sleep(1)
        print(f"[rank 1] connecting to {peer_ip}:{PORT}...", flush=True)
        ep = await ucxx.create_endpoint(peer_ip, PORT)
        print(f"[rank 1] connected!", flush=True)

        # Send small array
        data = np.arange(10, dtype=np.float32)
        await ep.send(data)
        print(f"[rank 1] sent small", flush=True)

        # Send 1MB
        data_large = np.ones(262144, dtype=np.float32)
        await ep.send(data_large)
        print(f"[rank 1] sent large", flush=True)

        # Send 100MB
        data_xl = np.ones(26214400, dtype=np.float32)
        await ep.send(data_xl)
        print(f"[rank 1] sent XL", flush=True)

        print(f"[rank 1] done", flush=True)

asyncio.run(run())
print(f"[rank {rank}] all done", flush=True)
