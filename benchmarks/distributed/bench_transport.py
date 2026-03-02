# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark TensorDict distributed transport methods across two nodes.

Compares: torch.distributed send/recv (leaf & consolidated), broadcast,
init_remote/from_remote_init, and UCXX TensorDictPipe.

Usage (on a 2-node Ray cluster):
    # Both nodes:
    python benchmarks/distributed/bench_transport.py

    # Or via steve:
    steve step <JOBID> "python /root/tensordict/benchmarks/distributed/bench_transport.py" --init-ray
"""

from __future__ import annotations

import asyncio
import gc
import os
import socket
import time

# UCX transport config — must be set before ucxx is imported.
# Force-set because cluster step0_env.sh may set IB devices that are
# unavailable inside the container.
os.environ["UCX_TLS"] = "tcp"
os.environ["UCX_NET_DEVICES"] = "all"
os.environ["UCX_WARN_UNUSED_ENV_VARS"] = "n"

import torch
import torch.distributed as dist

from tensordict import TensorDict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

WARMUP = 2
ROUNDS = 10
_ucxx_port_counter = 23456


def _next_ucxx_port():
    global _ucxx_port_counter
    port = _ucxx_port_counter
    _ucxx_port_counter += 2
    return port

SIZES = {
    "1KB": 256,
    "1MB": 262_144,
    "100MB": 26_214_400,
    "1GB": 268_435_456,
}

DEVICES = ["cpu"]
if torch.cuda.is_available():
    DEVICES.append("cuda")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_ip():
    """Get the node's IP address visible to other nodes."""
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)


def _data_bytes(td):
    total = 0
    for v in td.values(True, True):
        if isinstance(v, torch.Tensor):
            total += v.numel() * v.element_size()
    return total


def _fmt_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:8.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:8.2f} ms"
    return f"{seconds:8.3f}  s"


def _fmt_throughput(data_bytes, seconds):
    if seconds <= 0:
        return "    inf GB/s"
    gbps = data_bytes / seconds / 1e9
    return f"{gbps:7.2f} GB/s"


def _make_td(n_floats, device="cpu"):
    return TensorDict(
        {
            "a": torch.randn(n_floats, device=device),
            "b": torch.randn(n_floats // 4 or 1, 4, device=device),
        },
        batch_size=[],
    )


def _barrier():
    dist.barrier()


def _timeit(fn, warmup=WARMUP, rounds=ROUNDS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std


# ---------------------------------------------------------------------------
# torch.distributed benchmarks
# ---------------------------------------------------------------------------


def bench_send_recv_leaf(rank, n_floats, device, group):
    """send/recv, one message per leaf tensor."""
    td = _make_td(n_floats, device)
    td_recv = _make_td(n_floats, device)
    nbytes = _data_bytes(td)
    _barrier()

    if rank == 0:
        mean, std = _timeit(lambda: (td.send(dst=1, group=group), _barrier()))
    else:
        mean, std = _timeit(lambda: (td_recv.recv(src=0, group=group), _barrier()))
    return mean, std, nbytes


def bench_send_recv_consolidated(rank, n_floats, device, group):
    """send/recv with consolidated=True."""
    td = _make_td(n_floats, device)
    td_c = td.consolidate(metadata=True)
    td_recv = _make_td(n_floats, device).consolidate(metadata=True)
    nbytes = _data_bytes(td)
    _barrier()

    if rank == 0:
        mean, std = _timeit(
            lambda: (td_c.send(dst=1, consolidated=True, group=group), _barrier())
        )
    else:
        mean, std = _timeit(
            lambda: (td_recv.recv(src=0, consolidated=True, group=group), _barrier())
        )
    return mean, std, nbytes


def bench_broadcast(rank, n_floats, device, group):
    """broadcast from rank 0."""
    td = _make_td(n_floats, device)
    nbytes = _data_bytes(td)
    _barrier()

    mean, std = _timeit(lambda: (td.broadcast(src=0, group=group), _barrier()))
    return mean, std, nbytes


def bench_init_remote(rank, n_floats, device, group):
    """init_remote + from_remote_init (includes schema transfer)."""
    td = _make_td(n_floats, device)
    nbytes = _data_bytes(td)
    _barrier()

    def _run():
        if rank == 0:
            td.init_remote(dst=1, group=group)
        else:
            TensorDict.from_remote_init(src=0, group=group)
        _barrier()

    mean, std = _timeit(_run)
    return mean, std, nbytes


# ---------------------------------------------------------------------------
# UCXX benchmarks
# ---------------------------------------------------------------------------


def bench_ucxx(rank, n_floats, device, peer_ip):
    """UCXX pipe — measures first-send (with metadata) and steady-state.

    Returns (first_send_time, steady_mean, steady_std, nbytes).
    """
    from tensordict._ucxx import TensorDictPipe

    td = _make_td(n_floats, device)
    nbytes = _data_bytes(td)

    port = _next_ucxx_port()
    _barrier()

    timeout = max(60, n_floats * 4 / 1e8)  # at least 60s, scale with data size

    async def _run():
        if rank == 0:
            pipe = await TensorDictPipe.listen(port)

            # First receive: includes metadata handshake
            t0 = time.perf_counter()
            td_recv = await pipe.arecv()
            first_time = time.perf_counter() - t0

            # Steady-state: schema already known, zero-alloc recv
            steady_times = []
            for i in range(WARMUP + ROUNDS):
                t0 = time.perf_counter()
                await pipe.arecv(td_recv)
                t1 = time.perf_counter()
                if i >= WARMUP:
                    steady_times.append(t1 - t0)

            await pipe.aclose()
        else:
            pipe = await TensorDictPipe.connect(peer_ip, port)

            # First send
            t0 = time.perf_counter()
            await pipe.asend(td)
            first_time = time.perf_counter() - t0

            # Steady-state
            steady_times = []
            for i in range(WARMUP + ROUNDS):
                t0 = time.perf_counter()
                await pipe.asend(td)
                t1 = time.perf_counter()
                if i >= WARMUP:
                    steady_times.append(t1 - t0)

            await pipe.aclose()

        steady_mean = sum(steady_times) / len(steady_times)
        steady_std = (sum((t - steady_mean) ** 2 for t in steady_times) / len(steady_times)) ** 0.5
        return first_time, steady_mean, steady_std

    async def _run_with_timeout():
        return await asyncio.wait_for(_run(), timeout=timeout)

    first_time, steady_mean, steady_std = asyncio.run(_run_with_timeout())
    _barrier()
    return first_time, steady_mean, steady_std, nbytes


# ---------------------------------------------------------------------------
# Discovery via environment or Ray
# ---------------------------------------------------------------------------


def _resolve_slurm_nodelist(nodelist):
    """Expand a SLURM nodelist like 'node[1-3,5]' into a list of hostnames."""
    import subprocess

    result = subprocess.run(
        ["scontrol", "show", "hostnames", nodelist],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip().split("\n")
    return [nodelist]


def _discover_peers():
    """Return (rank, world_size, master_ip, peer_ip, my_ip).

    Discovery order:
    1. RANK + MASTER_ADDR + PEER_ADDR env vars (fully explicit)
    2. SLURM_PROCID + SLURM_NODELIST (standard SLURM 2-node job)
    3. Ray named actors (fallback)
    """
    my_ip = _get_ip()

    # Method 1: explicit env vars
    if "RANK" in os.environ and "MASTER_ADDR" in os.environ:
        rank = int(os.environ["RANK"])
        master_ip = os.environ["MASTER_ADDR"]
        world_size = int(os.environ.get("WORLD_SIZE", 2))
        peer_ip = os.environ.get("PEER_ADDR", master_ip if rank != 0 else my_ip)
        return rank, world_size, master_ip, peer_ip, my_ip

    # Method 2: SLURM
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ.get("SLURM_NTASKS", 2))
        nodelist = os.environ.get("SLURM_NODELIST", "")
        nodes = _resolve_slurm_nodelist(nodelist)
        if len(nodes) >= 2:
            master_hostname = nodes[0]
            master_ip = socket.gethostbyname(master_hostname)
            my_node_idx = min(rank, len(nodes) - 1)
            peer_idx = 1 if my_node_idx == 0 else 0
            peer_ip = socket.gethostbyname(nodes[peer_idx])
        else:
            master_ip = my_ip
            peer_ip = my_ip
        return rank, world_size, master_ip, peer_ip, my_ip

    # Method 3: Ray
    import ray

    if not ray.is_initialized():
        ray.init()

    @ray.remote
    class _NodeInfo:
        def __init__(self):
            self.ips = {}

        def register(self, node_id, ip):
            self.ips[node_id] = ip

        def get_all(self):
            return dict(self.ips)

    info = _NodeInfo.options(name="bench_node_info", get_if_exists=True).remote()
    node_id = ray.get_runtime_context().get_node_id()
    ray.get(info.register.remote(node_id, my_ip))

    for _ in range(120):
        all_ips = ray.get(info.get_all.remote())
        if len(all_ips) >= 2:
            break
        time.sleep(0.5)
    else:
        raise RuntimeError(f"Timed out waiting for 2 nodes, got: {all_ips}")

    sorted_nodes = sorted(all_ips.items(), key=lambda x: x[1])
    node_ids = [nid for nid, _ in sorted_nodes]
    ips = [ip for _, ip in sorted_nodes]

    rank = node_ids.index(node_id)
    master_ip = ips[0]
    peer_ip = ips[1] if rank == 0 else ips[0]

    return rank, 2, master_ip, peer_ip, my_ip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    rank, world_size, master_ip, peer_ip, my_ip = _discover_peers()

    print(  # noqa: T201
        f"[rank {rank}] ip={my_ip} master={master_ip} peer={peer_ip}",
        flush=True,
    )

    # ---- torch.distributed setup ----
    os.environ["MASTER_ADDR"] = master_ip
    os.environ["MASTER_PORT"] = "29500"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
    print(f"[rank {rank}] Initializing process group ({backend})...", flush=True)  # noqa: T201
    dist.init_process_group(
        backend=backend, rank=rank, world_size=world_size,
        init_method=f"tcp://{master_ip}:29500",
    )
    print(f"[rank {rank}] Process group initialized.", flush=True)  # noqa: T201

    if rank == 0:
        print(flush=True)  # noqa: T201
        print(  # noqa: T201
            "=" * 100, flush=True
        )
        print(  # noqa: T201
            f"  TensorDict Distributed Transport Benchmark  "
            f"(warmup={WARMUP}, rounds={ROUNDS})",
            flush=True,
        )
        print(  # noqa: T201
            f"  Nodes: {my_ip} <-> {peer_ip}  |  Backend: {backend}",
            flush=True,
        )
        print("=" * 100, flush=True)  # noqa: T201
        print(flush=True)  # noqa: T201

    # Check if UCXX is available
    try:
        from tensordict._ucxx import TensorDictPipe  # noqa: F401

        has_ucxx = True
    except ImportError:
        has_ucxx = False

    dist_benchmarks = [
        ("send/recv (leaf)", bench_send_recv_leaf),
        ("send/recv (consolidated)", bench_send_recv_consolidated),
        ("broadcast", bench_broadcast),
        ("init_remote/from_remote_init", bench_init_remote),
    ]

    has_ucxx_bench = has_ucxx

    for device in DEVICES:
        if rank == 0:
            print(f"\n--- Device: {device.upper()} ---\n", flush=True)  # noqa: T201
            print(  # noqa: T201
                f"{'Method':<35s} | {'Size':>6s} | {'Latency':>18s} | {'Throughput':>12s}",
                flush=True,
            )
            print("-" * 80, flush=True)  # noqa: T201

        for size_name, n_floats in SIZES.items():
            # Skip 1GB on CPU to avoid OOM issues
            if device == "cpu" and n_floats > 100_000_000:
                continue

            for bench_name, bench_fn in dist_benchmarks:
                gc.collect()
                torch.cuda.empty_cache() if device == "cuda" else None
                _barrier()

                try:
                    mean, std, nbytes = bench_fn(
                        rank, n_floats, device, group=None
                    )
                except Exception as e:
                    if rank == 0:
                        print(  # noqa: T201
                            f"{bench_name:<35s} | {size_name:>6s} | {'ERROR: ' + str(e)[:30]:>18s} |",
                            flush=True,
                        )
                    _barrier()
                    continue

                if rank == 0:
                    print(  # noqa: T201
                        f"{bench_name:<35s} | {size_name:>6s} | "
                        f"{_fmt_time(mean)} +/- {_fmt_time(std)} | "
                        f"{_fmt_throughput(nbytes, mean)}",
                        flush=True,
                    )

            # UCXX over TCP can hang for very large transfers; cap at 10M floats
            ucxx_max_floats = 10_000_000
            if has_ucxx_bench and device == "cpu" and n_floats <= ucxx_max_floats:
                gc.collect()
                _barrier()

                try:
                    first_time, steady_mean, steady_std, nbytes = (
                        bench_ucxx(rank, n_floats, device, peer_ip)
                    )
                except Exception as e:
                    has_ucxx_bench = False
                    if rank == 0:
                        print(  # noqa: T201
                            f"{'UCXX pipe (first send)':<35s} | {size_name:>6s} | {'ERROR: ' + str(e)[:30]:>18s} |",
                            flush=True,
                        )
                        print(  # noqa: T201
                            f"{'UCXX pipe (steady-state)':<35s} | {size_name:>6s} | {'(skipped)':>18s} |",
                            flush=True,
                        )
                else:
                    if rank == 0:
                        print(  # noqa: T201
                            f"{'UCXX pipe (first send)':<35s} | {size_name:>6s} | "
                            f"{_fmt_time(first_time):>18s} | "
                            f"{_fmt_throughput(nbytes, first_time)}",
                            flush=True,
                        )
                        print(  # noqa: T201
                            f"{'UCXX pipe (steady-state)':<35s} | {size_name:>6s} | "
                            f"{_fmt_time(steady_mean)} +/- {_fmt_time(steady_std)} | "
                            f"{_fmt_throughput(nbytes, steady_mean)}",
                            flush=True,
                        )

            if rank == 0:
                print("", flush=True)  # noqa: T201

    if rank == 0:
        print("=" * 80, flush=True)  # noqa: T201
        if not has_ucxx:
            print(  # noqa: T201
                "NOTE: ucxx not installed — UCXX benchmarks skipped.",
                flush=True,
            )
        print("Done.", flush=True)  # noqa: T201

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
