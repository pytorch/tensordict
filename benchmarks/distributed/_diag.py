"""Diagnostic: print SLURM env vars and test init_process_group."""
import os
import socket
import subprocess
import sys

print(f"=== DIAG pid={os.getpid()} ===", flush=True)

rank = os.environ.get("SLURM_PROCID", "?")
nodelist = os.environ.get("SLURM_NODELIST", "?")
ntasks = os.environ.get("SLURM_NTASKS", "?")
hostname = socket.gethostname()
ip = socket.gethostbyname(hostname)

print(f"[rank={rank}] hostname={hostname} ip={ip}", flush=True)
print(f"  SLURM_NODELIST={nodelist}", flush=True)
print(f"  SLURM_NTASKS={ntasks}", flush=True)
print(f"  SLURM_NODEID={os.environ.get('SLURM_NODEID', '?')}", flush=True)
print(f"  SLURM_LOCALID={os.environ.get('SLURM_LOCALID', '?')}", flush=True)

try:
    r = subprocess.run(["scontrol", "show", "hostnames", nodelist],
                       capture_output=True, text=True, timeout=5)
    nodes = r.stdout.strip().split("\n") if r.returncode == 0 else [nodelist]
    print(f"  expanded_nodes={nodes}", flush=True)
    if len(nodes) >= 2:
        master_ip = socket.gethostbyname(nodes[0])
        print(f"  master_ip={master_ip}", flush=True)
except Exception as e:
    print(f"  scontrol error: {e}", flush=True)

print("Trying init_process_group...", flush=True)
import torch
import torch.distributed as dist

slurm_rank = int(os.environ.get("SLURM_PROCID", 0))
slurm_ntasks = int(os.environ.get("SLURM_NTASKS", 1))

if len(nodes) >= 2:
    master = socket.gethostbyname(nodes[0])
else:
    master = ip

os.environ["MASTER_ADDR"] = master
os.environ["MASTER_PORT"] = "29500"

print(f"[rank={slurm_rank}] init_process_group(gloo, rank={slurm_rank}, ws={slurm_ntasks}, master={master}:29500)", flush=True)

dist.init_process_group("gloo", rank=slurm_rank, world_size=slurm_ntasks)
print(f"[rank={slurm_rank}] SUCCESS!", flush=True)
dist.barrier()
print(f"[rank={slurm_rank}] barrier passed", flush=True)
dist.destroy_process_group()
print(f"[rank={slurm_rank}] done", flush=True)
