# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compares TensorDict with hand-written PyTorch code on one GPU.

Each section times the same task written with plain tensors (a dict of
tensors, a NumPy memmap, one file per sample, ...) and with TensorDict, in
one process, with the variants of a comparison interleaved. Every timed
region runs between two ``torch.cuda.synchronize()`` calls after a warm-up,
the order of the variants is rotated each round, the seeds are fixed, and
every table reports the median with the 10th-90th percentile range. Before
timing, the script checks that the compared variants give the same results,
and stops with an error if they do not.

Sections:

- A, host-to-device transfer of a structured batch (a few large leaves or
  many small ones): a loop of ``.to(device)`` over a dict, a careful loop
  over pre-pinned tensors with ``non_blocking=True``, ``td.to(device)``, a
  pre-pinned or pre-consolidated tensordict, and
  ``td.to(device, non_blocking_pin=True)`` with several thread counts.
- B, training-step overhead of a small MLP fed from plain tensors or from a
  TensorDict, in eager mode, with ``torch.compile(fullgraph=True)`` and
  with ``tensordict.nn.CudaGraphModule`` around the whole step.
- C, random batches read from a memory-mapped dataset:
  ``TensorDict.memmap`` / ``TensorDict.load_memmap`` against one
  ``torch.save`` file per sample and one NumPy memmap per field.
- D, batches of variable-size systems (atoms per molecule): padded tensors
  with a mask, a TensorDict of jagged nested tensors, a ``lazy_stack`` of
  per-system tensordicts, and flat values with offsets. Build, select,
  transfer and a per-system reduction, plus the memory each layout uses.

Usage::

    python benchmarks/scripts/structured_batch_bench.py [--sections A B C D]
        [--device cuda] [--quick] [--pin-alone] [--json out.json]
        [--plot out_dir] [--data-dir DIR] [--memmap-mb 512]

``--quick`` runs toy sizes in seconds; ``--device cpu --quick`` is a smoke
test for machines without a GPU (the variants that need CUDA, such as
pinned memory and CUDA graphs, are skipped and listed). ``--pin-alone``
also times the section A variants that pin inside the call each on its own
rather than interleaved, so that each reuses its own pinned blocks.
Section C writes its dataset (about ``--memmap-mb`` MB in total) to a
temporary directory under ``--data-dir`` and deletes it at the end.
``--plot`` writes one figure per section (requires matplotlib).
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tensordict
import torch
from tensordict import lazy_stack, TensorDict
from tensordict.nn import CudaGraphModule, TensorDictModule, TensorDictSequential
from torch import nn

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None

MiB = 2**20


def log(*args):
    print(f"[{time.strftime('%H:%M:%S')}]", *args, file=sys.stderr, flush=True)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stats(xs):
    a = np.asarray(xs, dtype=float)
    return {
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "n": int(a.size),
    }


def interleaved(variants, device, n_iter, n_warmup=5, inner=1):
    """Times each variant ``n_iter`` times, interleaved and rotated.

    ``variants`` maps a name to a zero-argument callable. With ``inner > 1``,
    one timed region runs the callable ``inner`` times and the sample is the
    time per call. Returns name -> summary statistics in seconds.
    """
    names = list(variants)
    for _ in range(n_warmup):
        for name in names:
            variants[name]()
    sync(device)
    samples = {name: [] for name in names}
    for r in range(n_iter):
        k = r % len(names)
        for name in names[k:] + names[:k]:
            fn = variants[name]
            sync(device)
            t0 = time.perf_counter()
            for _ in range(inner):
                res = fn()
            sync(device)
            samples[name].append((time.perf_counter() - t0) / inner)
            del res
    return {name: stats(xs) for name, xs in samples.items()}


def check(cond, msg):
    if not cond:
        raise AssertionError(f"compared variants disagree: {msg}")


def timing_table(title, timings, **extra):
    """A table of timings; ``extra`` maps a column name to a function of the median."""
    for s in timings.values():
        for col, fn in extra.items():
            s[col] = fn(s["median"])
    return {"title": title, "variants": timings}


def info_table(title, values):
    return {"title": title, "values": values}


def env_info(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "tensordict": getattr(tensordict, "__version__", "unknown"),
        "cpu": platform.processor() or platform.machine(),
        "device": str(device),
    }
    try:
        with open("/proc/cpuinfo") as f:
            models = [
                line.split(":", 1)[1].strip() for line in f if "model name" in line
            ]
        if models:
            info["cpu"] = models[0]
    except OSError:
        pass
    if device.type == "cuda":
        info["cuda"] = torch.version.cuda
        info["gpu"] = torch.cuda.get_device_name(device)
    return info


# --------------------------------------------------------------------------
# A. Host-to-device transfer
# --------------------------------------------------------------------------


def transfer_variants(d, td, device):
    """The section A variants for one batch, and the names skipped off CUDA."""
    variants = {
        "dict: loop of .to(device)": lambda: {k: v.to(device) for k, v in d.items()},
        "td.to(device)": lambda: td.to(device),
    }
    if device.type != "cuda":
        return variants, [
            "dict: pinned (pre), non_blocking, 1 sync",
            "dict: pin_memory() + non_blocking in loop",
            "td.pin_memory() (pre), to(non_blocking=True)",
            "td.consolidate(pin_memory=True) (pre), to(non_blocking=True)",
            "td.to(device, non_blocking_pin=True[, num_threads=...])",
        ]
    d_pinned = {k: v.clone().pin_memory() for k, v in d.items()}
    td_pinned = td.pin_memory()
    td_cons = td.consolidate(pin_memory=True)
    variants.update(
        {
            "dict: pinned (pre), non_blocking, 1 sync": lambda: {
                k: v.to(device, non_blocking=True) for k, v in d_pinned.items()
            },
            "dict: pin_memory() + non_blocking in loop": lambda: {
                k: v.pin_memory().to(device, non_blocking=True) for k, v in d.items()
            },
            "td.pin_memory() (pre), to(non_blocking=True)": lambda: td_pinned.to(
                device, non_blocking=True
            ),
            "td.consolidate(pin_memory=True) (pre), to(non_blocking=True)": lambda: (
                td_cons.to(device, non_blocking=True)
            ),
        }
    )
    variants.update(pin_inside_variants(td, device))
    return variants, []


def pin_inside_variants(td, device):
    # The default num_threads is max(1, torch.get_num_threads() // 2).
    return {
        "td.to(device, non_blocking_pin=True)": lambda: td.to(
            device, non_blocking_pin=True
        ),
        "td.to(device, non_blocking_pin=True, num_threads=8)": lambda: td.to(
            device, non_blocking_pin=True, num_threads=8
        ),
        "td.to(device, non_blocking_pin=True, num_threads=0)": lambda: td.to(
            device, non_blocking_pin=True, num_threads=0
        ),
    }


def section_a(device, quick, pin_alone):
    cuda = device.type == "cuda"
    ks = [8, 64] if quick else [8, 64, 512]
    total = (1 if quick else 64) * MiB
    cases = [(f"{total // MiB} MiB in {k} leaves", k, total // k) for k in ks]
    cases += [(f"{k} leaves of 4 KiB", k, 4096) for k in ks]
    n_iter = 20 if quick else 200
    n_alone = 20 if quick else 100
    tables, skipped = [], []
    for case, k, leaf_bytes in cases:
        torch.manual_seed(0)
        d = {f"leaf{i:03d}": torch.randn(leaf_bytes // 4) for i in range(k)}
        td = TensorDict(d, batch_size=[])
        nbytes = sum(v.numel() * v.element_size() for v in d.values())
        variants, skipped = transfer_variants(d, td, device)
        for name, fn in variants.items():
            res = fn()
            sync(device)
            for key in d:
                check(torch.equal(res[key].cpu(), d[key]), f"A {case}, {name}, {key}")
            del res
        timings = interleaved(variants, device, n_iter, n_warmup=10)
        tables.append(
            timing_table(f"A: {case}", timings, GBps=lambda t, n=nbytes: n / t / 1e9)
        )
        log(f"A {case} done")
        # The variants that pin inside the call, and td.to(device) for
        # reference, each timed alone so that it reuses its own pinned blocks:
        # the large-leaf cases and the largest small-leaf case.
        if pin_alone and cuda and (leaf_bytes > 4096 or k == ks[-1]):
            alone = {}
            for name, fn in variants.items():
                if (
                    name == "td.to(device)"
                    or "pin_memory() +" in name
                    or ("non_blocking_pin" in name)
                ):
                    alone.update(interleaved({name: fn}, device, n_alone, n_warmup=10))
            tables.append(
                timing_table(
                    f"A, each variant timed alone: {case}",
                    alone,
                    GBps=lambda t, n=nbytes: n / t / 1e9,
                )
            )
            log(f"A {case} timed alone done")
        del d, td, variants
        gc.collect()
    notes = []
    if skipped:
        notes.append(f"skipped on {device.type} (need CUDA): " + "; ".join(skipped))
    if pin_alone and not cuda:
        notes.append(f"--pin-alone skipped on {device.type} (needs CUDA)")
    return {"tables": tables, "notes": notes}


# --------------------------------------------------------------------------
# B. Training-step overhead
# --------------------------------------------------------------------------

IN_KEYS = {"obs": 32, "goal": 16, "action": 16}


def make_mlp(hidden, device):
    torch.manual_seed(1)
    d_in = sum(IN_KEYS.values())
    return nn.Sequential(
        nn.Linear(d_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    ).to(device)


def make_opt(model, device):
    return torch.optim.Adam(
        model.parameters(), lr=1e-4, capturable=device.type == "cuda"
    )


def loss_plain(obs, goal, action, target, m):
    x = torch.cat([obs, goal, action], -1)
    return (m(x) - target).pow(2).mean()


def loss_td(td, m):
    x = torch.cat([td["obs"], td["goal"], td["action"]], -1)
    return (m(x) - td["target"]).pow(2).mean()


class Cycle:
    """Returns the batches of a pool in turn."""

    def __init__(self, pool):
        self.pool = pool
        self.i = 0

    def __call__(self):
        b = self.pool[self.i % len(self.pool)]
        self.i += 1
        return b


def on_cycle(fn, pool):
    """A zero-argument callable that applies ``fn`` to the batches of ``pool`` in turn."""
    nxt = Cycle(pool)
    return lambda: fn(nxt())


def eager_step(loss_fn, pool, m, device):
    """A training step: zero_grad, ``loss_fn(batch, m)``, backward, Adam."""
    opt = make_opt(m, device)
    nxt = Cycle(pool)

    def step():
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(nxt(), m)
        loss.backward()
        opt.step()
        return loss.detach()

    return step


def train_step_variants(pool_t, pool_td, hidden, device):
    cuda = device.type == "cuda"
    compiled_plain = torch.compile(loss_plain, fullgraph=True)
    compiled_td = torch.compile(loss_td, fullgraph=True)

    def plain(fn):
        return lambda b, m: fn(b["obs"], b["goal"], b["action"], b["target"], m)

    variants = {}
    m_ref = make_mlp(hidden, device)
    variants["plain tensors, eager"] = eager_step(
        plain(loss_plain), pool_t, m_ref, device
    )
    # A/A control: the same code with its own model, to show the spread that
    # comes from the identity of a variant alone (allocation, order).
    variants["plain tensors, eager (A/A repeat)"] = eager_step(
        plain(loss_plain), pool_t, make_mlp(hidden, device), device
    )
    variants["TensorDict, eager"] = eager_step(
        loss_td, pool_td, make_mlp(hidden, device), device
    )

    # tensordict.nn modules; the step works on a shallow copy of the batch so
    # that the pool does not keep the outputs.
    m_mod = make_mlp(hidden, device)
    tdmod = TensorDictSequential(
        TensorDictModule(
            lambda o, g, a: torch.cat([o, g, a], -1),
            in_keys=list(IN_KEYS),
            out_keys=["x"],
        ),
        TensorDictModule(m_mod, in_keys=["x"], out_keys=["pred"]),
    )

    def loss_module(td, m):  # m is the model that tdmod wraps
        out = tdmod(td.copy())
        return (out["pred"] - out["target"]).pow(2).mean()

    variants["TensorDictModule, eager"] = eager_step(
        loss_module, pool_td, m_mod, device
    )

    # torch.compile(fullgraph=True) of forward and loss; backward and Adam eager.
    m_compile = make_mlp(hidden, device)
    variants["plain tensors, compile"] = eager_step(
        plain(compiled_plain), pool_t, m_compile, device
    )
    variants["TensorDict, compile"] = eager_step(
        compiled_td, pool_td, make_mlp(hidden, device), device
    )

    skipped = []
    if cuda:
        # The whole step (zero_grad, forward, backward, Adam) in one CUDA graph.
        m_pg = make_mlp(hidden, device)
        o_pg = make_opt(m_pg, device)

        def full_plain(obs, goal, action, target):
            o_pg.zero_grad(set_to_none=False)
            loss = loss_plain(obs, goal, action, target, m_pg)
            loss.backward()
            o_pg.step()
            return loss.detach()

        g_plain = CudaGraphModule(full_plain, warmup=3)
        nxt_pg = Cycle(pool_t)

        def step_plain_graph():
            b = nxt_pg()
            return g_plain(b["obs"], b["goal"], b["action"], b["target"])

        variants["plain tensors, CudaGraphModule"] = step_plain_graph

        m_tg = make_mlp(hidden, device)
        o_tg = make_opt(m_tg, device)

        def full_td(td):
            o_tg.zero_grad(set_to_none=False)
            loss = loss_td(td, m_tg)
            loss.backward()
            o_tg.step()
            # CudaGraphModule needs a function that takes a tensordict to
            # return a tensordict too, so the loss goes back in one.
            return TensorDict(loss=loss.detach())

        g_td = CudaGraphModule(full_td, warmup=3, in_keys=[])
        nxt_tg = Cycle(pool_td)
        variants["TensorDict, CudaGraphModule"] = lambda: g_td(nxt_tg())["loss"]
    else:
        skipped = ["plain tensors, CudaGraphModule", "TensorDict, CudaGraphModule"]

    # Same weights, same batch: the compiled losses match the eager one.
    b0 = pool_t[0]
    with torch.no_grad():
        l_eager = float(plain(loss_plain)(b0, m_compile))
        l_plain = float(plain(compiled_plain)(b0, m_compile))
        l_td = float(compiled_td(pool_td[0], m_compile))
    for name, value in (("plain", l_plain), ("TensorDict", l_td)):
        check(
            math.isclose(value, l_eager, rel_tol=1e-4),
            f"B hidden={hidden}: initial loss, eager {l_eager} vs compiled {name} {value}",
        )
    return variants, skipped, sum(p.numel() for p in m_ref.parameters())


def section_b(device, quick):
    rows = 256 if quick else 4096
    hiddens = [64, 512] if quick else [64, 512, 4096]
    n_iter = 10 if quick else 50
    inner = 5 if quick else 20
    g = torch.Generator().manual_seed(0)
    pool_t = []
    for _ in range(4):
        batch = {k: torch.randn(rows, d, generator=g) for k, d in IN_KEYS.items()}
        batch["target"] = torch.randn(rows, 1, generator=g)
        pool_t.append({k: v.to(device) for k, v in batch.items()})
    pool_td = [TensorDict(b, batch_size=[rows], device=device) for b in pool_t]
    tables, skipped = [], []
    for hidden in hiddens:
        variants, skipped, n_params = train_step_variants(
            pool_t, pool_td, hidden, device
        )
        # Warm-up (compilation, graph capture) outside the timed rounds.
        t0 = time.perf_counter()
        for fn in variants.values():
            for _ in range(10):
                fn()
        sync(device)
        log(f"B hidden={hidden}: warm-up {time.perf_counter() - t0:.1f}s")
        timings = interleaved(variants, device, n_iter, n_warmup=3, inner=inner)
        # Every variant has now trained its own copy of the same model on the
        # same batches for the same number of steps, so each TensorDict variant
        # must match its plain-tensor twin in the same mode. Eager and compiled
        # kernels round differently and the gap grows with every step (about 6%
        # after the full-size run on an H100), so the modes are compared only on
        # the first step, above.
        finals = {name: float(fn()) for name, fn in variants.items()}
        twins = {
            "plain tensors, eager (A/A repeat)": "plain tensors, eager",
            "TensorDict, eager": "plain tensors, eager",
            "TensorDictModule, eager": "plain tensors, eager",
            "TensorDict, compile": "plain tensors, compile",
            "TensorDict, CudaGraphModule": "plain tensors, CudaGraphModule",
        }
        for name, twin in twins.items():
            if name in finals and twin in finals:
                check(
                    math.isclose(finals[name], finals[twin], rel_tol=1e-4),
                    f"B hidden={hidden}: final loss of {name!r} is "
                    f"{finals[name]}, {twin!r} gives {finals[twin]}",
                )
        title = (
            f"B: training step, batch {rows}, MLP hidden {hidden} "
            f"({n_params:,} parameters), time per step"
        )
        tables.append(timing_table(title, timings))
        del variants
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    notes = []
    if skipped:
        notes.append(f"skipped on {device.type} (need CUDA): " + "; ".join(skipped))
    return {"tables": tables, "notes": notes}


# --------------------------------------------------------------------------
# C. Memory-mapped dataset reads
# --------------------------------------------------------------------------

FIELDS_C = {
    "image": ((4, 64, 64), torch.uint8),  # 16,384 B
    "state": ((8, 1024), torch.float32),  # 32,768 B
    "action": ((4096,), torch.float32),  # 16,384 B
    "label": ((1,), torch.int64),  # 8 B
}
SAMPLE_BYTES_C = sum(
    math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    for shape, dtype in FIELDS_C.values()
)


def gen_field(n, shape, dtype, g):
    if dtype == torch.uint8:
        return torch.randint(0, 256, (n, *shape), dtype=dtype, generator=g)
    if dtype == torch.int64:
        return torch.randint(0, 1000, (n, *shape), dtype=dtype, generator=g)
    return torch.randn((n, *shape), generator=g, dtype=dtype)


def dir_bytes(p):
    return sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())


def dataset_sizes_c(memmap_mb):
    """Samples in the memmap and NumPy copies, and in the one-file-per-sample copy.

    The dataset is written three times: as a TensorDict memmap and as NumPy
    arrays (``n`` samples each), and as one ``torch.save`` file per sample
    for the first ``n // 4`` samples, so that the total is ``memmap_mb`` MB.
    """
    n = max(int(memmap_mb * MiB / (2.25 * SAMPLE_BYTES_C)), 64)
    return n, n // 4


def section_c(device, quick, data_dir, memmap_mb):
    cuda = device.type == "cuda"
    n, n_files = dataset_sizes_c(memmap_mb)
    batch = 32 if quick else 256
    n_iter = 10 if quick else 100
    root = Path(tempfile.mkdtemp(prefix="section_c_", dir=data_dir))
    tables = []
    try:
        g = torch.Generator().manual_seed(0)
        data = TensorDict(
            {k: gen_field(n, s, dt, g) for k, (s, dt) in FIELDS_C.items()},
            batch_size=[n],
        )
        writes = {}
        t0 = time.perf_counter()
        data.memmap(root / "td_memmap", num_threads=8)
        writes["TensorDict.memmap"] = time.perf_counter() - t0
        (root / "pt").mkdir()
        t0 = time.perf_counter()
        for i in range(n_files):
            sample = {k: data[k][i].clone() for k in FIELDS_C}
            torch.save(sample, root / "pt" / f"{i:05d}.pt")
        writes["torch.save per sample"] = time.perf_counter() - t0
        (root / "npy").mkdir()
        t0 = time.perf_counter()
        for k in FIELDS_C:
            np.save(root / "npy" / f"{k}.npy", data[k].numpy())
        writes["np.save per field"] = time.perf_counter() - t0
        os.sync()
        sizes = {
            "TensorDict.memmap": dir_bytes(root / "td_memmap"),
            "torch.save per sample": dir_bytes(root / "pt"),
            "np.save per field": dir_bytes(root / "npy"),
        }
        tables.append(
            info_table(
                f"C: dataset written ({n} samples of {SAMPLE_BYTES_C:,} B; "
                f"{n_files} sample files)",
                {
                    name: f"{sizes[name] / MiB:.1f} MiB in {writes[name]:.2f} s"
                    for name in writes
                },
            )
        )
        del data
        gc.collect()

        mm = TensorDict.load_memmap(root / "td_memmap")
        npys = {k: np.load(root / "npy" / f"{k}.npy", mmap_mode="r") for k in FIELDS_C}
        files = [root / "pt" / f"{i:05d}.pt" for i in range(n_files)]
        # Warm the page cache: read every file once.
        for f in root.rglob("*"):
            if f.is_file():
                with open(f, "rb") as fh:
                    while fh.read(64 * MiB):
                        pass

        rng = np.random.default_rng(0)
        idx_pool = [torch.from_numpy(rng.integers(0, n, batch)) for _ in range(64)]
        file_idx_pool = [rng.integers(0, n_files, batch) for _ in range(64)]
        nxt_idx, nxt_file_idx = Cycle(idx_pool), Cycle(file_idx_pool)

        def pin(t):
            return t.pin_memory() if cuda else t

        def torch_load_batch():
            samples = [torch.load(files[i], weights_only=True) for i in nxt_file_idx()]
            # pinned on CUDA, like the other readers that pass non_blocking
            return {
                k: pin(torch.stack([s[k] for s in samples])).to(
                    device, non_blocking=True
                )
                for k in FIELDS_C
            }

        variants = {
            "TensorDict.load_memmap()[idx].to(device)": lambda: mm[nxt_idx()].to(
                device
            ),
            "np.load(mmap_mode='r')[idx] per field, .to(device)": lambda: {
                k: torch.from_numpy(a[nxt_idx().numpy()]).to(device)
                for k, a in npys.items()
            },
            "torch.load per sample file + torch.stack": torch_load_batch,
        }
        skipped = []
        if cuda:
            variants[
                "TensorDict.load_memmap()[idx].pin_memory().to(device, non_blocking)"
            ] = lambda: mm[nxt_idx()].pin_memory().to(device, non_blocking=True)
            variants["np.load(mmap_mode='r')[idx] per field, pinned, non_blocking"] = (
                lambda: {
                    k: torch.from_numpy(a[nxt_idx().numpy()])
                    .pin_memory()
                    .to(device, non_blocking=True)
                    for k, a in npys.items()
                }
            )
        else:
            skipped = ["the two pinned memmap readers"]

        # The readers agree on one batch.
        idx = idx_pool[0]
        a = mm[idx]
        sample = torch.load(files[int(file_idx_pool[0][0])], weights_only=True)
        for k in FIELDS_C:
            check(
                torch.equal(a[k], torch.from_numpy(npys[k][idx.numpy()])),
                f"C memmap vs NumPy, {k}",
            )
            check(
                torch.equal(sample[k], mm[int(file_idx_pool[0][0])][k]),
                f"C memmap vs torch.load, {k}",
            )
        timings = interleaved(variants, device, n_iter, n_warmup=5)
        tables.append(
            timing_table(
                f"C: random batch of {batch} samples to {device.type}, warm page cache",
                timings,
                samples_per_s=lambda t: batch / t,
                GBps=lambda t: batch * SAMPLE_BYTES_C / t / 1e9,
            )
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    log(f"C removed {root}")
    notes = []
    if skipped:
        notes.append(f"skipped on {device.type} (need CUDA): " + "; ".join(skipped))
    return {"tables": tables, "notes": notes}


# --------------------------------------------------------------------------
# D. Batches of variable-size systems
# --------------------------------------------------------------------------

PER_ATOM = ["pos", "species", "forces"]


def storage_bytes(tensors):
    seen = {}
    for t in tensors:
        if t.is_nested:
            parts = [t.values(), t.offsets()]
            if t.lengths() is not None:
                parts.append(t.lengths())
        else:
            parts = [t]
        for p in parts:
            s = p.untyped_storage()
            seen[(s.data_ptr(), p.device)] = s.nbytes()
    return int(sum(seen.values()))


def td_tensors(td):
    if hasattr(td, "tensordicts"):  # lazy stack: its leaves live in the sub-tds
        return [v for sub in td.tensordicts for v in sub.values(True, True)]
    return list(td.values(True, True))


def offsets_of(lengths):
    return torch.cat([torch.zeros(1, dtype=torch.long), lengths.cumsum(0)])


def build_padded(samples):
    out = {
        k: nn.utils.rnn.pad_sequence([s[k] for s in samples], batch_first=True)
        for k in PER_ATOM
    }
    lengths = torch.tensor([s["pos"].shape[0] for s in samples])
    out["mask"] = torch.arange(out["pos"].shape[1])[None] < lengths[:, None]
    out["energy"] = torch.stack([s["energy"] for s in samples])
    return out


def build_flat(samples):
    out = {k: torch.cat([s[k] for s in samples]) for k in PER_ATOM}
    lengths = torch.tensor([s["pos"].shape[0] for s in samples])
    out["offsets"] = offsets_of(lengths)
    out["batch"] = torch.repeat_interleave(torch.arange(len(samples)), lengths)
    out["energy"] = torch.stack([s["energy"] for s in samples])
    return out


def build_njt(samples):
    offsets = offsets_of(torch.tensor([s["pos"].shape[0] for s in samples]))
    leaves = {
        k: torch.nested.nested_tensor_from_jagged(
            torch.cat([s[k] for s in samples]), offsets
        )
        for k in PER_ATOM
    }
    leaves["energy"] = torch.stack([s["energy"] for s in samples])
    return TensorDict(leaves, batch_size=[len(samples)])


def select_flat(flat, idx):
    off = flat["offsets"]
    starts, lengths = off[:-1][idx], off.diff()[idx]
    new_off = offsets_of(lengths)
    atom = torch.arange(int(new_off[-1])) + torch.repeat_interleave(
        starts - new_off[:-1], lengths
    )
    res = {k: flat[k][atom] for k in PER_ATOM}
    res["offsets"] = new_off
    res["batch"] = torch.repeat_interleave(torch.arange(len(idx)), lengths)
    res["energy"] = flat["energy"][idx]
    return res


def section_d(device, quick, data_dir):
    cuda = device.type == "cuda"
    b = 32 if quick else 256
    n_lo, n_hi = (10, 50) if quick else (10, 500)
    b_sel = 8 if quick else 64
    n_iter = 10 if quick else 100
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    ns = rng.integers(n_lo, n_hi + 1, b).tolist()
    samples = [
        {
            "pos": torch.randn(n, 3),
            "species": torch.randint(0, 100, (n,)),
            "forces": torch.randn(n, 3),
            "energy": torch.randn(()),
        }
        for n in ns
    ]
    sample_tds = [TensorDict(s, batch_size=[]) for s in samples]
    padded, flat, njt = build_padded(samples), build_flat(samples), build_njt(samples)
    lazy = lazy_stack([TensorDict(s, batch_size=[]) for s in samples])
    names = {
        "padded": "padded + mask",
        "njt": "TensorDict of NJTs (shared offsets)",
        "lazy": "lazy_stack of TensorDicts",
        "flat": "flat + offsets + batch index",
    }
    desc = f"{b} systems of {n_lo}-{n_hi} atoms"
    tables, skipped = [], []
    payload = sum(s[k].numel() * s[k].element_size() for s in samples for k in s)
    memory = {
        names["padded"]: storage_bytes(list(padded.values())),
        names["njt"]: storage_bytes(td_tensors(njt)),
        names["lazy"]: storage_bytes(td_tensors(lazy)),
        names["flat"]: storage_bytes(list(flat.values())),
    }
    tables.append(
        info_table(
            f"D: memory of {desc} (payload {payload:,} B)",
            {k: f"{v:,} B ({v / payload:.2f}x payload)" for k, v in memory.items()},
        )
    )

    build = interleaved(
        {
            names["padded"]: lambda: build_padded(samples),
            names["njt"]: lambda: build_njt(samples),
            f"{names['lazy']} (from dicts)": lambda: lazy_stack(
                [TensorDict(s, batch_size=[]) for s in samples]
            ),
            f"{names['lazy']} (pre-built TDs)": lambda: lazy_stack(sample_tds),
            "lazy_stack(...).densify(layout=jagged)": lambda: lazy_stack(
                sample_tds
            ).densify(layout=torch.jagged),
            names["flat"]: lambda: build_flat(samples),
        },
        device,
        n_iter,
    )
    tables.append(timing_table(f"D: build a batch of {desc}", build))

    # Select b_sel random systems from the batch (on the CPU).
    sel_pool = [
        torch.from_numpy(rng.choice(b, b_sel, replace=False)) for _ in range(32)
    ]
    select_fns = {
        names["padded"]: lambda idx: {k: v[idx] for k, v in padded.items()},
        names["njt"]: lambda idx: njt[idx],
        names["lazy"]: lambda idx: lazy[idx],
        names["flat"]: lambda idx: select_flat(flat, idx),
    }
    sel = {name: fn(sel_pool[0]) for name, fn in select_fns.items()}
    for j, i in enumerate(sel_pool[0].tolist()):
        ref, n = samples[i]["pos"], ns[i]
        sf = sel[names["flat"]]
        for name, got in (
            (names["padded"], sel[names["padded"]]["pos"][j, :n]),
            (names["njt"], sel[names["njt"]]["pos"][j]),
            (names["lazy"], sel[names["lazy"]][j]["pos"]),
            (names["flat"], sf["pos"][sf["offsets"][j] : sf["offsets"][j + 1]]),
        ):
            check(torch.equal(got, ref), f"D select, {name}, system {i}")
    select = interleaved(
        {name: on_cycle(fn, sel_pool) for name, fn in select_fns.items()},
        device,
        n_iter,
    )
    tables.append(timing_table(f"D: select {b_sel} of {desc}", select))

    to_device = {
        names["padded"]: lambda: {k: v.to(device) for k, v in padded.items()},
        names["njt"]: lambda: njt.to(device),
        names["lazy"]: lambda: lazy.to(device),
        names["flat"]: lambda: {k: v.to(device) for k, v in flat.items()},
    }
    if cuda:
        njt_cons = njt.consolidate(pin_memory=True)
        lazy_cons = lazy.consolidate(pin_memory=True)
        to_device[f"{names['njt']}, consolidated+pinned (pre)"] = lambda: njt_cons.to(
            device, non_blocking=True
        )
        to_device["lazy_stack, consolidated+pinned (pre)"] = lambda: lazy_cons.to(
            device, non_blocking=True
        )
    else:
        skipped.append("the consolidated+pinned transfers")
    tables.append(
        timing_table(
            f"D: move {desc} to {device.type}",
            interleaved(to_device, device, n_iter),
        )
    )

    # A per-atom op and a per-system reduction on the device.
    padded_g = {k: v.to(device) for k, v in padded.items()}
    flat_g = {k: v.to(device) for k, v in flat.items()}
    njt_g, lazy_g = njt.to(device), lazy.to(device)

    def op_padded():
        return ((padded_g["pos"] ** 2).sum(-1) * padded_g["mask"]).sum(1)

    def op_flat():
        v = (flat_g["pos"] ** 2).sum(-1)
        return torch.zeros(b, device=device, dtype=v.dtype).index_add_(
            0, flat_g["batch"], v
        )

    def op_flat_segment():
        v = (flat_g["pos"] ** 2).sum(-1)
        return torch.segment_reduce(v, "sum", lengths=flat_g["offsets"].diff())

    def op_njt():
        return (njt_g["pos"] ** 2).sum(-1).sum(1)

    def op_lazy_loop():
        return torch.stack([(t["pos"] ** 2).sum() for t in lazy_g.tensordicts])

    def op_lazy_as_njt():
        p = lazy_g.get("pos", as_nested_tensor=True, layout=torch.jagged)
        return (p**2).sum(-1).sum(1)

    ops = {
        names["padded"]: op_padded,
        f"{names['padded']}, torch.compile": torch.compile(op_padded, fullgraph=True),
        names["njt"]: op_njt,
        "TensorDict of NJTs, torch.compile": torch.compile(op_njt, fullgraph=True),
        "lazy_stack: loop over systems": op_lazy_loop,
        "lazy_stack: get(as_nested_tensor=True) then NJT op": op_lazy_as_njt,
        "flat: index_add_ by batch index": op_flat,
        "flat: segment_reduce by lengths": op_flat_segment,
    }
    ref = torch.stack([(s["pos"] ** 2).sum() for s in samples]).to(device)
    for name, fn in ops.items():
        res = fn()
        check(
            torch.allclose(res, ref, rtol=1e-4, atol=1e-3),
            f"D per-system reduction, {name}",
        )
    tables.append(
        timing_table(
            f"D: sum of |pos|^2 per system on {device.type}, {desc}",
            interleaved(ops, device, n_iter),
        )
    )

    # What each representation cannot do (with this torch and tensordict).
    probes = {}
    probe_dir = Path(tempfile.mkdtemp(prefix="section_d_", dir=data_dir))
    try:
        for name, fn in {
            "NJT td: memmap": lambda: njt.memmap(probe_dir / "njt"),
            "lazy_stack: memmap": lambda: lazy.memmap(probe_dir / "lazy"),
            "NJT td: td[idx] (tensor index)": lambda: njt[sel_pool[0]],
            "raw NJT: njt[tensor index]": lambda: njt["pos"][sel_pool[0]],
            "NJT td: torch.stack of two batches": lambda: torch.stack([njt, njt]),
            "NJT td: torch.cat of two batches": lambda: torch.cat([njt, njt]),
            "lazy_stack: torch.cat of two batches": lambda: torch.cat([lazy, lazy]),
            "NJT td: pos * forces (shared ragged dim)": lambda: (
                njt["pos"] * njt["forces"]
            ),
            "lazy_stack: ls['pos'] as one dense tensor": lambda: lazy["pos"],
        }.items():
            try:
                fn()
                probes[name] = "ok"
            except Exception as e:  # noqa: BLE001
                probes[name] = f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"
    finally:
        shutil.rmtree(probe_dir, ignore_errors=True)
    tables.append(info_table("D: capabilities", probes))
    notes = []
    if skipped:
        notes.append(f"skipped on {device.type} (need CUDA): " + "; ".join(skipped))
    return {"tables": tables, "notes": notes}


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------


def human(t):
    if t < 1e-3:
        return f"{t * 1e6:.1f} us"
    if t < 1:
        return f"{t * 1e3:.2f} ms"
    return f"{t:.2f} s"


EXTRA_COLUMNS = {"GBps": ("GB/s", "{:.2f}"), "samples_per_s": ("samples/s", "{:,.0f}")}


def print_section(name, result):
    print(f"\n## Section {name}\n")
    for note in result["notes"]:
        print(f"Note: {note}\n")
    for table in result["tables"]:
        print(f"### {table['title']}\n")
        if "values" in table:
            print("| | |\n|---|---|")
            for key, value in table["values"].items():
                # an error message can hold a "|", which would end the cell
                value = str(value).replace("|", "\\|")
                print(f"| {key} | {value} |")
            print()
            continue
        extra = [
            c for c in EXTRA_COLUMNS if c in next(iter(table["variants"].values()))
        ]
        header = ["variant", "median", "p10-p90"] + [EXTRA_COLUMNS[c][0] for c in extra]
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for variant, s in table["variants"].items():
            cells = [
                variant,
                human(s["median"]),
                f"{human(s['p10'])} - {human(s['p90'])}",
            ]
            cells += [EXTRA_COLUMNS[c][1].format(s[c]) for c in extra]
            print("| " + " | ".join(cells) + " |")
        print()


def plot(name, result, out_dir):
    if not _has_matplotlib:
        raise ImportError("Plotting requires matplotlib (pip install matplotlib).")
    import matplotlib.pyplot as plt

    tables = [t for t in result["tables"] if "variants" in t]
    if not tables:
        return
    ncols = 2 if len(tables) > 1 else 1
    nrows = math.ceil(len(tables) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, table in zip(axes, tables):
        names = list(table["variants"])
        med = np.array([table["variants"][n]["median"] for n in names])
        lo = med - np.array([table["variants"][n]["p10"] for n in names])
        hi = np.array([table["variants"][n]["p90"] for n in names]) - med
        y = np.arange(len(names))
        ax.barh(y, med, xerr=[lo, hi], color="#4C72B0", capsize=2)
        ax.set_yticks(y, names, fontsize=7)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("time (s, log scale; median, 10th-90th percentile)", fontsize=8)
        ax.set_title(table["title"], fontsize=9)
    for ax in axes[len(tables) :]:
        ax.axis("off")
    fig.tight_layout()
    path = Path(out_dir) / f"section_{name}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    log(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sections", nargs="+", default=list("ABCD"), choices=list("ABCD")
    )
    parser.add_argument("--device", default="cuda", help="cuda (default) or cpu")
    parser.add_argument("--quick", action="store_true", help="toy sizes, seconds")
    parser.add_argument(
        "--pin-alone",
        action="store_true",
        help="section A: also time the variants that pin inside the call alone",
    )
    parser.add_argument("--json", default=None, help="write the results here")
    parser.add_argument(
        "--plot", default=None, help="write one figure per section here"
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="parent of the temporary directory for section C (default: system temp)",
    )
    parser.add_argument(
        "--memmap-mb",
        type=float,
        default=None,
        help="MB that section C writes to disk (default: 512, or 32 with --quick)",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --device cpu for a smoke test")
    if args.plot and not _has_matplotlib:
        parser.error("--plot requires matplotlib (pip install matplotlib)")
    memmap_mb = args.memmap_mb or (32 if args.quick else 512)

    info = env_info(device)
    print("# TensorDict vs hand-written PyTorch code\n")
    for key, value in info.items():
        print(f"- {key}: {value}")
    print(f"- quick: {args.quick}")
    if "C" in args.sections:
        n, n_files = dataset_sizes_c(memmap_mb)
        written = (2 * n + n_files) * SAMPLE_BYTES_C / MiB
        print(f"- section C writes about {written:.0f} MiB to disk")

    data_dir = tempfile.mkdtemp(prefix="structured_batch_bench_", dir=args.data_dir)
    results = {}
    try:
        for name in args.sections:
            t0 = time.perf_counter()
            log(f"section {name}")
            if name == "A":
                results[name] = section_a(device, args.quick, args.pin_alone)
            elif name == "B":
                results[name] = section_b(device, args.quick)
            elif name == "C":
                results[name] = section_c(device, args.quick, data_dir, memmap_mb)
            else:
                results[name] = section_d(device, args.quick, data_dir)
            log(f"section {name} done in {time.perf_counter() - t0:.1f}s")
            print_section(name, results[name])
            if args.plot:
                Path(args.plot).mkdir(parents=True, exist_ok=True)
                plot(name, results[name], args.plot)
    finally:
        shutil.rmtree(data_dir, ignore_errors=True)
    if args.json:
        payload = {"env": info, "args": vars(args), "sections": results}
        Path(args.json).write_text(json.dumps(payload, indent=1))
        log(f"wrote {args.json}")


if __name__ == "__main__":
    main()
