# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compares the on-disk tensordict representations: memmap directory, consolidated file and zip archive.

Measures, per (layout, total size, format):

- save:  in-memory tensordict -> disk
- open:  disk -> loaded tensordict (lazy: metadata + view construction)
- read:  ``sum()`` over every leaf of a freshly opened tensordict (forces
  page-in; warm page cache since the files were just written)
- copy:  duplicating the on-disk artifact (``cp -R`` equivalent)

Usage::

    python benchmarks/scripts/serialization_formats_bench.py [--plot out.png]

The ``--plot`` option renders the figure embedded in
``docs/source/saving.rst`` (requires matplotlib).
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

from tensordict import pack_memmap, TensorDict

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None

MB = 1024 * 1024

FORMATS = ["memmap dir", "consolidated", "archive (tdz)"]
COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def make_flat_large(total_bytes):
    n = total_bytes // 4 // 8
    return TensorDict({f"w{i}": torch.randn(n) for i in range(8)}, batch_size=[])


def make_many_small(total_bytes):
    n = total_bytes // 4 // 2000
    return TensorDict(
        {
            f"g{i}": {f"t{j}": torch.randn(max(n, 1)) for j in range(20)}
            for i in range(100)
        },
        batch_size=[],
    )


def make_deep(total_bytes):
    depth, per_level = 12, 4
    n = total_bytes // 4 // (depth * per_level)
    td = leaf = TensorDict({}, batch_size=[])
    for d in range(depth):
        for j in range(per_level):
            leaf[f"t{j}"] = torch.randn(max(n, 1))
        if d < depth - 1:
            nxt = TensorDict({}, batch_size=[])
            leaf["sub"] = nxt
            leaf = nxt
    return td


LAYOUTS = {
    "flat\n8 leaves": make_flat_large,
    "many small\n2000 leaves": make_many_small,
    "deep\n12 levels": make_deep,
}
SIZES = [10 * MB, 100 * MB, 1024 * MB]
REPS = {10 * MB: 5, 100 * MB: 3, 1024 * MB: 2}


def timeit(fn, reps):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


def read_all(td):
    total = 0.0
    for value in td.values(True, True):
        if isinstance(value, torch.Tensor):
            total += value.sum().item()
    return total


def bench(root: Path):
    results = []
    for lname, factory in LAYOUTS.items():
        for size in SIZES:
            reps = REPS[size]
            td = factory(size)
            base = root / f"{lname.split(chr(10))[0].replace(' ', '_')}_{size}"
            base.mkdir()
            d_dir, d_cons, d_tdz = base / "dir", base / "c.mmap", base / "d.tdz"

            row = {"layout": lname, "size_mb": size // MB}
            row["save_dir"] = timeit(lambda td=td, d=d_dir: td.memmap(d), reps)
            row["save_cons"] = timeit(lambda td=td, d=d_cons: td.consolidate(d), reps)
            row["save_tdz"] = timeit(lambda td=td, d=d_tdz: td.save(d), reps)
            d_packed = base / "packed.tdz"

            def pack(d_dir=d_dir, d_packed=d_packed):
                d_packed.unlink(missing_ok=True)
                pack_memmap(d_dir, d_packed)

            row["pack_only"] = timeit(pack, reps)

            row["open_dir"] = timeit(
                lambda d=d_dir: TensorDict.load_memmap(d), reps * 2
            )
            row["open_cons"] = timeit(
                lambda d=d_cons: TensorDict.from_consolidated(d), reps * 2
            )
            row["open_tdz"] = timeit(
                lambda d=d_tdz: TensorDict.load_memmap(d), reps * 2
            )

            row["read_dir"] = timeit(
                lambda d=d_dir: read_all(TensorDict.load_memmap(d)), reps
            )
            row["read_cons"] = timeit(
                lambda d=d_cons: read_all(TensorDict.from_consolidated(d)), reps
            )
            row["read_tdz"] = timeit(
                lambda d=d_tdz: read_all(TensorDict.load_memmap(d)), reps
            )

            dst = base / "copy_target"

            def copy_dir(d_dir=d_dir, dst=dst):
                if dst.exists():
                    shutil.rmtree(dst)
                subprocess.run(["cp", "-R", str(d_dir), str(dst)], check=True)

            def copy_file(src, dst=dst):
                subprocess.run(["cp", str(src), str(dst / "f")], check=True)

            row["copy_dir"] = timeit(copy_dir, reps)
            dst.mkdir(exist_ok=True)
            row["copy_cons"] = timeit(lambda d=d_cons: copy_file(d), reps)
            row["copy_tdz"] = timeit(lambda d=d_tdz: copy_file(d), reps)

            results.append(row)
            print(f"done: {lname.split(chr(10))[0]} @ {size // MB}MB", file=sys.stderr)
            shutil.rmtree(base)
    return results


def human(t):
    if t < 1e-3:
        return f"{t * 1e6:.0f}us"
    if t < 1:
        return f"{t * 1e3:.0f}ms" if t >= 0.01 else f"{t * 1e3:.1f}ms"
    return f"{t:.2f}s"


def print_tables(results):
    for op in ("save", "open", "read", "copy"):
        print(f"\n=== {op} ===")
        header = f"{'layout':<26}{'MB':>6} | {'dir':>8} {'consol':>8} {'tdz':>8}"
        if op == "save":
            header += f" {'pack-only':>10}"
        print(header)
        for r in results:
            layout = r["layout"].replace("\n", " ")
            line = f"{layout:<26}{r['size_mb']:>6} | "
            line += (
                f"{human(r[f'{op}_dir']):>8} "
                f"{human(r[f'{op}_cons']):>8} "
                f"{human(r[f'{op}_tdz']):>8}"
            )
            if op == "save":
                line += f" {human(r['pack_only']):>10}"
            print(line)


def plot(results, out: str):
    if not _has_matplotlib:
        raise ImportError("Plotting requires matplotlib (pip install matplotlib).")
    import matplotlib.pyplot as plt
    import numpy as np

    by_key = {(r["layout"], r["size_mb"]): r for r in results}
    panels = {
        "save (1 GB)": ("save", 1024),
        "open, lazy (size-independent)": ("open", 1024),
        "open + read all (1 GB, warm cache)": ("read", 1024),
        "copy the saved artifact (100 MB)": ("copy", 100),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (title, (op, size_mb)) in zip(axes.flat, panels.items()):
        labels = list(LAYOUTS)
        values = np.array(
            [
                [
                    by_key[(lab, size_mb)][f"{op}_{fmt}"]
                    for fmt in ("dir", "cons", "tdz")
                ]
                for lab in labels
            ]
        )
        x = np.arange(len(labels))
        width = 0.26
        for i, (fmt_name, color) in enumerate(zip(FORMATS, COLORS)):
            bars = ax.bar(
                x + (i - 1) * width, values[:, i], width, label=fmt_name, color=color
            )
            for rect, v in zip(bars, values[:, i]):
                ax.annotate(
                    human(v),
                    xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylabel("time (s, log scale)", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.margins(y=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.flat[0].legend(fontsize=9, frameon=False, loc="upper left")
    fig.suptitle(
        "TensorDict serialization formats: memmap directory vs consolidated vs zip archive\n"
        "(lower is better; laptop SSD, warm page cache)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=160)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", default=None, help="path of the figure to render")
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="tdbench_"))
    try:
        results = bench(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print_tables(results)
    if args.plot:
        plot(results, args.plot)
