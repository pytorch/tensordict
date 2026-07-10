"""
Benchmarking TensorDict serialization speed
===========================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

In this example you will learn how the main TensorDict serialization paths
compare in terms of speed, how ``num_threads`` affects each of them, and
how the on-disk formats compare with ``torch.save`` and safetensors.

The script runs each measurement 32 times and renders the median with the
interquartile range as error bars. It is executed during the documentation
build, so the figures below reflect the machine that built these docs;
download the script at the bottom of this page (or run
``python tutorials/sphinx_tuto/serialization_speed.py`` from a tensordict
checkout) to measure your own hardware. The only extra dependencies are
``matplotlib`` and, optionally, ``safetensors``.

The first benchmark compares the write paths:

- :meth:`~tensordict.TensorDictBase.consolidate`: fuse all leaves into a
  single in-memory storage;
- :meth:`~tensordict.TensorDictBase.consolidate` with a ``filename``: fuse
  all leaves into a single memory-mapped file;
- :meth:`~tensordict.TensorDictBase.save`: write one memory-mapped file per
  leaf in a directory tree.

Each method is measured single-threaded and multithreaded, on two layouts
of the same kind of payload: a few large leaves and many small leaves.
The second benchmark compares the on-disk formats (memmap directory,
consolidated file, ``.tdz`` archive, ``torch.save``, safetensors) on
saving, opening and copying the saved artifact.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import torch

from tensordict import TensorDict

_has_safetensors = importlib.util.find_spec("safetensors") is not None

##############################################################################
# Benchmark configuration
# -----------------------
# The default payload is 256 MiB per layout, small enough for a
# documentation build while large enough for threading effects to show;
# ``SCALE`` scales it linearly (e.g. ``SCALE=4`` for 1 GiB). Every
# measurement is repeated ``N_REPEATS`` times after one warmup run, and we
# report the median with the interquartile range as error bars: disk writes
# occasionally hiccup (page-cache writeback), which would dominate a mean.

SCALE = 1
N_REPEATS = 32
NUM_THREADS = min(8, os.cpu_count() or 1)

LAYOUTS = {
    # 8 leaves of 32 MiB each (per unit of SCALE)
    "8 large leaves": TensorDict(
        {f"key{i}": torch.randn(SCALE * 8 * 2**20) for i in range(8)}
    ),
    # 512 leaves of 512 KiB each (per unit of SCALE)
    "512 small leaves": TensorDict(
        {f"key{i}": torch.randn(SCALE * 2**17) for i in range(512)}
    ),
}


def summarize(times):
    """Returns the median and the (lower, upper) distances to the 25th/75th percentiles."""
    q25, median, q75 = np.percentile(times, [25, 50, 75])
    return median, (median - q25, q75 - median)


##############################################################################
# The serialization methods
# -------------------------
# Each method returns the path it wrote (or ``None`` for the in-memory
# variant) so that the benchmark loop can clean up between runs.


def make_methods(td, tmpdir):
    counter = iter(range(1_000_000))

    def consolidate(num_threads):
        td.consolidate(num_threads=num_threads)

    def consolidate_to_file(num_threads):
        filename = Path(tmpdir) / f"consolidate{next(counter)}.td"
        td.consolidate(filename=filename, num_threads=num_threads)
        return filename

    def save(num_threads):
        prefix = Path(tmpdir) / f"save{next(counter)}"
        td.save(prefix, num_threads=num_threads)
        return prefix

    return {
        "consolidate\n(memory)": consolidate,
        "consolidate\n(file)": consolidate_to_file,
        "save\n(directory)": save,
    }


def cleanup(path):
    """Removes a written artifact and flushes the writeback backlog.

    Runs outside the timed region: it keeps disk usage bounded and, more
    importantly, makes the runs independent. Without it, dirty pages from
    earlier runs accumulate and the operating system throttles later
    writes, which inflates both the timings and their spread.
    """
    if path is None:
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    if hasattr(os, "sync"):
        os.sync()


##############################################################################
# Running the benchmark
# ---------------------
# For each method, the single-threaded and multithreaded runs are
# interleaved rather than run in two separate batches: repeated writes make
# the filesystem progressively slower (page-cache fill, writeback backlog),
# and interleaving spreads that drift equally over both settings.

results = {}
for layout_name, td in LAYOUTS.items():
    with tempfile.TemporaryDirectory() as tmpdir:
        for method_name, method in make_methods(td, tmpdir).items():
            times = {0: [], NUM_THREADS: []}
            for num_threads in times:  # warmup
                cleanup(method(num_threads))
            for _ in range(N_REPEATS):
                for num_threads in times:
                    t0 = time.perf_counter()
                    path = method(num_threads)
                    times[num_threads].append((time.perf_counter() - t0) * 1000)
                    cleanup(path)
            for num_threads, values in times.items():
                results[layout_name, method_name, num_threads] = summarize(values)

##############################################################################
# Plotting
# --------
# One panel per layout; within each panel, one pair of bars per method
# (single-threaded vs. multithreaded), with the interquartile range as
# error bars.

fig, axes = plt.subplots(1, len(LAYOUTS), figsize=(5 * len(LAYOUTS), 4), sharey=False)
total_bytes = {
    name: sum(v.numel() * v.element_size() for v in td.values(True, True))
    for name, td in LAYOUTS.items()
}
for ax, layout_name in zip(axes, LAYOUTS):
    methods = [m for (l, m, n) in results if l == layout_name and n == 0]
    x = np.arange(len(methods))
    width = 0.38
    for offset, num_threads, label in (
        (-width / 2, 0, "single-threaded"),
        (width / 2, NUM_THREADS, f"num_threads={NUM_THREADS}"),
    ):
        medians = [results[layout_name, m, num_threads][0] for m in methods]
        errors = np.array([results[layout_name, m, num_threads][1] for m in methods]).T
        ax.bar(x + offset, medians, width, yerr=errors, capsize=3, label=label)
    ax.set_xticks(x, methods)
    ax.set_ylabel("time (ms), lower is better")
    size_mb = total_bytes[layout_name] / 2**20
    ax.set_title(f"{layout_name} ({size_mb:.0f} MiB)")
    ax.legend()
fig.suptitle(f"TensorDict serialization speed (median over {N_REPEATS} runs)")
fig.tight_layout()
plt.show()

##############################################################################
# Reading the results
# -------------------
# A few rules of thumb emerge (see the docstring of
# :meth:`~tensordict.TensorDictBase.consolidate` for details):
#
# - In-memory consolidation benefits the most from ``num_threads``: the
#   leaves are copied by contiguous chunks of roughly equal byte size, one
#   fused copy per thread.
# - When consolidating to a memory-mapped file, threads are only used when
#   the data needs a device change (e.g. CUDA to CPU): concurrent writes to
#   a fresh file mapping are slower than a single sequential copy on most
#   local filesystems, so ``num_threads`` is ignored in that case.
# - The per-leaf directory format (:meth:`~tensordict.TensorDictBase.save`)
#   pays a per-file cost, which dominates with many small leaves whatever
#   the thread count.
#
# Absolute numbers depend heavily on the filesystem and the state of the
# page cache: benchmark on your own storage before picking a strategy, and
# prefer larger payloads (``SCALE=4`` or more, i.e. 1 GiB+) for stable
# measurements.
#
# Choosing an on-disk format
# --------------------------
# The second benchmark compares the on-disk formats tensordict can write --
# the per-leaf memmap directory, the consolidated single file and the
# ``.tdz`` zip archive -- with ``torch.save`` and, when installed,
# safetensors. The two external formats are measured on their fastest
# paths: a flat dict of tensors, written with ``torch.save`` /
# ``save_file`` and reloaded with ``torch.load(mmap=True,
# weights_only=True)`` / the mmap-backed ``load_file``. They do not
# represent tensordict structure natively, so keys are flattened at save
# time and the nesting is rebuilt at load time.
#
# Three operations are timed. *Save* writes a fresh artifact (removed
# outside the timed region). *Open* constructs a lazy tensordict over an
# existing artifact: every format here memory-maps its payload, so this is
# a metadata and view-construction cost -- bulk read throughput is
# essentially identical across formats once open. *Copy* duplicates the
# saved artifact, which is where single files beat directories: a
# directory pays per-file latency, which grows with the number of leaves
# (and with round-trip time on network filesystems).


def make_formats(td, tmpdir):
    """Maps format name -> (save() -> path, open(path)) for one layout."""
    base = Path(tmpdir)
    counter = iter(range(1_000_000))

    def fresh(suffix):
        return base / f"artifact{next(counter)}{suffix}"

    def save_dir():
        prefix = fresh("")
        td.save(prefix)
        return prefix

    def open_dir(path):
        TensorDict.load_memmap(path)

    def save_consolidated():
        filename = fresh(".td")
        td.consolidate(filename=filename)
        return filename

    def open_consolidated(path):
        TensorDict.from_consolidated(path)

    def save_archive():
        filename = fresh(".tdz")
        td.save(filename)
        return filename

    def save_pt():
        filename = fresh(".pt")
        torch.save(dict(td.flatten_keys(".").items()), filename)
        return filename

    def open_pt(path):
        flat = torch.load(path, mmap=True, weights_only=True)
        TensorDict(flat, batch_size=[]).unflatten_keys(".")

    formats = {
        "memmap\ndirectory": (save_dir, open_dir),
        "consolidated\nfile": (save_consolidated, open_consolidated),
        "archive\n(.tdz)": (save_archive, open_dir),
        "torch.save": (save_pt, open_pt),
    }
    if _has_safetensors:
        from safetensors.torch import load_file, save_file

        def save_safetensors():
            filename = fresh(".safetensors")
            save_file(dict(td.flatten_keys(".").items()), filename)
            return filename

        def open_safetensors(path):
            TensorDict(load_file(path), batch_size=[]).unflatten_keys(".")

        formats["safetensors"] = (save_safetensors, open_safetensors)
    return formats


def copy_artifact(path, dst):
    if path.is_dir():
        shutil.copytree(path, dst)
    else:
        shutil.copyfile(path, dst)


def bench_format(save, open_):
    """Times save / open / copy for one format on one layout."""
    out = {}
    cleanup(save())  # warmup
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        path = save()
        times.append((time.perf_counter() - t0) * 1000)
        cleanup(path)
    out["save"] = summarize(times)

    # open and copy operate on a single saved artifact
    artifact = save()
    open_(artifact)  # warmup
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        open_(artifact)
        times.append((time.perf_counter() - t0) * 1000)
    out["open"] = summarize(times)

    dst = artifact.parent / f"copy_{artifact.name}"
    copy_artifact(artifact, dst)  # warmup
    cleanup(dst)
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        copy_artifact(artifact, dst)
        times.append((time.perf_counter() - t0) * 1000)
        cleanup(dst)
    out["copy"] = summarize(times)
    cleanup(artifact)
    return out


format_results = {}
for layout_name, td in LAYOUTS.items():
    with tempfile.TemporaryDirectory() as tmpdir:
        for format_name, (save, open_) in make_formats(td, tmpdir).items():
            format_results[layout_name, format_name] = bench_format(save, open_)

##############################################################################
# One row per operation, one panel per layout, one bar per format. The time
# axis is logarithmic: opening is orders of magnitude faster than saving,
# and the formats differ by orders of magnitude on copying.

OPERATIONS = ("save", "open", "copy")
fig, axes = plt.subplots(
    len(OPERATIONS),
    len(LAYOUTS),
    figsize=(5 * len(LAYOUTS), 3.2 * len(OPERATIONS)),
    sharey="row",
)
for i, operation in enumerate(OPERATIONS):
    for j, layout_name in enumerate(LAYOUTS):
        ax = axes[i, j]
        names = [f for (l, f) in format_results if l == layout_name]
        medians = [format_results[layout_name, f][operation][0] for f in names]
        errors = np.array(
            [format_results[layout_name, f][operation][1] for f in names]
        ).T
        x = np.arange(len(names))
        ax.bar(
            x,
            medians,
            0.6,
            yerr=errors,
            capsize=3,
            color=plt.cm.tab10.colors[: len(names)],
        )
        ax.set_yscale("log")
        ax.set_xticks(x, names, fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{operation} time (ms)\nlower is better")
        if i == 0:
            size_mb = total_bytes[layout_name] / 2**20
            ax.set_title(f"{layout_name} ({size_mb:.0f} MiB)")
fig.suptitle(f"On-disk format comparison (median over {N_REPEATS} runs, log scale)")
fig.tight_layout()
plt.show()

##############################################################################
# The single-file formats (consolidated, archive, ``torch.save``,
# safetensors) open with a single metadata parse and copy at raw disk
# bandwidth, while the directory pays a per-file cost on both operations,
# which dominates with many small leaves. Saving is bandwidth-bound for
# every format, with per-entry overheads showing in the many-small-leaves
# layout. Keep in mind that the tensordict formats carry nested structure,
# lazy stacks, non-tensor data and (for directories and archives)
# in-place writability, which the flat external formats do not.
#
# Further reading
# ---------------
#
# - The :ref:`saving documentation <saving>` covers the full
#   serialization API, including ``memmap_like`` and ``return_early``.
# - The benchmark suite in ``benchmarks/common/memmap_benchmarks_test.py``
#   tracks these operations with ``pytest-benchmark``, and
#   ``benchmarks/scripts/serialization_formats_bench.py`` runs a larger
#   offline version of the format comparison (bigger payloads, more
#   layouts, read timings).
