"""
Benchmarking TensorDict serialization speed
===========================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

In this example you will learn how the main TensorDict serialization paths
compare in terms of speed, and how ``num_threads`` affects each of them.

The script runs each serialization method several times, averages the
timings and renders them as a bar plot. It is executed during the
documentation build, so the figure below reflects the machine that built
these docs; download the script at the bottom of this page (or run
``python tutorials/sphinx_tuto/serialization_speed.py`` from a tensordict
checkout) to measure your own hardware. The only extra dependency is
``matplotlib``.

The methods compared are:

- :meth:`~tensordict.TensorDictBase.consolidate`: fuse all leaves into a
  single in-memory storage;
- :meth:`~tensordict.TensorDictBase.consolidate` with a ``filename``: fuse
  all leaves into a single memory-mapped file;
- :meth:`~tensordict.TensorDictBase.save`: write one memory-mapped file per
  leaf in a directory tree.

Each method is measured single-threaded and multithreaded, on two layouts
of the same kind of payload: a few large leaves and many small leaves.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import torch

from tensordict import TensorDict

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
N_REPEATS = 7
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


def timed(fn):
    """Runs ``fn`` once as warmup, then ``N_REPEATS`` times.

    Returns the median and the (lower, upper) distances to the 25th/75th
    percentiles, in milliseconds.
    """
    fn()
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    q25, median, q75 = np.percentile(times, [25, 50, 75])
    return median, (median - q25, q75 - median)


##############################################################################
# The serialization methods
# -------------------------
# Files and directories are created fresh for every run (a new name each
# time), so that we measure cold writes rather than in-place overwrites.


def make_methods(td, tmpdir):
    counter = iter(range(1_000_000))

    def consolidate(num_threads):
        td.consolidate(num_threads=num_threads)

    def consolidate_to_file(num_threads):
        td.consolidate(
            filename=Path(tmpdir) / f"consolidate{next(counter)}.td",
            num_threads=num_threads,
        )

    def save(num_threads):
        td.save(Path(tmpdir) / f"save{next(counter)}", num_threads=num_threads)

    return {
        "consolidate\n(memory)": consolidate,
        "consolidate\n(file)": consolidate_to_file,
        "save\n(directory)": save,
    }


##############################################################################
# Running the benchmark
# ---------------------

results = {}
for layout_name, td in LAYOUTS.items():
    with tempfile.TemporaryDirectory() as tmpdir:
        for method_name, method in make_methods(td, tmpdir).items():
            for num_threads in (0, NUM_THREADS):
                results[layout_name, method_name, num_threads] = timed(
                    lambda method=method, num_threads=num_threads: method(num_threads)
                )

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
# Further reading
# ---------------
#
# - The :ref:`saving documentation <saving>` covers the full
#   serialization API, including ``memmap_like`` and ``return_early``.
# - The benchmark suite in ``benchmarks/common/memmap_benchmarks_test.py``
#   tracks these operations with ``pytest-benchmark``.
