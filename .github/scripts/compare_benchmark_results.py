#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Compare two pytest-benchmark JSON files and write a markdown summary.

The PR benchmark workflow runs the benchmarks once on the base commit and once
on the head of the pull request, and appends the output of this script to
``$GITHUB_STEP_SUMMARY``. It compares the median time of each benchmark, which
outlier rounds move less than the mean, and lists only the benchmarks whose
median changed by :data:`THRESHOLD` or more, the slower ones and the faster
ones in separate tables.

Usage::

    python .github/scripts/compare_benchmark_results.py base.json pr.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# The smallest relative change of the median that the summary lists. On a PR
# that changed no code, 2-3% of the benchmarks moved by 20% or more.
THRESHOLD = 0.25


def medians(results: dict) -> dict[str, float]:
    """Return the median time in seconds of each benchmark, by full name."""
    return {
        benchmark["fullname"]: benchmark["stats"]["median"]
        for benchmark in results["benchmarks"]
    }


def format_time(seconds: float) -> str:
    """Return ``seconds`` with three significant digits and a unit."""
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("us", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def _table(rows: list[tuple[str, float, float, float]]) -> list[str]:
    lines = ["| Benchmark | Base median | PR median | Change |", "|---|---:|---:|---:|"]
    for name, base, pr, change in rows:
        name = name.replace("|", "\\|")
        lines.append(
            f"| `{name}` | {format_time(base)} | {format_time(pr)} | {change:+.0%} |"
        )
    return lines


def compare(base_results: dict, pr_results: dict) -> str:
    """Return the markdown summary of the PR run against the base run."""
    base = medians(base_results)
    pr = medians(pr_results)
    common = [name for name in pr if name in base]
    changes = [
        (name, base[name], pr[name], pr[name] / base[name] - 1)
        for name in common
        if base[name] > 0
    ]
    slower = sorted(
        (row for row in changes if row[3] >= THRESHOLD), key=lambda row: -row[3]
    )
    faster = sorted(
        (row for row in changes if row[3] <= -THRESHOLD), key=lambda row: row[3]
    )

    lines = [
        f"Compared the median time of {len(common)} benchmarks; "
        f"{len(pr) - len(common)} new and {len(base) - len(common)} removed ones "
        f"are not compared. Listed: changes of {THRESHOLD:.0%} or more.",
        "",
        "Base and PR ran once each, one after the other, on a shared runner, so "
        "a single listed benchmark can be noise: rerun it before you trust it.",
    ]
    for title, rows in (("Slower", slower), ("Faster", faster)):
        lines += ["", f"### {title} ({len(rows)})", ""]
        lines += _table(rows) if rows else ["None."]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("base", type=Path, help="pytest-benchmark JSON of the base")
    parser.add_argument("pr", type=Path, help="pytest-benchmark JSON of the PR")
    args = parser.parse_args()
    with args.base.open() as file:
        base_results = json.load(file)
    with args.pr.open() as file:
        pr_results = json.load(file)
    print(compare(base_results, pr_results), end="")


if __name__ == "__main__":
    main()
