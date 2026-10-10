#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Round the results in pytest-benchmark JSON files before they go to gh-pages.

benchmark-action/github-action-benchmark writes ``stats.ops``,
``stats.stddev`` and ``stats.mean`` of each benchmark to
``dev/bench/data.js`` with up to 17 significant digits. GitHub push
protection reads a 15-digit number that starts with 1000, such as the
fractional part of ``32.100025873592784``, as a secret of type "BII - UID",
and rejects the push to gh-pages. This script rounds the three values so that
the action writes them with at most 12 significant digits, and leaves the other
fields alone. It writes the files back without indentation, which about halves
their size.

Usage::

    python .github/scripts/round_benchmark_results.py output.json [output.json ...]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

SIGNIFICANT_DIGITS = 12
# How many units of the last digit round_mean moves its target at most.
_MAX_TARGET_OFFSET = 100


def mean_scale(seconds: float) -> float:
    """Return the factor that the action multiplies a mean in seconds by.

    Mirrors ``getHumanReadableUnitValue`` in the action's ``src/extract.ts``,
    which turns the mean into nsec, usec, msec or sec.
    """
    if seconds < 1.0e-6:
        return 1e9
    if seconds < 1.0e-3:
        return 1e6
    if seconds < 1.0:
        return 1e3
    return 1.0


def significant_digits(value: float) -> int:
    """Return the number of significant digits in the shortest repr of ``value``.

    JavaScript writes a number with the same digits as Python's ``repr``.
    """
    mantissa = repr(value).partition("e")[0]
    return len(mantissa.lstrip("-").replace(".", "").strip("0"))


def round_value(value: float) -> float:
    """Round ``value`` to :data:`SIGNIFICANT_DIGITS` significant digits."""
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def round_mean(seconds: float) -> float:
    """Return a mean close to ``seconds`` that the action writes with few digits.

    The action writes ``seconds * mean_scale(seconds)``, and the product of a
    rounded mean and the scale can have 17 significant digits again. So this
    tries the means next to the rounded target and, if none of them gives a
    short product, next to the targets one or more units of the last digit
    away.
    """
    scale = mean_scale(seconds)
    mantissa, _, exponent = f"{seconds * scale:.{SIGNIFICANT_DIGITS - 1}e}".partition(
        "e"
    )
    digits = int(mantissa.replace(".", ""))
    power = int(exponent) - SIGNIFICANT_DIGITS + 1
    for offset in sorted(range(-_MAX_TARGET_OFFSET, _MAX_TARGET_OFFSET + 1), key=abs):
        candidate = float(f"{digits + offset}e{power}") / scale
        for mean in (
            candidate,
            math.nextafter(candidate, math.inf),
            math.nextafter(candidate, -math.inf),
        ):
            if significant_digits(mean * mean_scale(mean)) <= SIGNIFICANT_DIGITS:
                return mean
    return seconds


def round_benchmark_results(results: dict) -> None:
    """Round the values the action reads from each benchmark in ``results``."""
    for benchmark in results["benchmarks"]:
        stats = benchmark["stats"]
        stats["ops"] = round_value(stats["ops"])
        stats["stddev"] = round_value(stats["stddev"])
        stats["mean"] = round_mean(stats["mean"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="pytest-benchmark JSON")
    for path in parser.parse_args().paths:
        with path.open() as file:
            results = json.load(file)
        round_benchmark_results(results)
        with path.open("w") as file:
            json.dump(results, file)


if __name__ == "__main__":
    main()
