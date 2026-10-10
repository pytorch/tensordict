# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).parents[1] / ".github" / "scripts" / "round_benchmark_results.py"
)
_SPEC = importlib.util.spec_from_file_location("round_benchmark_results", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
round_benchmark_results = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(round_benchmark_results)


def _benchmark(ops, stddev, mean):
    return {
        "group": None,
        "name": "test_x[cpu]",
        "fullname": "benchmarks/test_x.py::test_x[cpu]",
        "params": {"device": "cpu"},
        "param": "cpu",
        "extra_info": {},
        "options": {"min_rounds": 5},
        "stats": {
            "min": mean,
            "max": mean,
            "mean": mean,
            "stddev": stddev,
            "rounds": 3,
            "median": mean,
            "ops": ops,
            "total": 3 * mean,
            "data": [mean, mean, mean],
            "iterations": 1,
        },
    }


@pytest.mark.parametrize(
    "seconds",
    [
        # Means whose scaled values github-action-benchmark wrote to data.js
        # in the gh-pages pushes that push protection rejected as "BII - UID".
        32.100025873592784e-6,
        4.100040912552448e-3,
        2.100055425615008e-6,
        44.100054312249725e-6,
        17.100025620650484e-6,
        4.100073413136166e-6,
        13.100046266484798e-6,
        # The unit thresholds of getHumanReadableUnitValue.
        1e-6,
        1e-3,
        1.0,
        9.9999999999999e-4,
        0.0,
    ],
)
def test_round_mean_shortens_the_scaled_mean(seconds):
    mean = round_benchmark_results.round_mean(seconds)
    scaled = mean * round_benchmark_results.mean_scale(mean)
    assert round_benchmark_results.significant_digits(scaled) <= 12
    assert mean == pytest.approx(seconds, rel=1e-10, abs=0.0)


def test_round_mean_shortens_random_means():
    rng = random.Random(0)
    for _ in range(10_000):
        seconds = 10 ** rng.uniform(-10, 2)
        mean = round_benchmark_results.round_mean(seconds)
        scaled = mean * round_benchmark_results.mean_scale(mean)
        assert round_benchmark_results.significant_digits(scaled) <= 12, seconds
        assert mean == pytest.approx(seconds, rel=1e-10, abs=0.0)


def test_main_keeps_only_the_values_the_action_reads_rounded(tmp_path):
    # The value that push protection rejected for commit ea27b1c2a.
    ops = 10.100043678007761
    stddev = 0.000015181885534602878
    mean = 32.100025873592784e-6
    results = {
        "machine_info": {"node": "runner"},
        "commit_info": {"id": "ea27b1c2a43772d5277011a15da24a19781e102d"},
        "benchmarks": [_benchmark(ops, stddev, mean)],
        "datetime": "2026-10-10T13:44:14+00:00",
        "version": "5.1.0",
    }
    path = tmp_path / "output.json"
    path.write_text(json.dumps(results))

    subprocess.run([sys.executable, str(_SCRIPT_PATH), str(path)], check=True)

    rounded = json.loads(path.read_text())
    assert list(rounded) == ["benchmarks"]
    (benchmark,) = rounded["benchmarks"]
    assert list(benchmark) == ["fullname", "stats"]
    assert benchmark["fullname"] == "benchmarks/test_x.py::test_x[cpu]"
    stats = benchmark["stats"]
    assert list(stats) == ["ops", "stddev", "mean", "rounds"]
    assert repr(stats["ops"]) == "10.100043678"
    assert repr(stats["stddev"]) == "1.51818855346e-05"
    assert repr(stats["mean"] * 1e6) == "32.1000258736"
    assert stats["rounds"] == 3
