# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).parents[1] / ".github" / "scripts" / "analyze_flaky_tests.py"
)
_SPEC = importlib.util.spec_from_file_location("analyze_flaky_tests", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
analyze_flaky_tests = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(analyze_flaky_tests)


def test_flaky_detection_requires_same_revision_and_environment_evidence():
    tests = []
    for nodeid, outcome, run_id, date, sha, artifact, xml_file in [
        (
            "test_mod.py::test_regression",
            "failed",
            1,
            "2026-08-25T10:00:00Z",
            "regression",
            "test-results-cpu-3.10",
            "junit-tests.xml",
        ),
        (
            "test_mod.py::test_regression",
            "failed",
            2,
            "2026-08-25T11:00:00Z",
            "regression",
            "test-results-cpu-3.10",
            "junit-tests.xml",
        ),
        (
            "test_mod.py::test_regression",
            "passed",
            3,
            "2026-08-26T10:00:00Z",
            "fixed",
            "test-results-cpu-3.10",
            "junit-tests.xml",
        ),
        (
            "test_mod.py::test_environment_specific",
            "failed",
            4,
            "2026-08-26T10:00:00Z",
            "same-sha",
            "test-results-cpu-3.10",
            "junit-tests.xml",
        ),
        (
            "test_mod.py::test_environment_specific",
            "passed",
            4,
            "2026-08-26T10:00:00Z",
            "same-sha",
            "test-results-gpu-3.10",
            "junit-tests.xml",
        ),
        (
            "test_mod.py::test_confirmed[param]",
            "failed",
            5,
            "2026-08-26T10:00:00Z",
            "flaky-sha",
            "test-results-cpu-3.10",
            "junit-tests.xml",
        ),
        (
            "test_mod.py::test_confirmed[param]",
            "passed",
            5,
            "2026-08-26T10:05:00Z",
            "flaky-sha",
            "test-results-cpu-3.10",
            "junit-tests-rerun.xml",
        ),
        (
            "test_mod.py::test_stale",
            "failed",
            6,
            "2026-07-01T10:00:00Z",
            "stale-sha",
            "test-results-cpu-3.10",
            "junit-tests.xml",
        ),
        (
            "test_mod.py::test_stale",
            "passed",
            6,
            "2026-07-01T10:05:00Z",
            "stale-sha",
            "test-results-cpu-3.10",
            "junit-tests-rerun.xml",
        ),
    ]:
        tests.append(
            {
                "nodeid": nodeid,
                "outcome": outcome,
                "duration": 0.1,
                "_run_id": run_id,
                "_run_date": date,
                "_commit_sha": sha,
                "_workflow": "test-linux.yml",
                "_artifact": artifact,
                "_xml_file": xml_file,
            }
        )

    stats = analyze_flaky_tests.aggregate_test_stats(tests)
    flaky_tests = analyze_flaky_tests.identify_flaky_tests(
        stats, now=datetime(2026, 8, 27, tzinfo=timezone.utc)
    )

    assert [test["nodeid"] for test in flaky_tests] == [
        "test_mod.py::test_confirmed[param]"
    ]
    assert flaky_tests[0]["environments"] == ["test-linux.yml / test-results-cpu-3.10"]
    assert flaky_tests[0]["confirmed_revisions"] == [
        {
            "sha": "flaky-sha",
            "date": "2026-08-26T10:00:00+00:00",
            "environment": "test-linux.yml / test-results-cpu-3.10",
            "run_ids": [5],
        }
    ]


def test_report_counts_families_and_changes_from_previous_report(tmp_path):
    tests = []
    for parameter, run_id in [("first", 1), ("second", 2)]:
        for outcome, xml_file in [
            ("failed", "junit-tests.xml"),
            ("passed", "junit-tests-rerun.xml"),
        ]:
            tests.append(
                {
                    "nodeid": f"test_mod.py::test_current[{parameter}]",
                    "outcome": outcome,
                    "duration": 0.1,
                    "_run_id": run_id,
                    "_run_date": "2026-08-26T10:00:00Z",
                    "_commit_sha": "flaky-sha",
                    "_workflow": "test-linux.yml",
                    "_artifact": "test-results-cpu-3.10",
                    "_xml_file": xml_file,
                }
            )

    stats = analyze_flaky_tests.aggregate_test_stats(tests)
    flaky_tests = analyze_flaky_tests.identify_flaky_tests(
        stats, now=datetime(2026, 8, 27, tzinfo=timezone.utc)
    )
    report = analyze_flaky_tests.generate_json_report(
        flaky_tests,
        stats,
        {
            1: {
                "date": "2026-08-26T10:00:00Z",
                "sha": "flaky-sha",
                "conclusion": "failure",
            }
        },
        tmp_path / "flaky-tests.json",
        "pytorch/tensordict",
        previous_report={
            "flaky_tests": [{"nodeid": "test_mod.py::test_resolved[param]"}]
        },
        now=datetime(2026, 8, 27, tzinfo=timezone.utc),
    )

    assert report["summary"] == {
        "total_tests": 2,
        "flaky_count": 1,
        "flaky_case_count": 2,
        "new_flaky_count": 1,
        "resolved_count": 1,
    }
    assert report["resolved_test_families"] == ["test_mod.py::test_resolved"]
    assert all(test["is_new"] for test in report["flaky_tests"])
