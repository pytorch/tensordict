#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Analyze per-test results from recent CI runs to identify flaky tests.

This script:
1. Fetches workflow run data via the GitHub API
2. Downloads JUnit XML test-result artifacts from each run
3. Parses per-test pass/fail outcomes from the XML
4. Aggregates statistics across runs
5. Identifies flaky tests based on intermittent failure patterns
6. Generates JSON and Markdown reports

Requires that CI jobs produce JUnit XML (via ``pytest --junitxml``)
and upload it as an artifact through the ``upload-artifact`` parameter
of the pytorch/test-infra reusable workflow.
"""

import argparse
import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

FLAKY_THRESHOLD_MIN = 0.05  # 5%
FLAKY_THRESHOLD_MAX = 0.80  # 80%
MIN_FAILURES_FOR_FLAKY = 2
MIN_EXECUTIONS = 3
NEW_FLAKY_DAYS = 7


# =============================================================================
# GitHub API / CLI helpers
# =============================================================================


def gh_api(endpoint: str) -> dict | list | None:
    """Call the GitHub API via ``gh api`` and return parsed JSON."""
    try:
        result = subprocess.run(
            ["gh", "api", endpoint],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)
    except subprocess.CalledProcessError as exc:
        print(f"Warning: gh api failed for {endpoint}: {exc.stderr.strip()}")
        return None
    except json.JSONDecodeError:
        return None


def gh_run_download(repo: str, run_id: int, pattern: str, dest: str) -> bool:
    """Download artifacts from a workflow run matching *pattern*."""
    try:
        subprocess.run(
            [
                "gh",
                "run",
                "download",
                str(run_id),
                "--repo",
                repo,
                "--pattern",
                pattern,
                "--dir",
                dest,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_repo() -> str:
    return os.environ.get("GITHUB_REPOSITORY", "pytorch/tensordict")


# =============================================================================
# Data collection
# =============================================================================


def list_workflow_runs(
    repo: str, workflow_name: str, branch: str, num_runs: int
) -> list[dict]:
    runs: list[dict] = []
    page = 1
    per_page = min(100, num_runs)

    while len(runs) < num_runs:
        endpoint = (
            f"/repos/{repo}/actions/workflows/{workflow_name}/runs"
            f"?branch={branch}&status=completed&per_page={per_page}&page={page}"
        )
        data = gh_api(endpoint)
        if not data or "workflow_runs" not in data:
            break
        batch = data["workflow_runs"]
        if not batch:
            break
        runs.extend(batch)
        page += 1
        if len(batch) < per_page:
            break

    return runs[:num_runs]


def parse_junit_xml(xml_path: Path) -> list[dict]:
    """Parse a JUnit XML file and return a list of per-test records."""
    tests: list[dict] = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        print(f"Warning: could not parse {xml_path}: {exc}")
        return tests

    root = tree.getroot()
    testcases = root.iter("testcase")

    for tc in testcases:
        classname = tc.get("classname", "")
        name = tc.get("name", "")
        if not name:
            continue

        nodeid = f"{classname}::{name}" if classname else name
        duration = float(tc.get("time", "0") or "0")

        failure = tc.find("failure")
        error = tc.find("error")
        skipped = tc.find("skipped")

        if failure is not None:
            outcome = "failed"
        elif error is not None:
            outcome = "error"
        elif skipped is not None:
            outcome = "skipped"
        else:
            outcome = "passed"

        tests.append(
            {
                "nodeid": nodeid,
                "outcome": outcome,
                "duration": duration,
            }
        )

    return tests


def collect_test_data(
    repo: str, workflow_name: str, num_runs: int
) -> tuple[list[dict], dict]:
    """Download artifacts and parse JUnit XML from recent runs."""
    print(f"Fetching last {num_runs} runs of {workflow_name} on main...")
    runs = list_workflow_runs(repo, workflow_name, "main", num_runs)
    print(f"  Found {len(runs)} completed runs")

    all_tests: list[dict] = []
    run_metadata: dict = {}

    for run in runs:
        run_id = run["id"]
        run_date = run["created_at"]
        commit_sha = run["head_sha"]

        with tempfile.TemporaryDirectory() as tmpdir:
            ok = gh_run_download(repo, run_id, "test-results-*", tmpdir)
            if not ok:
                run_metadata[run_id] = {
                    "date": run_date,
                    "sha": commit_sha,
                    "conclusion": run["conclusion"],
                }
                continue

            xml_files = list(Path(tmpdir).rglob("*.xml"))
            for xml_file in xml_files:
                artifact_dir = xml_file.parent.name
                tests = parse_junit_xml(xml_file)
                for t in tests:
                    t["_run_id"] = run_id
                    t["_run_date"] = run_date
                    t["_commit_sha"] = commit_sha
                    t["_artifact"] = artifact_dir
                    t["_xml_file"] = xml_file.name
                all_tests.extend(tests)

        run_metadata[run_id] = {
            "date": run_date,
            "sha": commit_sha,
            "conclusion": run["conclusion"],
        }

    print(
        f"  Collected {len(all_tests)} test records from {len(run_metadata)} runs"
    )
    return all_tests, run_metadata


# =============================================================================
# Analysis
# =============================================================================


def aggregate_test_stats(tests: list[dict]) -> dict[str, dict]:
    """Aggregate statistics per test nodeid across all runs."""
    stats_map: dict[str, dict] = defaultdict(
        lambda: {
            "executions": 0,
            "passed": 0,
            "failed": 0,
            "error": 0,
            "skipped": 0,
            "total_duration": 0.0,
            "failure_dates": [],
            "run_ids": set(),
        }
    )

    for t in tests:
        nodeid = t["nodeid"]
        outcome = t["outcome"]
        if outcome == "skipped":
            continue

        stats = stats_map[nodeid]
        stats["executions"] += 1
        stats["run_ids"].add(t["_run_id"])
        stats["total_duration"] += t.get("duration", 0.0)

        if outcome == "passed":
            stats["passed"] += 1
        elif outcome == "failed":
            stats["failed"] += 1
            if t.get("_run_date"):
                stats["failure_dates"].append(t["_run_date"])
        elif outcome == "error":
            stats["error"] += 1
            if t.get("_run_date"):
                stats["failure_dates"].append(t["_run_date"])

    for _nodeid, s in stats_map.items():
        s["run_ids"] = list(s["run_ids"])

    return dict(stats_map)


def calculate_flaky_score(stats: dict) -> float:
    if stats["executions"] < MIN_EXECUTIONS:
        return 0.0

    total_failures = stats["failed"] + stats["error"]
    failure_rate = total_failures / stats["executions"]

    if failure_rate >= FLAKY_THRESHOLD_MAX or failure_rate <= FLAKY_THRESHOLD_MIN:
        return 0.0

    if failure_rate <= 0.5:
        base_score = failure_rate * 2
    else:
        base_score = (1 - failure_rate) * 2

    confidence = min(1.0, total_failures / 5)
    return min(1.0, base_score * confidence)


def identify_flaky_tests(test_stats: dict[str, dict]) -> list[dict]:
    flaky: list[dict] = []

    for nodeid, stats in test_stats.items():
        if stats["executions"] < MIN_EXECUTIONS:
            continue

        total_failures = stats["failed"] + stats["error"]
        if total_failures < MIN_FAILURES_FOR_FLAKY:
            continue

        failure_rate = total_failures / stats["executions"]
        if failure_rate <= FLAKY_THRESHOLD_MIN or failure_rate >= FLAKY_THRESHOLD_MAX:
            continue

        flaky_score = calculate_flaky_score(stats)
        if flaky_score <= 0:
            continue

        first_failure = None
        is_new = False
        if stats["failure_dates"]:
            first_failure = sorted(stats["failure_dates"])[0]
            try:
                first_dt = datetime.fromisoformat(first_failure.replace("Z", "+00:00"))
                cutoff = datetime.now(timezone.utc) - timedelta(days=NEW_FLAKY_DAYS)
                is_new = first_dt > cutoff
            except ValueError:
                pass

        avg_duration = (
            stats["total_duration"] / stats["executions"]
            if stats["executions"]
            else 0
        )

        flaky.append(
            {
                "nodeid": nodeid,
                "executions": stats["executions"],
                "failures": total_failures,
                "failure_rate": round(failure_rate, 4),
                "flaky_score": round(flaky_score, 4),
                "passed": stats["passed"],
                "failed": stats["failed"],
                "error": stats["error"],
                "avg_duration_s": round(avg_duration, 3),
                "recent_failures": (
                    sorted(stats["failure_dates"])[-5:]
                    if stats["failure_dates"]
                    else []
                ),
                "first_seen_flaky": first_failure,
                "is_new": is_new,
            }
        )

    flaky.sort(key=lambda x: x["flaky_score"], reverse=True)
    return flaky


# =============================================================================
# Report generation
# =============================================================================


def generate_json_report(
    flaky_tests: list[dict],
    test_stats: dict[str, dict],
    run_metadata: dict,
    output_path: Path,
) -> dict:
    now = datetime.now(timezone.utc)

    if run_metadata:
        dates = [v["date"] for v in run_metadata.values()]
        start_date = min(dates)[:10] if dates else now.strftime("%Y-%m-%d")
        end_date = max(dates)[:10] if dates else now.strftime("%Y-%m-%d")
    else:
        start_date = end_date = now.strftime("%Y-%m-%d")

    new_flaky_count = sum(1 for t in flaky_tests if t.get("is_new"))

    report = {
        "generated_at": now.isoformat(),
        "analysis_period": {
            "start": start_date,
            "end": end_date,
            "runs_analyzed": len(run_metadata),
        },
        "summary": {
            "total_tests": len(test_stats),
            "flaky_count": len(flaky_tests),
            "new_flaky_count": new_flaky_count,
            "resolved_count": 0,
        },
        "flaky_tests": flaky_tests,
        "thresholds": {
            "min_failure_rate": FLAKY_THRESHOLD_MIN,
            "max_failure_rate": FLAKY_THRESHOLD_MAX,
            "min_failures": MIN_FAILURES_FOR_FLAKY,
            "min_executions": MIN_EXECUTIONS,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def generate_markdown_report(report: dict, output_path: Path) -> None:
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary = report["summary"]
    flaky_tests = report["flaky_tests"]

    lines = [
        f"# Flaky Test Report - {now_str}",
        "",
        "## Summary",
        "",
        f"- **Flaky tests**: {summary['flaky_count']}",
        f"- **Newly flaky** (last 7 days): {summary['new_flaky_count']}",
        f"- **Total tests analyzed**: {summary['total_tests']}",
        f"- **CI runs analyzed**: {report['analysis_period']['runs_analyzed']}",
        "",
        "---",
        "",
    ]

    if flaky_tests:
        lines.extend(
            [
                "## Flaky Tests",
                "",
                "| Test | Failure Rate | Failures | Flaky Score | Last Failed |",
                "|------|--------------|----------|-------------|-------------|",
            ]
        )

        for test in flaky_tests[:30]:
            nodeid = test["nodeid"]
            if len(nodeid) > 80:
                nodeid = "..." + nodeid[-77:]

            rate_str = (
                f"{test['failure_rate'] * 100:.1f}%"
                f" ({test['failures']}/{test['executions']})"
            )
            score_str = f"{test['flaky_score']:.2f}"
            last_failed = (
                test["recent_failures"][-1][:10]
                if test["recent_failures"]
                else "N/A"
            )
            new_marker = " **NEW**" if test.get("is_new") else ""

            lines.append(
                f"| `{nodeid}`{new_marker} | {rate_str} | "
                f"{test['failures']} | {score_str} | {last_failed} |"
            )

        lines.extend(["", ""])

        if summary["new_flaky_count"] > 0:
            lines.extend(["### Newly Flaky", ""])
            for t in flaky_tests:
                if t.get("is_new"):
                    lines.append(f"- `{t['nodeid']}`")
            lines.append("")
    else:
        lines.extend(
            [
                "## No Flaky Tests Detected!",
                "",
                "All tests are passing consistently across recent CI runs.",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## Configuration",
            "",
            f"- Minimum failure rate: {report['thresholds']['min_failure_rate'] * 100:.0f}%",
            f"- Maximum failure rate: {report['thresholds']['max_failure_rate'] * 100:.0f}%",
            f"- Minimum failures required: {report['thresholds']['min_failures']}",
            f"- Minimum executions required: {report['thresholds']['min_executions']}",
            "",
            "---",
            "",
            f"*Generated at {report['generated_at']}*",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_badge_json(flaky_count: int, output_path: Path) -> None:
    if flaky_count == 0:
        color = "brightgreen"
    elif flaky_count <= 5:
        color = "yellow"
    elif flaky_count <= 10:
        color = "orange"
    else:
        color = "red"

    badge = {
        "schemaVersion": 1,
        "label": "flaky tests",
        "message": str(flaky_count),
        "color": color,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(badge, f, indent=2)


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Analyze flaky tests from JUnit XML artifacts"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=30,
        help="Number of runs to analyze per workflow",
    )
    parser.add_argument(
        "--workflows",
        default="test-linux.yml",
        help="Comma-separated list of workflow file names",
    )
    parser.add_argument(
        "--output-dir",
        default="flaky-reports",
        help="Output directory",
    )
    args = parser.parse_args()

    repo = get_repo()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflows = [w.strip() for w in args.workflows.split(",") if w.strip()]

    print(f"Analyzing flaky tests for {repo}")
    print(f"  Workflows: {', '.join(workflows)}")
    print(f"  Runs to analyze per workflow: {args.runs}")
    print()

    all_tests: list[dict] = []
    all_run_metadata: dict = {}

    for workflow in workflows:
        print(f"\n{'=' * 60}")
        print(f"Processing workflow: {workflow}")
        print("=" * 60)

        tests, run_metadata = collect_test_data(repo, workflow, args.runs)
        all_tests.extend(tests)
        all_run_metadata.update(run_metadata)

    if not all_tests:
        print(
            "\nNo test-level data collected (artifacts may not exist yet)."
            "\nGenerating empty report."
        )
        test_stats: dict[str, dict] = {}
        flaky_tests: list[dict] = []
    else:
        print("\n" + "=" * 60)
        print("Aggregating per-test statistics...")
        print("=" * 60)
        test_stats = aggregate_test_stats(all_tests)
        print(f"  Analyzed {len(test_stats)} unique tests")

        print("  Identifying flaky tests...")
        flaky_tests = identify_flaky_tests(test_stats)
        print(f"  Found {len(flaky_tests)} flaky tests")

    print("\nGenerating reports...")

    json_report = generate_json_report(
        flaky_tests, test_stats, all_run_metadata, output_dir / "flaky-tests.json"
    )

    json_report["workflows_analyzed"] = workflows
    with open(output_dir / "flaky-tests.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    generate_markdown_report(json_report, output_dir / "flaky-tests.md")
    generate_badge_json(len(flaky_tests), output_dir / "badge.json")

    print(f"\nReports written to {output_dir}/")
    print("  - flaky-tests.json")
    print("  - flaky-tests.md")
    print("  - badge.json")

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"flaky_count={len(flaky_tests)}\n")
            f.write(f"new_flaky_count={json_report['summary']['new_flaky_count']}\n")
            f.write(f"resolved_count={json_report['summary']['resolved_count']}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
