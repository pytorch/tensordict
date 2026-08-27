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
4. Aggregates statistics by test and CI environment
5. Identifies flaky tests from fail/pass evidence on the same revision
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
from datetime import datetime, timedelta, timezone
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

NEW_FLAKY_DAYS = 7
ACTIVE_FLAKY_DAYS = 14
FAILURE_OUTCOMES = {"failed", "error"}


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
            f"?branch={branch}&event=push&status=completed"
            f"&per_page={per_page}&page={page}"
        )
        data = gh_api(endpoint)
        if not data or "workflow_runs" not in data:
            break
        raw_batch = data["workflow_runs"]
        if not raw_batch:
            break
        # The API filters above are the primary boundary. Keep this fail-closed
        # check as defense in depth before downloading executable-repository
        # artifacts such as JUnit XML.
        batch = [
            run
            for run in raw_batch
            if run.get("event") == "push"
            and run.get("head_branch") == branch
            and (run.get("head_repository") or {}).get("full_name") == repo
        ]
        runs.extend(batch)
        page += 1
        if len(raw_batch) < per_page:
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
                continue

            xml_files = list(Path(tmpdir).rglob("*.xml"))
            for xml_file in xml_files:
                artifact_dir = xml_file.parent.name
                tests = parse_junit_xml(xml_file)
                for t in tests:
                    t["_run_id"] = run_id
                    t["_run_date"] = run_date
                    t["_commit_sha"] = commit_sha
                    t["_workflow"] = workflow_name
                    t["_artifact"] = artifact_dir
                    t["_xml_file"] = xml_file.name
                all_tests.extend(tests)

        if xml_files:
            run_metadata[run_id] = {
                "date": run_date,
                "sha": commit_sha,
                "conclusion": run["conclusion"],
            }

    print(f"  Collected {len(all_tests)} test records from {len(run_metadata)} runs")
    return all_tests, run_metadata


# =============================================================================
# Analysis
# =============================================================================


def aggregate_test_stats(tests: list[dict]) -> dict[tuple[str, str], dict]:
    """Aggregate attempts without mixing CI matrix environments."""
    stats_map: dict[tuple[str, str], dict] = {}

    for t in tests:
        nodeid = t["nodeid"]
        outcome = t["outcome"]
        if outcome == "skipped":
            continue

        environment = f"{t['_workflow']} / {t['_artifact']}"
        stats = stats_map.setdefault(
            (nodeid, environment),
            {
                "nodeid": nodeid,
                "environment": environment,
                "executions": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "total_duration": 0.0,
                "failure_dates": [],
                "attempts": [],
            },
        )
        stats["executions"] += 1
        stats["total_duration"] += t.get("duration", 0.0)
        stats["attempts"].append(
            {
                "outcome": outcome,
                "run_id": t["_run_id"],
                "date": t["_run_date"],
                "sha": t["_commit_sha"],
                "xml_file": t["_xml_file"],
            }
        )

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

    return stats_map


def parse_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def identify_flaky_tests(
    test_stats: dict[tuple[str, str], dict], now: datetime | None = None
) -> list[dict]:
    """Return tests with recent fail/pass evidence on one revision and environment."""
    if now is None:
        now = datetime.now(timezone.utc)
    active_cutoff = now - timedelta(days=ACTIVE_FLAKY_DAYS)
    flaky_by_nodeid: dict[str, dict] = {}

    for stats in test_stats.values():
        revisions: dict[str, dict] = {}
        for attempt in stats["attempts"]:
            revision = revisions.setdefault(
                attempt["sha"],
                {"attempts": [], "outcomes": set()},
            )
            revision["attempts"].append(attempt)
            revision["outcomes"].add(attempt["outcome"])

        confirmed_revisions = []
        for sha, revision in revisions.items():
            outcomes = revision["outcomes"]
            if "passed" not in outcomes or not outcomes.intersection(FAILURE_OUTCOMES):
                continue
            failure_attempts = [
                attempt
                for attempt in revision["attempts"]
                if attempt["outcome"] in FAILURE_OUTCOMES
            ]
            parsed_failure_dates = [
                parsed
                for attempt in failure_attempts
                if (parsed := parse_datetime(attempt["date"])) is not None
            ]
            if not parsed_failure_dates or max(parsed_failure_dates) < active_cutoff:
                continue
            confirmed_revisions.append(
                {
                    "sha": sha,
                    "date": max(parsed_failure_dates).isoformat(),
                    "environment": stats["environment"],
                    "run_ids": sorted(
                        {attempt["run_id"] for attempt in revision["attempts"]}
                    ),
                }
            )

        if not confirmed_revisions:
            continue

        nodeid = stats["nodeid"]
        flaky = flaky_by_nodeid.setdefault(
            nodeid,
            {
                "nodeid": nodeid,
                "family": nodeid.split("[", 1)[0],
                "executions": 0,
                "passed": 0,
                "failed": 0,
                "error": 0,
                "total_duration": 0.0,
                "recent_failures": [],
                "environments": [],
                "confirmed_revisions": [],
            },
        )
        flaky["executions"] += stats["executions"]
        flaky["passed"] += stats["passed"]
        flaky["failed"] += stats["failed"]
        flaky["error"] += stats["error"]
        flaky["total_duration"] += stats["total_duration"]
        flaky["recent_failures"].extend(stats["failure_dates"])
        flaky["environments"].append(stats["environment"])
        flaky["confirmed_revisions"].extend(confirmed_revisions)

    flaky_tests = []
    for flaky in flaky_by_nodeid.values():
        flaky["failures"] = flaky["failed"] + flaky["error"]
        flaky["failure_rate"] = round(flaky["failures"] / flaky["executions"], 4)
        flaky["avg_duration_s"] = round(
            flaky.pop("total_duration") / flaky["executions"], 3
        )
        flaky["recent_failures"] = sorted(flaky["recent_failures"])[-5:]
        flaky["environments"] = sorted(set(flaky["environments"]))
        flaky["confirmed_revisions"].sort(key=lambda revision: revision["date"])
        flaky["first_seen_flaky"] = flaky["confirmed_revisions"][0]["date"]
        flaky["last_failed"] = flaky["confirmed_revisions"][-1]["date"]
        flaky_tests.append(flaky)

    flaky_tests.sort(
        key=lambda test: (-len(test["confirmed_revisions"]), test["nodeid"])
    )
    return flaky_tests


# =============================================================================
# Report generation
# =============================================================================


def generate_json_report(
    flaky_tests: list[dict],
    test_stats: dict[tuple[str, str], dict],
    run_metadata: dict,
    output_path: Path,
    repo: str,
    previous_report: dict | None = None,
    now: datetime | None = None,
) -> dict:
    if now is None:
        now = datetime.now(timezone.utc)

    if run_metadata:
        dates = [v["date"] for v in run_metadata.values()]
        start_date = min(dates)[:10] if dates else now.strftime("%Y-%m-%d")
        end_date = max(dates)[:10] if dates else now.strftime("%Y-%m-%d")
    else:
        start_date = end_date = now.strftime("%Y-%m-%d")

    current_families = {test["family"] for test in flaky_tests}
    if previous_report is not None:
        previous_families = {
            test.get("family", test["nodeid"].split("[", 1)[0])
            for test in previous_report.get("flaky_tests", [])
        }
        new_families = current_families - previous_families
        resolved_families = previous_families - current_families
    else:
        new_cutoff = now - timedelta(days=NEW_FLAKY_DAYS)
        new_families = {
            test["family"]
            for test in flaky_tests
            if (first_seen := parse_datetime(test["first_seen_flaky"])) is not None
            and first_seen > new_cutoff
        }
        resolved_families = set()

    for test in flaky_tests:
        test["is_new"] = test["family"] in new_families

    report = {
        "generated_at": now.isoformat(),
        "repository": repo,
        "analysis_period": {
            "start": start_date,
            "end": end_date,
            "runs_analyzed": len(run_metadata),
        },
        "summary": {
            "total_tests": len({nodeid for nodeid, _environment in test_stats}),
            "flaky_count": len(current_families),
            "flaky_case_count": len(flaky_tests),
            "new_flaky_count": len(new_families),
            "resolved_count": len(resolved_families),
        },
        "flaky_tests": flaky_tests,
        "resolved_test_families": sorted(resolved_families),
        "thresholds": {
            "active_failure_days": ACTIVE_FLAKY_DAYS,
            "evidence": "fail_and_pass_on_same_revision_and_environment",
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
        f"- **Confirmed flaky test families**: {summary['flaky_count']}",
        f"- **Affected parameterized cases**: {summary['flaky_case_count']}",
        f"- **Newly confirmed**: {summary['new_flaky_count']}",
        f"- **Resolved since previous report**: {summary['resolved_count']}",
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
                "| Test | Environments | Confirmed revisions | Failures | Last failed |",
                "|------|--------------|---------------------|----------|-------------|",
            ]
        )

        for test in flaky_tests[:30]:
            nodeid = test["nodeid"]
            if len(nodeid) > 80:
                nodeid = "..." + nodeid[-77:]

            environments = "<br>".join(test["environments"])
            revisions = []
            for revision in test["confirmed_revisions"][-3:]:
                sha = revision["sha"][:7]
                if revision["run_ids"]:
                    run_id = revision["run_ids"][-1]
                    revisions.append(
                        f"[`{sha}`](https://github.com/"
                        f"{report['repository']}/actions/runs/{run_id})"
                    )
                else:
                    revisions.append(f"`{sha}`")
            revision_str = ", ".join(revisions)
            new_marker = " **NEW**" if test.get("is_new") else ""

            lines.append(
                f"| `{nodeid}`{new_marker} | {environments} | {revision_str} | "
                f"{test['failures']}/{test['executions']} | "
                f"{test['last_failed'][:10]} |"
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
                "No test has recent fail/pass evidence on the same commit and CI environment.",
                "",
            ]
        )

    resolved_tests = report.get("resolved_test_families", [])
    if resolved_tests:
        lines.extend(["## Resolved Since Previous Report", ""])
        for test in resolved_tests[:30]:
            lines.append(f"- `{test}`")
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## Configuration",
            "",
            "- Required evidence: fail and pass on the same commit and CI environment",
            f"- Active failure window: {report['thresholds']['active_failure_days']} days",
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
    parser.add_argument(
        "--previous-report",
        help="Previous JSON report used to calculate new and resolved tests",
    )
    args = parser.parse_args()

    repo = get_repo()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflows = [w.strip() for w in args.workflows.split(",") if w.strip()]
    previous_report = None
    if args.previous_report:
        previous_report_path = Path(args.previous_report)
        if previous_report_path.exists():
            try:
                with open(previous_report_path, encoding="utf-8") as f:
                    previous_report = json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Warning: could not read previous report: {exc}")

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
        raise RuntimeError(
            "No test-level data collected; refusing to replace the existing report"
        )

    print("\n" + "=" * 60)
    print("Aggregating per-test statistics...")
    print("=" * 60)
    test_stats = aggregate_test_stats(all_tests)
    print(f"  Analyzed {len(test_stats)} test/environment combinations")

    print("  Identifying flaky tests...")
    flaky_tests = identify_flaky_tests(test_stats)
    flaky_families = {test["family"] for test in flaky_tests}
    print(
        f"  Found {len(flaky_families)} flaky test families "
        f"across {len(flaky_tests)} parameterized cases"
    )

    print("\nGenerating reports...")

    json_report = generate_json_report(
        flaky_tests,
        test_stats,
        all_run_metadata,
        output_dir / "flaky-tests.json",
        repo,
        previous_report,
    )

    json_report["workflows_analyzed"] = workflows
    with open(output_dir / "flaky-tests.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    generate_markdown_report(json_report, output_dir / "flaky-tests.md")
    generate_badge_json(
        json_report["summary"]["flaky_count"], output_dir / "badge.json"
    )

    print(f"\nReports written to {output_dir}/")
    print("  - flaky-tests.json")
    print("  - flaky-tests.md")
    print("  - badge.json")

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"flaky_count={json_report['summary']['flaky_count']}\n")
            f.write(f"new_flaky_count={json_report['summary']['new_flaky_count']}\n")
            f.write(f"resolved_count={json_report['summary']['resolved_count']}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
