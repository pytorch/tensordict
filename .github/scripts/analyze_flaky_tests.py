#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Analyze CI job results from recent workflow runs to identify flaky jobs.

This script:
1. Fetches workflow run data via the GitHub API
2. Lists jobs within each run and their pass/fail status
3. Aggregates per-job statistics across runs
4. Identifies flaky jobs based on intermittent failure patterns
5. Generates JSON and Markdown reports

Unlike artifact-based approaches, this works by checking job-level
pass/fail outcomes directly from the GitHub Actions API -- no test
result artifacts are required.
"""

import argparse
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

# =============================================================================
# Configuration - Thresholds for flaky detection
# =============================================================================

# Minimum failure rate to be considered flaky (below this = probably just fixed)
FLAKY_THRESHOLD_MIN = 0.05  # 5%

# Maximum failure rate to be considered flaky (above this = broken, not flaky)
FLAKY_THRESHOLD_MAX = 0.80  # 80%

# Minimum number of failures required to flag as flaky
MIN_FAILURES_FOR_FLAKY = 2

# Minimum number of executions required for analysis
MIN_EXECUTIONS = 3

# Days to consider a job "newly flaky"
NEW_FLAKY_DAYS = 7


# =============================================================================
# GitHub API Helpers (using gh CLI)
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


def get_repo() -> str:
    """Get repository from environment."""
    return os.environ.get("GITHUB_REPOSITORY", "pytorch/tensordict")


# =============================================================================
# Data Collection
# =============================================================================


def list_workflow_runs(
    repo: str, workflow_name: str, branch: str, num_runs: int
) -> list[dict]:
    """List recent completed workflow runs for a specific workflow."""
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

        workflow_runs = data["workflow_runs"]
        if not workflow_runs:
            break

        runs.extend(workflow_runs)
        page += 1

        if len(workflow_runs) < per_page:
            break

    return runs[:num_runs]


def get_run_jobs(repo: str, run_id: int) -> list[dict]:
    """Fetch all jobs for a given workflow run."""
    jobs: list[dict] = []
    page = 1

    while True:
        endpoint = (
            f"/repos/{repo}/actions/runs/{run_id}/jobs" f"?per_page=100&page={page}"
        )
        data = gh_api(endpoint)

        if not data or "jobs" not in data:
            break

        batch = data["jobs"]
        if not batch:
            break

        jobs.extend(batch)
        page += 1

        if len(batch) < 100:
            break

    return jobs


def collect_job_data(
    repo: str, workflow_name: str, num_runs: int
) -> tuple[list[dict], dict]:
    """Collect job-level pass/fail data from recent workflow runs."""
    print(f"Fetching last {num_runs} runs of {workflow_name} on main...")

    runs = list_workflow_runs(repo, workflow_name, "main", num_runs)
    print(f"  Found {len(runs)} completed runs")

    all_jobs: list[dict] = []
    run_metadata: dict = {}

    for run in runs:
        run_id = run["id"]
        run_date = run["created_at"]
        commit_sha = run["head_sha"]

        jobs = get_run_jobs(repo, run_id)

        for job in jobs:
            job["_run_id"] = run_id
            job["_run_date"] = run_date
            job["_commit_sha"] = commit_sha
            all_jobs.append(job)

        run_metadata[run_id] = {
            "date": run_date,
            "sha": commit_sha,
            "conclusion": run["conclusion"],
        }

    print(f"  Collected {len(all_jobs)} jobs from {len(run_metadata)} runs")
    return all_jobs, run_metadata


# =============================================================================
# Analysis
# =============================================================================


def aggregate_job_stats(jobs: list[dict]) -> dict[str, dict]:
    """Aggregate statistics per job name across all runs."""
    job_stats: dict[str, dict] = defaultdict(
        lambda: {
            "executions": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "cancelled": 0,
            "total_duration": 0.0,
            "failure_dates": [],
            "run_ids": set(),
        }
    )

    for job in jobs:
        name = job.get("name", "")
        if not name:
            continue

        conclusion = job.get("conclusion", "")
        # Skip jobs that were cancelled or skipped -- they aren't informative
        if conclusion in ("cancelled", "skipped", None, ""):
            continue

        started = job.get("started_at")
        completed = job.get("completed_at")
        duration = 0.0
        if started and completed:
            try:
                t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(completed.replace("Z", "+00:00"))
                duration = (t1 - t0).total_seconds()
            except ValueError:
                pass

        run_id = job.get("_run_id", "unknown")
        run_date = job.get("_run_date", "")

        stats = job_stats[name]
        stats["executions"] += 1
        stats["run_ids"].add(run_id)
        stats["total_duration"] += duration

        if conclusion == "success":
            stats["passed"] += 1
        elif conclusion == "failure":
            stats["failed"] += 1
            if run_date:
                stats["failure_dates"].append(run_date)

    for _name, stats in job_stats.items():
        stats["run_ids"] = list(stats["run_ids"])

    return dict(job_stats)


def calculate_flaky_score(stats: dict) -> float:
    """Calculate a flaky score for a job.

    Returns a score between 0 and 1, where higher = more flaky.
    Peak flakiness is at 50% failure rate.
    """
    if stats["executions"] < MIN_EXECUTIONS:
        return 0.0

    failure_rate = stats["failed"] / stats["executions"]

    if failure_rate >= FLAKY_THRESHOLD_MAX or failure_rate <= FLAKY_THRESHOLD_MIN:
        return 0.0

    if failure_rate <= 0.5:
        base_score = failure_rate * 2
    else:
        base_score = (1 - failure_rate) * 2

    confidence = min(1.0, stats["failed"] / 5)

    return min(1.0, base_score * confidence)


def identify_flaky_jobs(job_stats: dict[str, dict]) -> list[dict]:
    """Identify flaky jobs based on statistics."""
    flaky_jobs = []

    for name, stats in job_stats.items():
        if stats["executions"] < MIN_EXECUTIONS:
            continue

        if stats["failed"] < MIN_FAILURES_FOR_FLAKY:
            continue

        failure_rate = stats["failed"] / stats["executions"]

        if failure_rate <= FLAKY_THRESHOLD_MIN or failure_rate >= FLAKY_THRESHOLD_MAX:
            continue

        flaky_score = calculate_flaky_score(stats)
        if flaky_score <= 0:
            continue

        first_failure = None
        if stats["failure_dates"]:
            failure_dates = sorted(stats["failure_dates"])
            first_failure = failure_dates[0]

        is_new = False
        if first_failure:
            try:
                first_dt = datetime.fromisoformat(first_failure.replace("Z", "+00:00"))
                cutoff = datetime.now(timezone.utc) - timedelta(days=NEW_FLAKY_DAYS)
                is_new = first_dt > cutoff
            except ValueError:
                pass

        avg_duration = (
            stats["total_duration"] / stats["executions"] if stats["executions"] else 0
        )

        flaky_jobs.append(
            {
                "name": name,
                "executions": stats["executions"],
                "failures": stats["failed"],
                "failure_rate": round(failure_rate, 4),
                "flaky_score": round(flaky_score, 4),
                "passed": stats["passed"],
                "avg_duration_s": round(avg_duration, 1),
                "recent_failures": (
                    sorted(stats["failure_dates"])[-5:]
                    if stats["failure_dates"]
                    else []
                ),
                "first_seen_flaky": first_failure,
                "is_new": is_new,
            }
        )

    flaky_jobs.sort(key=lambda x: x["flaky_score"], reverse=True)
    return flaky_jobs


# =============================================================================
# Report Generation
# =============================================================================


def generate_json_report(
    flaky_jobs: list[dict],
    job_stats: dict[str, dict],
    run_metadata: dict,
    output_path: Path,
) -> dict:
    """Generate JSON report."""
    now = datetime.now(timezone.utc)

    if run_metadata:
        dates = [v["date"] for v in run_metadata.values()]
        start_date = min(dates)[:10] if dates else now.strftime("%Y-%m-%d")
        end_date = max(dates)[:10] if dates else now.strftime("%Y-%m-%d")
    else:
        start_date = end_date = now.strftime("%Y-%m-%d")

    new_flaky_count = sum(1 for t in flaky_jobs if t.get("is_new", False))

    report = {
        "generated_at": now.isoformat(),
        "analysis_period": {
            "start": start_date,
            "end": end_date,
            "runs_analyzed": len(run_metadata),
        },
        "summary": {
            "total_jobs": len(job_stats),
            "total_tests": len(job_stats),
            "flaky_count": len(flaky_jobs),
            "new_flaky_count": new_flaky_count,
            "resolved_count": 0,
        },
        "flaky_tests": flaky_jobs,
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
    """Generate Markdown report."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    summary = report["summary"]
    flaky_jobs = report["flaky_tests"]

    lines = [
        f"# Flaky Test Report - {now_str}",
        "",
        "## Summary",
        "",
        f"- **Flaky jobs**: {summary['flaky_count']}",
        f"- **Newly flaky** (last 7 days): {summary['new_flaky_count']}",
        f"- **Total jobs analyzed**: {summary['total_jobs']}",
        f"- **CI runs analyzed**: {report['analysis_period']['runs_analyzed']}",
        "",
        "---",
        "",
    ]

    if flaky_jobs:
        lines.extend(
            [
                "## Flaky Jobs",
                "",
                "| Job | Failure Rate | Failures | Flaky Score | Last Failed |",
                "|-----|--------------|----------|-------------|-------------|",
            ]
        )

        for job in flaky_jobs[:20]:
            name = job["name"]
            if len(name) > 60:
                name = "..." + name[-57:]

            rate_str = f"{job['failure_rate'] * 100:.1f}% ({job['failures']}/{job['executions']})"
            score_str = f"{job['flaky_score']:.2f}"
            last_failed = (
                job["recent_failures"][-1][:10] if job["recent_failures"] else "N/A"
            )

            new_marker = " NEW" if job.get("is_new") else ""

            lines.append(
                f"| `{name}`{new_marker} | {rate_str} | "
                f"{job['failures']} | {score_str} | {last_failed} |"
            )

        lines.extend(["", ""])

        if summary["new_flaky_count"] > 0:
            lines.extend(["### Newly Flaky", ""])
            new_jobs = [j for j in flaky_jobs if j.get("is_new")]
            for job in new_jobs:
                lines.append(f"- `{job['name']}`")
            lines.append("")
    else:
        lines.extend(
            [
                "## No Flaky Jobs Detected!",
                "",
                "All CI jobs are passing consistently.",
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
    """Generate shields.io endpoint badge JSON."""
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
        description="Analyze flaky CI jobs from recent workflow runs"
    )
    parser.add_argument(
        "--runs", type=int, default=30, help="Number of runs to analyze per workflow"
    )
    parser.add_argument(
        "--workflows",
        default="test-linux.yml",
        help="Comma-separated list of workflow file names",
    )
    parser.add_argument(
        "--output-dir", default="flaky-reports", help="Output directory"
    )
    args = parser.parse_args()

    repo = get_repo()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflows = [w.strip() for w in args.workflows.split(",") if w.strip()]

    print(f"Analyzing flaky jobs for {repo}")
    print(f"  Workflows: {', '.join(workflows)}")
    print(f"  Runs to analyze per workflow: {args.runs}")
    print()

    all_jobs: list[dict] = []
    all_run_metadata: dict = {}

    for workflow in workflows:
        print(f"\n{'=' * 60}")
        print(f"Processing workflow: {workflow}")
        print("=" * 60)

        job_data, run_metadata = collect_job_data(repo, workflow, args.runs)
        all_jobs.extend(job_data)
        all_run_metadata.update(run_metadata)

    if not all_jobs:
        print("\nNo job data collected. Generating empty report.")
        job_stats: dict[str, dict] = {}
        flaky_jobs: list[dict] = []
    else:
        print("\n" + "=" * 60)
        print("Aggregating job statistics...")
        print("=" * 60)
        job_stats = aggregate_job_stats(all_jobs)
        print(f"  Analyzed {len(job_stats)} unique jobs")

        print("  Identifying flaky jobs...")
        flaky_jobs = identify_flaky_jobs(job_stats)
        print(f"  Found {len(flaky_jobs)} flaky jobs")

    print("\nGenerating reports...")

    json_report = generate_json_report(
        flaky_jobs, job_stats, all_run_metadata, output_dir / "flaky-tests.json"
    )

    json_report["workflows_analyzed"] = workflows

    with open(output_dir / "flaky-tests.json", "w", encoding="utf-8") as f:
        json.dump(json_report, f, indent=2)

    generate_markdown_report(json_report, output_dir / "flaky-tests.md")
    generate_badge_json(len(flaky_jobs), output_dir / "badge.json")

    print(f"\nReports written to {output_dir}/")
    print("  - flaky-tests.json")
    print("  - flaky-tests.md")
    print("  - badge.json")

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"flaky_count={len(flaky_jobs)}\n")
            f.write(f"new_flaky_count={json_report['summary']['new_flaky_count']}\n")
            f.write(f"resolved_count={json_report['summary']['resolved_count']}\n")

    print("\nDone!")


if __name__ == "__main__":
    main()
