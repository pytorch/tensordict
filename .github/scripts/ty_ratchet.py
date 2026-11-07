"""Fail when a change adds ty diagnostics to tensordict.

Runs ``ty check`` on the ``tensordict`` package, counts its diagnostics per
(file, rule) and compares the counts with ``.github/ty-baseline.json``. The
check fails when any count rises above the baseline (a file that is not in the
baseline counts from 0), and prints every diagnostic of each such (file, rule).
Counts that fall pass: ``--update`` rewrites the baseline from the current
counts, to lock the lower counts in or to accept a ty false positive.

The counts depend on the ty version and on the packages ty resolves imports
against, so the check runs in the environment pinned in
``.github/ty-requirements.txt``. To run it locally, from the repository root:

    uv venv --python 3.12 .ty-venv
    uv pip install --python .ty-venv/bin/python --torch-backend cpu \\
        -c .github/ci-constraints.txt -r .github/ty-requirements.txt
    python .github/scripts/ty_ratchet.py [--update]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / ".github" / "ty-baseline.json"
PACKAGE = "tensordict"


def run_ty(python: Path) -> list[dict]:
    ty = python.parent / "ty"
    result = subprocess.run(
        [
            str(ty),
            "check",
            "--python",
            str(python),
            "--output-format",
            "gitlab",
            "--exit-zero",
            PACKAGE,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(f"ty exited with {result.returncode}")
    return json.loads(result.stdout)


def key(diagnostic: dict) -> tuple[str, str]:
    return diagnostic["location"]["path"], diagnostic["check_name"]


def format_diagnostic(diagnostic: dict) -> str:
    path, rule = key(diagnostic)
    begin = diagnostic["location"]["positions"]["begin"]
    description = diagnostic["description"].removeprefix(f"{rule}: ")
    return f"{path}:{begin['line']}:{begin['column']} {rule} {description}"


def to_nested(counts: Counter) -> dict[str, dict[str, int]]:
    nested: dict[str, dict[str, int]] = defaultdict(dict)
    for (path, rule), count in counts.items():
        nested[path][rule] = count
    return {path: dict(sorted(rules.items())) for path, rules in sorted(nested.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--python",
        type=Path,
        default=ROOT / ".ty-venv" / "bin" / "python",
        help="the Python of the environment built from .github/ty-requirements.txt; "
        "ty is run from the same bin directory (default: .ty-venv/bin/python)",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite .github/ty-baseline.json from the current counts",
    )
    args = parser.parse_args()

    diagnostics = run_ty(args.python.absolute())
    counts = Counter(key(d) for d in diagnostics)
    print(f"ty reports {len(diagnostics)} diagnostics in {PACKAGE}.")

    if args.update:
        BASELINE.write_text(json.dumps(to_nested(counts), indent=1) + "\n")
        print(f"Wrote {BASELINE.relative_to(ROOT)}.")
        return 0

    baseline = json.loads(BASELINE.read_text())
    baseline_total = sum(sum(rules.values()) for rules in baseline.values())
    risen = {
        k for k, count in counts.items() if count > baseline.get(k[0], {}).get(k[1], 0)
    }
    if risen:
        print(
            f"\nThese (file, rule) counts are above {BASELINE.relative_to(ROOT)}; "
            "all diagnostics of each are listed:\n"
        )
        for k in sorted(risen):
            was = baseline.get(k[0], {}).get(k[1], 0)
            print(f"{k[0]} {k[1]}: {was} -> {counts[k]}")
            for d in diagnostics:
                if key(d) == k:
                    print(f"  {format_diagnostic(d)}")
        print(
            "\nFix the new diagnostics. If one is a ty false positive, raise the "
            "baseline with `python .github/scripts/ty_ratchet.py --update` and say "
            "why in the PR."
        )
        return 1

    fallen = baseline_total - len(diagnostics)
    if fallen > 0:
        print(
            f"The count fell by {fallen} from the baseline's {baseline_total}. "
            "Run `python .github/scripts/ty_ratchet.py --update` to lock the lower "
            "counts in."
        )
    else:
        print("The counts match the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
