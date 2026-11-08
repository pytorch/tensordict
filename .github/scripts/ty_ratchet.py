"""Fail when a change adds ty diagnostics to tensordict.

Runs ``ty check`` on the ``tensordict`` package, counts its diagnostics per
(file, rule) and compares the counts with ``.github/ty-baseline.json``. The
check fails when a count rises above the baseline (a file that is not in the
baseline counts from 0), and prints every diagnostic of each such (file, rule).
For the rules in ``REPORT_ONLY`` a rise is printed but does not fail.
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
# Rules whose rises are reported without failing the check: on tensordict,
# almost all of their new diagnostics come from narrowing that ty cannot
# follow (is_tensor_collection checks, values that an earlier branch rules
# out) or from mixins that read attributes of the class they are mixed into.
# Replayed over the 40 merges to main before this check was added, these rules
# account for the rises in 11 of the 12 merges that a check failing on every
# rule would have failed, and none of those was a bug. The twelfth was an
# attribute read on a value that may be None (#3541), which NONE_ATTRIBUTE
# below counts as a rule of its own, one that fails the check.
REPORT_ONLY = frozenset(
    {
        "import-not-found",
        "invalid-argument-type",
        "unresolved-attribute",
        "unsupported-operator",
    }
)


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


# An attribute read on a value that may be None ("Attribute `x` is not defined
# on `None` in union ..."): counted as a rule of its own, which fails the
# check, because it can raise AttributeError at run time (#3541).
NONE_ATTRIBUTE = "unresolved-attribute-on-none"


def key(diagnostic: dict) -> tuple[str, str]:
    rule = diagnostic["check_name"]
    if (
        rule == "unresolved-attribute"
        and "is not defined on `None` in union" in (diagnostic["description"])
    ):
        rule = NONE_ATTRIBUTE
    return diagnostic["location"]["path"], rule


def format_diagnostic(diagnostic: dict) -> str:
    path, rule = key(diagnostic)
    begin = diagnostic["location"]["positions"]["begin"]
    description = diagnostic["description"].removeprefix(
        f"{diagnostic['check_name']}: "
    )
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
    reported = {k for k in risen if k[1] in REPORT_ONLY}
    failing = risen - reported

    def show(keys: set[tuple[str, str]]) -> None:
        for k in sorted(keys):
            was = baseline.get(k[0], {}).get(k[1], 0)
            print(f"{k[0]} {k[1]}: {was} -> {counts[k]}")
            for d in diagnostics:
                if key(d) == k:
                    print(f"  {format_diagnostic(d)}")

    if reported:
        print(
            f"\nThese (file, rule) counts are above {BASELINE.relative_to(ROOT)}, "
            "for rules that are reported without failing the check; all "
            "diagnostics of each are listed:\n"
        )
        show(reported)
    if failing:
        print(
            f"\nThese (file, rule) counts are above {BASELINE.relative_to(ROOT)}; "
            "all diagnostics of each are listed:\n"
        )
        show(failing)
        print(
            "\nFix the new diagnostics. If one is a ty false positive, or the file "
            "was renamed or moved (its counts start from 0 under the new path), "
            "rewrite the baseline with `python .github/scripts/ty_ratchet.py "
            "--update` and say why in the PR."
        )
        return 1

    fallen = baseline_total - len(diagnostics)
    if fallen > 0:
        print(
            f"The count fell by {fallen} from the baseline's {baseline_total}. "
            "Run `python .github/scripts/ty_ratchet.py --update` to lock the lower "
            "counts in."
        )
    elif reported:
        print("\nNo count of the other rules rose.")
    else:
        print("The counts match the baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
