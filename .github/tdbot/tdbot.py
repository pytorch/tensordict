#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""tensordictbot - A GitHub bot for managing PRs in the tensordict repository.

Triggered by PR comments starting with ``@tensordictbot``. Supports commands:
  - merge: Merge a PR (or ghstack) using ghstack land or gh pr merge
  - rebase: Rebase a PR onto a target branch
  - lint: Auto-fix lint issues and push the result

Inspired by PyTorch's @pytorchbot.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def gh(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a ``gh`` CLI command and return the result."""
    return subprocess.run(
        ["gh", *args],
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )


def git(*args: str, **kwargs) -> subprocess.CompletedProcess:
    """Run a ``git`` command and return the result."""
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        check=True,
        **kwargs,
    )


def post_comment(repo: str, pr_number: int, body: str) -> None:
    """Post a comment on the given PR."""
    gh("pr", "comment", str(pr_number), "--repo", repo, "--body", body)


def react_to_comment(repo: str, comment_id: int, reaction: str) -> None:
    """Add a reaction to a comment."""
    try:
        gh(
            "api",
            f"repos/{repo}/issues/comments/{comment_id}/reactions",
            "-f",
            f"content={reaction}",
            "--silent",
        )
    except subprocess.CalledProcessError:
        pass


def get_pr_info(repo: str, pr_number: int) -> dict:
    """Fetch PR metadata via ``gh``."""
    result = gh(
        "pr",
        "view",
        str(pr_number),
        "--repo",
        repo,
        "--json",
        "headRefName,baseRefName,author,title,url,labels,mergeable,reviewDecision",
    )
    return json.loads(result.stdout)


def get_permission(repo: str, username: str) -> str | None:
    """Return the permission level of *username* on *repo*, or ``None`` on error."""
    try:
        result = gh(
            "api",
            f"repos/{repo}/collaborators/{username}/permission",
        )
        data = json.loads(result.stdout)
        return data.get("permission")
    except subprocess.CalledProcessError:
        return None


def check_write_permission(repo: str, username: str) -> bool:
    """Return True if *username* has write (or higher) permission on *repo*."""
    return get_permission(repo, username) in ("admin", "maintain", "write")


def check_admin_permission(repo: str, username: str) -> bool:
    """Return True if *username* has admin or maintain permission on *repo*."""
    return get_permission(repo, username) in ("admin", "maintain")


def is_ghstack_pr(head_branch: str) -> bool:
    """Detect whether the PR was created by ghstack."""
    return bool(re.match(r"^gh/[^/]+/\d+/head$", head_branch))


def find_ghstack_stack_top(repo: str, head_branch: str) -> int:
    """Return the PR number at the top of the ghstack stack containing *head_branch*.

    Follows the baseRefName chain: each ghstack PR N+1 has
    ``baseRefName == gh/USER/N/orig``.  We walk forward from the current PR
    until no successor is found.
    """
    m = re.match(r"^gh/([^/]+)/(\d+)/head$", head_branch)
    username = m.group(1)

    result = gh(
        "pr",
        "list",
        "--repo",
        repo,
        "--json",
        "number,headRefName,baseRefName",
        "--limit",
        "200",
        "--state",
        "open",
    )
    all_prs = json.loads(result.stdout)

    stack_pattern = re.compile(rf"^gh/{re.escape(username)}/(\d+)/head$")
    by_base: dict[str, int] = {}
    pr_num_for: dict[int, int] = {}
    for pr in all_prs:
        sm = stack_pattern.match(pr["headRefName"])
        if sm:
            idx = int(sm.group(1))
            pr_num_for[idx] = pr["number"]
            by_base[pr["baseRefName"]] = idx

    current_idx = int(m.group(2))
    while True:
        orig_branch = f"gh/{username}/{current_idx}/orig"
        next_idx = by_base.get(orig_branch)
        if next_idx is None:
            break
        current_idx = next_idx

    return pr_num_for.get(current_idx, current_idx)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@dataclass
class CommandContext:
    repo: str
    pr_number: int
    comment_id: int
    comment_author: str
    pr_info: dict = field(default_factory=dict)


def cmd_merge(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Handle ``@tensordictbot merge``."""
    pr = ctx.pr_info
    head = pr["headRefName"]

    if not check_write_permission(ctx.repo, ctx.comment_author):
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"@{ctx.comment_author} you don't have write permission on this repository. "
            "Only collaborators with write access can merge PRs.",
        )
        return

    is_admin = check_admin_permission(ctx.repo, ctx.comment_author)
    if not args.force and not is_admin:
        decision = pr.get("reviewDecision", "")
        if decision != "APPROVED":
            post_comment(
                ctx.repo,
                ctx.pr_number,
                f"@{ctx.comment_author} this PR has not been approved yet "
                f"(current status: **{decision or 'REVIEW_REQUIRED'}**). "
                "Use `@tensordictbot merge -f 'reason'` to force merge.",
            )
            return

    if is_ghstack_pr(head):
        _merge_ghstack(ctx, args)
    else:
        _merge_regular(ctx, args)


def _merge_ghstack(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Merge a ghstack PR using ``ghstack land``."""
    pr = ctx.pr_info
    url = pr["url"]

    post_comment(
        ctx.repo,
        ctx.pr_number,
        f"Merging ghstack PR via `ghstack land` (requested by @{ctx.comment_author}).\n\n"
        + (f"Force reason: {args.force}\n" if args.force else ""),
    )

    try:
        subprocess.run(
            ["ghstack", "land", url],
            capture_output=True,
            text=True,
            check=True,
        )
        post_comment(ctx.repo, ctx.pr_number, "ghstack land completed successfully.")
    except subprocess.CalledProcessError as exc:
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"ghstack land **failed**.\n\n```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def _merge_regular(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Merge a regular (non-ghstack) PR using ``gh pr merge``."""
    merge_args = [
        "pr",
        "merge",
        str(ctx.pr_number),
        "--repo",
        ctx.repo,
        "--squash",
        "--delete-branch",
    ]

    msg_parts = [f"Merging PR (requested by @{ctx.comment_author})."]
    if args.force:
        msg_parts.append(f"Force reason: {args.force}")
        merge_args.append("--admin")

    post_comment(ctx.repo, ctx.pr_number, "\n".join(msg_parts))

    try:
        gh(*merge_args)
        post_comment(ctx.repo, ctx.pr_number, "PR merged successfully.")
    except subprocess.CalledProcessError as exc:
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Merge **failed**.\n\n```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def cmd_rebase(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Handle ``@tensordictbot rebase``."""
    pr = ctx.pr_info
    head = pr["headRefName"]

    if not check_write_permission(ctx.repo, ctx.comment_author):
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"@{ctx.comment_author} you don't have write permission on this repository. "
            "Only collaborators with write access can rebase PRs.",
        )
        return

    if is_ghstack_pr(head):
        _rebase_ghstack(ctx, args)
    else:
        _rebase_regular(ctx, args)


def _rebase_regular(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Rebase a regular (non-ghstack) PR branch."""
    head = ctx.pr_info["headRefName"]
    target_branch = args.branch

    post_comment(
        ctx.repo,
        ctx.pr_number,
        f"Rebasing `{head}` onto `{target_branch}` (requested by @{ctx.comment_author}).",
    )

    try:
        git("fetch", "origin", target_branch)
        git("fetch", "origin", head)
        git("checkout", head)
        git("rebase", f"origin/{target_branch}")
        git("push", "origin", head, "--force-with-lease")
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Rebase onto `{target_branch}` completed successfully.",
        )
    except subprocess.CalledProcessError as exc:
        try:
            git("rebase", "--abort")
        except subprocess.CalledProcessError:
            pass
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Rebase **failed**.\n\n```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def _rebase_ghstack(ctx: CommandContext, args: argparse.Namespace) -> None:
    """Rebase an entire ghstack stack onto the target branch."""
    head = ctx.pr_info["headRefName"]
    target_branch = args.branch
    top_pr = find_ghstack_stack_top(ctx.repo, head)

    post_comment(
        ctx.repo,
        ctx.pr_number,
        f"Rebasing ghstack stack onto `{target_branch}` "
        f"(top of stack: #{top_pr}, requested by @{ctx.comment_author}).",
    )

    try:
        git("fetch", "origin", target_branch)
        subprocess.run(
            ["ghstack", "checkout", f"https://github.com/{ctx.repo}/pull/{top_pr}"],
            capture_output=True,
            text=True,
            check=True,
        )
        git("rebase", f"origin/{target_branch}")
        subprocess.run(
            ["ghstack", "submit"],
            capture_output=True,
            text=True,
            check=True,
        )
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Ghstack stack rebased onto `{target_branch}` successfully.",
        )
    except subprocess.CalledProcessError as exc:
        try:
            git("rebase", "--abort")
        except subprocess.CalledProcessError:
            pass
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Ghstack rebase **failed**.\n\n```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def cmd_lint(ctx: CommandContext, _args: argparse.Namespace) -> None:
    """Handle ``@tensordictbot lint``."""
    pr = ctx.pr_info
    head = pr["headRefName"]

    if not check_write_permission(ctx.repo, ctx.comment_author):
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"@{ctx.comment_author} you don't have write permission on this repository. "
            "Only collaborators with write access can run lint fixes.",
        )
        return

    post_comment(
        ctx.repo,
        ctx.pr_number,
        f"Running lint auto-fix on `{head}` (requested by @{ctx.comment_author}).",
    )

    if is_ghstack_pr(head):
        _lint_ghstack(ctx)
    else:
        _lint_regular(ctx)


def _lint_regular(ctx: CommandContext) -> None:
    """Run lint auto-fix on a regular PR branch."""
    head = ctx.pr_info["headRefName"]

    try:
        git("fetch", "origin", head)
        git("checkout", head)
    except subprocess.CalledProcessError as exc:
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Lint **failed** (could not checkout branch).\n\n"
            f"```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)

    subprocess.run(
        ["pre-commit", "run", "--all-files"],
        capture_output=True,
        text=True,
    )

    diff = git("diff")
    if not diff.stdout.strip():
        post_comment(ctx.repo, ctx.pr_number, "Lint auto-fix found nothing to change.")
        return

    try:
        git("add", "-A")
        git("commit", "-m", "[tensordictbot] lint fixes")
        git("push", "origin", head)
        post_comment(
            ctx.repo,
            ctx.pr_number,
            "Lint fixes committed and pushed successfully.",
        )
    except subprocess.CalledProcessError as exc:
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Lint **failed** (could not push fixes).\n\n"
            f"```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def _lint_ghstack(ctx: CommandContext) -> None:
    """Run lint auto-fix on a ghstack PR (amend the commit and re-submit)."""
    try:
        subprocess.run(
            ["ghstack", "checkout", ctx.pr_info["url"]],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Lint **failed** (ghstack checkout failed).\n\n"
            f"```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)

    subprocess.run(
        ["pre-commit", "run", "--all-files"],
        capture_output=True,
        text=True,
    )

    diff = git("diff")
    if not diff.stdout.strip():
        post_comment(ctx.repo, ctx.pr_number, "Lint auto-fix found nothing to change.")
        return

    try:
        git("add", "-A")
        git("commit", "--amend", "--no-edit")
        subprocess.run(
            ["ghstack", "submit"],
            capture_output=True,
            text=True,
            check=True,
        )
        post_comment(
            ctx.repo,
            ctx.pr_number,
            "Lint fixes amended and pushed via ghstack successfully.",
        )
    except subprocess.CalledProcessError as exc:
        post_comment(
            ctx.repo,
            ctx.pr_number,
            f"Lint **failed** (could not push fixes).\n\n"
            f"```\n{exc.stderr or exc.stdout}\n```",
        )
        sys.exit(1)


def cmd_help(ctx: CommandContext, _args: argparse.Namespace) -> None:
    """Handle ``@tensordictbot help``."""
    post_comment(ctx.repo, ctx.pr_number, HELP_TEXT)


# ---------------------------------------------------------------------------
# CLI parser
# ---------------------------------------------------------------------------

HELP_TEXT = """\
## @tensordictbot Help

```
usage: @tensordictbot {merge,rebase,lint,help}
```

### `merge`
Merge a PR. For ghstack PRs, uses `ghstack land`; otherwise uses `gh pr merge --squash`.

```
@tensordictbot merge [-f MESSAGE]
```

| Flag | Description |
|------|-------------|
| `-f`, `--force` | Force merge with a reason (bypasses approval check, uses `--admin`) |

> **Note:** Repository admins and maintainers can merge without approval. The
> approval gate only applies to regular collaborators with write access.

### `rebase`
Rebase the PR branch onto a target branch. For ghstack PRs, rebases the
entire stack and re-submits via `ghstack submit`.

```
@tensordictbot rebase [-b BRANCH]
```

| Flag | Description |
|------|-------------|
| `-b`, `--branch` | Target branch (default: `main`) |

### `lint`
Run the linter (pre-commit) and auto-fix issues. For regular PRs, commits
the fixes. For ghstack PRs, amends the commit and re-submits the stack.

```
@tensordictbot lint
```

### `help`
Show this help message.

```
@tensordictbot help
```
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="@tensordictbot", add_help=False)
    sub = parser.add_subparsers(dest="command")

    merge_p = sub.add_parser("merge", add_help=False)
    merge_p.add_argument(
        "-f",
        "--force",
        type=str,
        default=None,
        help="Force merge with a reason (bypasses approval gate)",
    )

    rebase_p = sub.add_parser("rebase", add_help=False)
    rebase_p.add_argument(
        "-b",
        "--branch",
        type=str,
        default="main",
        help="Branch to rebase onto (default: main)",
    )

    sub.add_parser("lint", add_help=False)

    sub.add_parser("help", add_help=False)

    return parser


COMMAND_HANDLERS = {
    "merge": cmd_merge,
    "land": cmd_merge,
    "rebase": cmd_rebase,
    "lint": cmd_lint,
    "linter": cmd_lint,
    "help": cmd_help,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_command(comment_body: str) -> list[str] | None:
    """Extract the @tensordictbot command tokens from a comment body.

    Scans for the first line that starts with ``@tensordictbot`` (ignoring
    leading whitespace) and returns the tokens after ``@tensordictbot``.
    """
    for line in comment_body.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("@tensordictbot"):
            rest = stripped[len("@tensordictbot") :].strip()
            if not rest:
                return []
            return rest.split()
    return None


def main() -> None:
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        sys.stderr.write(
            "GITHUB_EVENT_PATH not set -- are you running inside GitHub Actions?\n"
        )
        sys.exit(1)

    with open(event_path) as f:
        event = json.load(f)

    repo = os.environ["GITHUB_REPOSITORY"]

    comment_body = event["comment"]["body"]
    comment_id = event["comment"]["id"]
    comment_author = event["comment"]["user"]["login"]
    pr_number = event["issue"]["number"]

    tokens = parse_command(comment_body)
    if tokens is None:
        return

    react_to_comment(repo, comment_id, "+1")

    parser = build_parser()

    if not tokens:
        ctx = CommandContext(
            repo=repo,
            pr_number=pr_number,
            comment_id=comment_id,
            comment_author=comment_author,
        )
        cmd_help(ctx, argparse.Namespace())
        return

    try:
        args = parser.parse_args(tokens)
    except SystemExit:
        ctx = CommandContext(
            repo=repo,
            pr_number=pr_number,
            comment_id=comment_id,
            comment_author=comment_author,
        )
        post_comment(
            repo,
            pr_number,
            f"@{comment_author} I couldn't parse that command. "
            f"Run `@tensordictbot help` to see available commands.\n\n"
            f"Input: `@tensordictbot {' '.join(tokens)}`",
        )
        return

    handler = COMMAND_HANDLERS.get(args.command)
    if handler is None:
        ctx = CommandContext(
            repo=repo,
            pr_number=pr_number,
            comment_id=comment_id,
            comment_author=comment_author,
        )
        cmd_help(ctx, args)
        return

    pr_info = get_pr_info(repo, pr_number)

    ctx = CommandContext(
        repo=repo,
        pr_number=pr_number,
        comment_id=comment_id,
        comment_author=comment_author,
        pr_info=pr_info,
    )

    handler(ctx, args)


if __name__ == "__main__":
    main()
