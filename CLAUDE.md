# Contributing rules for AI agents (Claude, Codex, …)

House rules for LLM-driven contributions to `tensordict`. Sits on top of
[`CONTRIBUTING.md`](CONTRIBUTING.md); this file wins for AI-generated changes.

## Imports

- Module-top imports only. Two exceptions:
  - **Optional deps**: gate with `_has_<name> = importlib.util.find_spec("<name>") is not None`,
    then import lazily inside the function (or cache on `self._<name>`).
  - **Genuine circular imports**: try `from typing import TYPE_CHECKING` first.
- No wildcard imports. Add `from __future__ import annotations` to new `.py` files.

## Versioning, logging, timing

- Dispatch on dependency versions with `from pyvers import implement_for` —
  no hand-rolled `if torch.__version__ >= …`.
- No `print()` in library code. Use
  `from tensordict.utils import logger as tensordict_logger`.
- Time with `tensordict.utils.timeit`, not `time.time()`.

## Container model

`tensordict` *is* the container library — don't fork the abstraction:

- Extend `TensorDictBase` / `TensorDict` / `LazyStackedTensorDict` /
  `PersistentTensorDict` / `Tensorclass` / `TypedTensorDict` rather than
  introducing parallel dict-like types.
- Use `NestedKey` semantics for keys.

## `torch.compile` / cudagraphs

Strongly encouraged — many downstream projects compile through us. Prefer
`torch.where` over Python `if`/`else` on tensor values; avoid data-dependent
shapes and `.item()` on hot paths; keep dtypes/devices stable. For hot paths
(core ops, keys, batched ops, `nn`, memmap/store), verify under
`torch.compile` and where reasonable cudagraphs — extend `test/test_compile.py`
and `benchmarks/compile/` rather than inventing a new harness.

## Tests

Every new public class/function needs tests. **Don't create new test files**
when an existing `test/test_*.py` covers the area — extend it.

## Documentation

- Every new public class/function must be referenced in
  `docs/source/reference/{td,tc,nn,ttd}.rst`.
- Docstrings: Sphinx-style (`Args:` / `Returns:` / `Examples:`) with a runnable
  `>>>` example for every new public class. Cite paper + arXiv link if
  applicable.
- No emojis anywhere (code, docs, commits, PRs).

## Tutorials

Headline features (new container type, major `nn` component, new storage
backend) ship a tutorial under `tutorials/sphinx_tuto/`. Sphinx-first: prose
in `# comments`, not `print(...)`. Mirror existing tutos: "What you will
learn" / "Conclusion" / "Further reading" sections.

## Benchmarks

Performance-relevant changes (core ops, keys, indexing/stacking, `nn`,
memmap, store, compile) extend `benchmarks/`. Pure correctness fixes don't
need one.

## Type hints

Public signatures carry hints. They're documentary (not mypy-enforced) but
must be accurate — wrong hints are worse than none.

## Deprecations

Two minor releases of warning before any breaking change. If the next
release is `0.X`: deprecate in `0.X`, default changes in `0.(X+1)`, removal
in `0.(X+2)`. Use `DeprecationWarning` for removals, `FutureWarning` for
upcoming default changes, and **state the target version explicitly** in the
message.

## PR labels

Prefix the PR title with one of the canonical tags:

```
[BE] [Benchmark] [BugFix] [CI] [Compile] [Deprecation] [Distributed]
[Doc] [Feature] [Minor] [Performance] [Quality] [Refactor] [Test]
[Versioning]
```

Pick the most specific. `[Feature]` = new user-facing capability;
`[BE]` = internal plumbing; `[Quality]` = lint/typing/cleanup;
`[Refactor]` = behavior-preserving restructure; `[Compile]` = `torch.compile` /
cudagraphs work.

## When in doubt

Read a recently-merged PR in the same area and match its conventions.
