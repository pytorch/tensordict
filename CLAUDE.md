# Contributing rules for AI agents (Claude, Codex, …)

These are the house rules for any LLM-driven contribution to `tensordict`. They
sit on top of [`CONTRIBUTING.md`](CONTRIBUTING.md), which still applies. When the
two disagree, this file wins for AI-generated changes.

Read this end-to-end before editing anything.

## 1. Imports

- **No local (function/method-level) imports.** Module-top imports only.
- Two exceptions, and only two:
  - **Optional dependencies.** Gate them with a module-level
    `_has_<name> = importlib.util.find_spec("<name>") is not None`, then import
    the lib lazily inside the function — or, preferred, cache it on
    `self._<name>` the first time it is needed so subsequent calls don't re-run
    `import`.
  - **Genuine circular imports.** Before deferring, try
    `from typing import TYPE_CHECKING` with a guarded import — that handles the
    type-annotation case without paying a runtime cost.
- **No wildcard imports** (`from x import *`). Remove them when you see them.
- **`from __future__ import annotations`** at the top of every new `.py` file.

## 2. Cross-version compatibility

- Use `implement_for` from `pyvers` to dispatch on dependency versions —
  torch, numpy, etc. Do not hand-roll `if torch.__version__ >= …` branches.

  ```python
  from pyvers import implement_for
  ```

## 3. Logging, printing, timing

- **Never use `print()`** in library code. Use the package logger:
  ```python
  from tensordict.utils import logger as tensordict_logger
  tensordict_logger.info("…")
  ```
- **For timing**, use `tensordict.utils.timeit` — never ad-hoc `time.time()`
  blocks.

## 4. Container model

`tensordict` *is* the container library, so the rule isn't "use TensorDict",
it's "don't fork the abstraction":

- New container-like behavior should extend `TensorDictBase` / `TensorDict` /
  `LazyStackedTensorDict` / `PersistentTensorDict` / `Tensorclass` /
  `TypedTensorDict`, not introduce parallel dict-like types.
- Stick to `NestedKey` semantics for keying; don't invent a parallel key
  representation.
- New tensorclass-style containers follow the existing `tensorclass` /
  `TypedTensorDict` patterns — match them rather than rolling your own.

## 5. `torch.compile` / cudagraphs friendliness

`tensordict` sits on the hot path of `torch.compile`-d code in many downstream
projects, so this matters more here than in most libraries. Not mandatory,
**strongly encouraged**:

- Prefer `torch.where(...)` and masking over Python-level `if`/`else` on tensor
  values.
- Avoid data-dependent shapes and `.item()` calls on hot paths.
- Keep tensor dtypes/devices stable across calls.

If a component touches a hot path (core `TensorDict` ops, key handling, batched
ops, `nn` modules, memmap / store), please verify it under `torch.compile` and,
where reasonable, under cudagraphs. `test/test_compile.py` and
`benchmarks/compile/` are the existing baselines — extend them rather than
inventing a new harness.

## 6. Tests

- **Every new public class / function needs tests.**
- **Do not create new test files** when an existing one already covers the
  module/area — extend the existing file (`test/test_tensordict.py`,
  `test/test_nn.py`, `test/test_tensorclass.py`, `test/test_compile.py`,
  `test/test_memmap.py`, `test/test_store.py`, …). New test files should be
  reserved for genuinely new areas of the library.

## 7. Documentation

- **Every new public class / function must be referenced** in the appropriate
  `docs/source/reference/*.rst` page (`td.rst`, `tc.rst`, `nn.rst`,
  `ttd.rst`). PRs that add a class but skip the `.rst` entry will be sent
  back.
- **Docstrings**: Sphinx-style (`Args:`, `Returns:`, `Examples:`) with a
  runnable `>>> …` example for every new public class.
- **Paper references**: if a feature is inspired by a paper, include the
  arXiv link (and a short citation) in the class docstring.
- **No emojis** anywhere — code, docstrings, comments, commit messages, PR
  bodies.

## 8. Tutorials

New "headline" features (a new container type, a major new `nn` component,
a new storage backend, etc.) should ship a tutorial under
`tutorials/sphinx_tuto/`, or extend an existing one. Take inspiration from the
tutorials already in the repo. Tutorials are **Sphinx-first**, not
script-first:

- Use `# regular prose comments` for explanation, **not** `print("…")`.
- Structure should mirror existing tutos, including (names can be rephrased):
  - a **"What you will learn"** section near the top,
  - a **"Conclusion"** section,
  - a **"Further reading"** section.

## 9. Benchmarks

If a change is performance-relevant — anything on a hot path (core
`TensorDict` ops, key handling, indexing/stacking, `nn` modules, memmap,
store, compile-related code) — add or extend a benchmark under `benchmarks/`.
"Performance-relevant" is the trigger; pure correctness fixes don't need one.

## 10. Type hints

- New public signatures should carry type hints. Internal helpers can skip
  them, but prefer to add them when convenient.
- Hints are documentary (we don't enforce them with mypy in CI — see
  `CONTRIBUTING.md`), but they must be **accurate** — wrong hints are worse
  than no hints.

## 11. Backwards compatibility & deprecations

We give users **two minor releases** of warning before any breaking change.
Concretely, if the next release is `0.X`:

- Deprecate in `0.X`,
- Default-value changes (if any) can land in `0.(X+1)`,
- Final removal / behavior switch in `0.(X+2)`.

Rules:

- Use `DeprecationWarning` for API removals; `FutureWarning` for upcoming
  default-value changes that users can already see.
- **Always state the schedule explicitly** in the warning message, naming the
  version where the change will happen, e.g.:
  > `TensorDict.foo` is deprecated and will be removed in v0.X+2. Use
  > `TensorDict.bar` instead.

## 12. PR labels

Use a `[Tag]` prefix on the PR title. Canonical set seen in the repo:

```
[BE] [Benchmark] [BugFix] [CI] [Compile] [Deprecation] [Distributed]
[Doc] [Feature] [Minor] [Performance] [Quality] [Refactor] [Test]
[Versioning]
```

Pick the most specific one. `[Feature]` for new user-facing capability,
`[BugFix]` for fixes, `[Doc]` for docs-only, `[CI]` for workflows,
`[BE]` for backend / internal plumbing, `[Performance]` for perf work,
`[Quality]` for lint/typing/cleanup, `[Refactor]` for behavior-preserving
restructure, `[Compile]` for `torch.compile` / cudagraphs work,
`[Distributed]` for distributed / multi-process plumbing,
`[Deprecation]` for deprecation warnings, `[Versioning]` for version bumps.

## 13. Commits

No squash requirement. Make commits that read sensibly on their own; that's
all.

## 14. When in doubt

Read a recently-merged PR in the same area and match the conventions there
before inventing your own.
