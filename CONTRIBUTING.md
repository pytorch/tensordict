# Contributing to tensordict

[![Nightly](https://github.com/pytorch/tensordict/actions/workflows/nightly_orchestrator.yml/badge.svg)](https://github.com/pytorch/tensordict/actions/workflows/nightly_orchestrator.yml)
[![Nightly Dashboard](https://img.shields.io/badge/Nightly-Dashboard-blue)](https://pytorch.github.io/tensordict/nightly-status/)
[![CPU Benchmarks](https://img.shields.io/badge/Benchmarks-CPU-blue)](https://pytorch.github.io/tensordict/dev/bench/cpu/)
[![GPU Benchmarks](https://img.shields.io/badge/Benchmarks-GPU-blue)](https://pytorch.github.io/tensordict/dev/bench/gpu/)

We want to make contributing to this project as easy and transparent as
possible.

[`ARCHITECTURE.md`](ARCHITECTURE.md) maps the code: what each module is for,
the class hierarchy, and the places a new operation touches.

## Installing the library
Install the library as suggested in the README. For advanced features,
it is preferable to install the nightly built of pytorch.

Make sure you install tensordict in develop mode by running
```
pip install -e .
```
in your shell. tensordict is pure Python, so this needs no compiler.

## Formatting your code
**Type annotation**

tensordict is not strongly-typed, i.e. we do not enforce type hints, neither do we check that the ones that are present are valid. We rely on type hints purely for documentary purposes. Although this might change in the future, there is currently no need for this to be enforced at the moment.

**Linting**

Before your PR is ready, you'll probably want your code to be checked. The
checks are [ruff](https://docs.astral.sh/ruff/) (formatting, linting and import
sorting, configured in `pyproject.toml`) and torchfix. They can be run easily by installing
```
pip install pre-commit
```
and running
```
pre-commit run --all-files
```
from within the tensordict cloned directory.

You can also install [pre-commit hooks](https://pre-commit.com/) (using `pre-commit install`
). You can disable the check by appending `-n` to your commit command: `git commit -m <commit message> -n`

## Writing docstrings

The API reference is built from the docstrings by Sphinx's napoleon extension,
in Google style. Its reader wants to know what a call does, what each argument
means, and what comes back.

- Start with one sentence that says what the call does or returns, in the
  third person: "Returns a new tensordict with ...", "Stacks ...".
- If needed, add a short paragraph on what a user must know: whether the result
  is a new object, a view, or `self`, and what happens in the cases users hit
  (a missing key, a locked tensordict, a lazy stack).
- Document every argument under `Args:`, and the keyword-only ones under
  `Keyword Args:`, as `name (type, optional): what it controls.`, and end the
  entry of an argument that has a default with `Defaults to ``value``.` Add
  `Returns:` when the first sentence does not say what comes back, and
  `Raises:` for the errors a user may want to catch.
- Give each class and each frequently used function an `Examples:` section
  that runs as copied: import what it uses, use fixed values or
  `torch.manual_seed`, and show the output of each expression.
- Write code and literals in double backticks (``` ``None`` ```): single
  backticks render in italics. Link other objects with `:class:`, `:meth:`
  and `:func:`, and the concept pages with `:ref:`.
- Keep it as short as the call allows. A property or an in-place twin needs one
  line; a class or a main entry point can need a page. Leave out how it is
  implemented, unless that changes what the user does.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

When submitting a PR, we encourage you to link it to the related issue (if any) and add some tags to it.

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to tensordict, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
