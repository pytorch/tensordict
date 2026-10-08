# Releasing TensorDict

This is the release procedure, written for an AI agent or a maintainer. The
maintainer says "release X.Y.Z", reads one message with the release notes and
the commit to release, and replies "ok". The release then goes live without
more questions.

The procedure has three parts:

1. **Prepare, without asking.** Create the release branch, test it, push it,
   do a dry run of the Release workflow, and draft the release notes. None of
   this is public: pushing a branch publishes nothing.
2. **Ask once.** Show the release notes, the commit to release, the newest
   `main` commit whose fixes are included, what you left out and why, the test
   results and the dry run. Then wait for "ok".
3. **Publish after "ok".** Push the tag, create the draft release, start the
   real run, check PyPI and publish the GitHub release. Stop and report only
   if something fails.

## Facts to know first

- Each release has its own branch, `release/X.Y.Z`. A patch release branch
  starts from the previous tag of its line (for 0.14.4, from `v0.14.3`), not
  from `main`.
- The Release workflow (`.github/workflows/release.yml`) starts only by hand,
  from `release/X.Y.Z`. It refuses other refs, because
  `.github/scripts/version_script.sh` gives wheels built on other refs a dev
  version.
- A real run publishes to PyPI as soon as all wheels are built. There is no
  approval step. PyPI never accepts the same file twice, so a broken release
  can be yanked but not replaced.
- Pushing the tag `vX.Y.Z` is already a publication. The docs build of the tag
  publishes the `X.Y` docs at https://docs.pytorch.org/tensordict, and the
  conda-forge bot builds `conda-forge/tensordict-feedstock` from the tag's
  source archive within hours (for 0.14.3, before PyPI had the wheels). Never
  move or delete a pushed tag. A real run checks that the tag points to the
  commit that it builds.
- CI on a release branch runs the unit tests with torch nightly and the
  latest stable torch, plus any other torch version that
  `.github/workflows/test-linux.yml` of the branch lists. Code paths for older
  torch versions are tested only if you run them, as in step 3.
- A dry run and a real run take about 45 minutes each, because the four
  platforms build one after another.

## 1. Choose the inputs

```bash
VERSION=0.14.4   # the version to release
PREV=v0.14.3     # the latest release of the same line; for X.Y.0, the latest X.(Y-1) release
```

**Python versions.** Build the same Python versions as the previous release,
unless the maintainer decides otherwise. For a minor release, propose any
change (a new CPython version, or one that stable torch dropped) in the message
of step 6 instead of making it silently.

```bash
PYTHON_VERSIONS=$(curl -fsS "https://pypi.org/pypi/tensordict/${PREV#v}/json" | python3 -c '
import json, re, sys
abis = {re.search(r"-cp\d+t?-cp(\d)(\d+)(t?)-", u["filename"]).groups() for u in json.load(sys.stdin)["urls"] if u["filename"].endswith(".whl")}
print(json.dumps([f"{a}.{b}{t}" for a, b, t in sorted(abis, key=lambda x: (int(x[0]), int(x[1]), x[2]))]))')
echo "$PYTHON_VERSIONS"   # ["3.10", "3.11", "3.12", "3.13", "3.14", "3.14t"] for 0.14.3
```

**test-infra branch (`pytorch_release`).** Use the pytorch/test-infra branch
of the latest stable torch:

```bash
TORCH_STABLE=$(curl -fsS https://pypi.org/pypi/torch/json | python3 -c 'import json, sys; print(json.load(sys.stdin)["info"]["version"])')
PYTORCH_RELEASE="release/${TORCH_STABLE%.*}"   # 2.14.1 -> release/2.14
gh api "repos/pytorch/test-infra/contents/tools/scripts/generate_binary_build_matrix.py?ref=$PYTORCH_RELEASE" \
  --jq .content | base64 -d | grep '^CURRENT_CANDIDATE_VERSION'   # the torch that the builds install
```

On a release branch, the builds install the torch version that the test-infra
branch sets as `CURRENT_CANDIDATE_VERSION`, and that torch needs a wheel for
every Python version that you build. On test-infra `main`, it is the next
PyTorch release, which can drop Python versions: with `main`, the 0.14.3
builds installed torch 2.15.0, which has no Python 3.10 wheels, and they
failed. For a minor release that comes out together with a new PyTorch
release, use that release's branch.

## 2. Prepare the release branch

### Patch release (X.Y.Z with Z > 0)

```bash
git fetch upstream --tags
git worktree add -b "release/$VERSION" "../tensordict-release-$VERSION" "$PREV"
cd "../tensordict-release-$VERSION"
```

List the `main` commits that the line does not have yet, oldest first:

```bash
LINE_BASE=$(git merge-base upstream/main "$PREV")
ON_LINE=$(git log --format=%s "$LINE_BASE..HEAD" | grep -oE '#[0-9]+' | sort -u)
git log --reverse --format='%h %s' "$LINE_BASE..upstream/main" | while read -r sha subject; do
  pr=$(grep -oE '#[0-9]+\)$' <<<"$subject" | tr -d ')')
  grep -qxF -- "$pr" <<<"$ON_LINE" || echo "$sha $subject"
done
```

Choose the commits with these rules:

- Take every `[BugFix]` and `[Compile]` fix that changes code the line has.
  Cherry-pick them in the order of `main`, with `git cherry-pick` and without
  `-x`. The original author stays the author.
- Leave out `[Feature]`, `[Deprecation]`, `[Performance]`, `[Versioning]`,
  `[Doc]`, `[BE]`, `[CI]` and `[Test]` commits, unless the maintainer asks for
  one, or a fix that you take, or the CI of the release branch, needs it. For
  0.14.3, the test change #1791 was taken because the tests of #1871 build on
  it.
- If a fix depends on code that only `main` has, take the part that applies,
  and say in the commit message what was left out (#1841 and #1880 on
  `release/0.14.3`). If no clean part is left, leave the fix out.
- Leave out a commit whose code change is already on the branch through
  another commit. #1857 was left out of 0.14.3 because #1871 had already made
  the same change.
- Keep a list of the commits that you leave out, with a reason for each. It
  goes into the message of step 6.

### Minor release (X.Y.0)

Create `release/X.Y.0` from the `main` commit that the maintainer names, or
else from the head of `main` once its CI passes. After the release, open a PR
that sets the four version files on `main` to X.Y.0. They set the base version
of the nightly builds.

### Bump the version

All four files must have the new version. The Release workflow checks them.

```bash
echo "$VERSION" > version.txt
echo "$VERSION" > .github/scripts/version.txt
sed -i "s/^BASE_VERSION=.*/BASE_VERSION=$VERSION/" .github/scripts/version_script.sh .github/scripts/version_script_windows.sh
```

If the `release.yml` of the branch has no `python_versions` input, as on the
0.14 line, take the workflow from `main` in the same commit. It passes
`python-versions` and `test-infra-ref` to the `build-wheels-*.yml` workflows of
the branch, which take both inputs on the 0.14 line.

```bash
grep -q 'python_versions:' .github/workflows/release.yml || git checkout upstream/main -- .github/workflows/release.yml
git commit -am "[Versioning] Prepare TensorDict $VERSION"
```

## 3. Test the branch

**Lint** the changed files with the tools and versions that the branch's
`.pre-commit-config.yaml` pins. `pre-commit` can pick a Python on which libcst
crashes (exit code -11): ufmt and torchfix use it. These commands avoid that.
On a branch whose hooks run ruff (TensorDict 0.15 and later):

```bash
files=$(git diff --name-only "$PREV..HEAD" -- '*.py')
uvx ruff@0.16.10 check $files
uvx ruff@0.16.10 format --check $files
uvx --python 3.12 --with torchfix==0.5.0 flake8==7.1.0 --select=TOR \
  --per-file-ignores='test_*.py:TOR101' $files
```

On a branch whose hooks run ufmt and flake8 (the 0.14 line):

```bash
files=$(git diff --name-only "$PREV..HEAD" -- '*.py')
uvx --python 3.12 --with flake8-bugbear==22.10.27 --with flake8-comprehensions==3.10.1 \
  --with torchfix==0.5.0 --with flake8-print==5.0.0 flake8==7.1.0 $files
uvx --python 3.11 --with black==24.4.2 --with usort==1.0.3 --with libcst==0.4.7 ufmt==2.7.0 check $files
```

**Run the test suite** on the latest stable torch and on an old torch that
the line still supports. For a line without a minimum torch version, such as
0.14.x, take the oldest torch that its users pin (torch 2.11 for 0.14.3).
Code for older torch versions is never tested in CI: on the 0.14 line, torch
2.13 and older use a fallback implementation of `UnbatchedTensor`, and a fix
picked for 0.14.3 had broken it.

Run the same suite on `$PREV` in the same environment. Every test that fails
on the branch but passes on `$PREV` needs an explanation before step 6. To
test `$PREV` with the venvs below, check it out in a second worktree, copy
`tensordict/_C.so` into it if `tensordict/csrc` is unchanged, and run pytest
there with `PYTHONPATH` set to that worktree. To find the commit that caused
a failure, test the cherry-picks one by one, for example with `git bisect`
over `$PREV..HEAD`.

```bash
TORCH=2.14.1   # then the old torch, e.g. 2.11.0
V=~/.cache/tensordict-release/venv-$TORCH
uv venv --python 3.12 "$V"
uv pip install --python "$V/bin/python" "torch==$TORCH" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$V/bin/python" numpy cloudpickle packaging importlib_metadata orjson "pyvers>=0.2,<0.3" \
  pytest pytest-xdist pytest-timeout pytest-rerunfailures pytest-instafail pytest-benchmark pytest-mock pyyaml \
  hypothesis expecttest h5py pandas pyarrow redis zarr setuptools wheel "pybind11[global]>=2.13" setuptools_scm cmake ninja
uv pip install --python "$V/bin/python" --no-deps --no-build-isolation -e .

# Redis and Dragonfly, for test/store and test/tensorclass/test_compatibility.py
docker run -d --rm --name td-release-redis -p 127.0.0.1:6379:6379 redis:7-alpine --save ""
docker run -d --rm --name td-release-dragonfly -p 127.0.0.1:6380:6380 --ulimit memlock=-1 \
  docker.dragonflydb.io/dragonflydb/dragonfly:v1.27.1 --port 6380 --dbfilename ""

TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1 TD_GET_DEFAULTS_TO_NONE=1 LIST_TO_STACK=1 PYTORCH_TEST_WITH_SLOW=1 \
  "$V/bin/python" -m pytest --runslow -n 8 --timeout 600 -p no:cacheprovider
```

What to expect:

- `test/utils/test_setup.py` builds tensordict in new environments. Its
  install tests fail when the machine has no complete build toolchain.
- The distributed tests use the fixed port 10017, and the store tests use the
  ports 6379 and 6380. Run only one test suite per machine at a time. On a
  shared machine, run `test/distributed` in its own network namespace, for
  example in `docker run --network none` with the venv and the source tree
  mounted.
- Multiprocessing and distributed tests can time out when the machine is
  heavily loaded. Run such a failure again on its own before you treat it as
  a regression.
- Keep the venvs outside `/tmp`, which other jobs can clean up.

## 4. Push the branch and do a dry run

```bash
git push upstream "release/$VERSION"
gh workflow run release.yml -R pytorch/tensordict --ref "release/$VERSION" \
  -f tag="v$VERSION" -f dry_run=true \
  -f python_versions="$PYTHON_VERSIONS" -f pytorch_release="$PYTORCH_RELEASE"
gh run list -R pytorch/tensordict --workflow release.yml --branch "release/$VERSION"   # the run id
```

The push also starts the CI of the branch: lint, the unit tests and a docs
build that publishes nothing. Wait for the CI and for the dry run. In the
"Collect Wheels" job of the dry run, check that there is one wheel per Python
version and platform, all with the new version. 0.14.3 had 24: six Python
versions on Linux x86_64, Linux aarch64, macOS arm64 and Windows.

If the Linux builds fail at "Install torch dependency", the torch version of
`pytorch_release` has no wheel for one of the Python versions (see step 1).

From now on the branch is public. Add new commits to it, but don't rewrite
it.

## 5. Draft the release notes

Write the notes to a file outside the repository, for example
`../tensordict-v$VERSION-release-notes.md`, and don't commit them. The PR
descriptions are the source; their Summary and Behaviour changes sections say
what changed for users. To collect them:

```bash
for pr in $(git log --format=%s "$PREV..HEAD" | grep -oE '#[0-9]+\)$' | tr -d '#)' | sort -n -u); do
  gh pr view "$pr" -R pytorch/tensordict --json number,title,author,closingIssuesReferences,body \
    --jq '"## #\(.number) \(.title) (@\(.author.login)), fixes: \([.closingIssuesReferences[].number] | join(", "))\n\n\(.body)\n"'
done > /tmp/tensordict-release-prs.md
```

A contributor's first contribution is one where they had no merged PR before
the previous release:

```bash
gh api -X GET search/issues --jq .total_count \
  -f q="repo:pytorch/tensordict is:pr is:merged author:LOGIN merged:<$(git log -1 --format=%cs "$PREV")"
```

Use this format for a patch release, as in v0.14.3:

````markdown
## Highlights

TensorDict X.Y.Z is a patch release with N bug fixes. They cover <areas>. There are no new features or deprecations. Several fixes change behavior that was previously wrong; see "Behavior changes" below.

## Bug Fixes

### <Area>

- <What works now, and what happened before, in the user's terms> ([#PR](https://github.com/pytorch/tensordict/pull/PR), fixes [#ISSUE](https://github.com/pytorch/tensordict/issues/ISSUE)).

## Behavior changes

These fixes can change the behavior of existing code:

- <What existing code now sees, for example a new error> ([#PR](https://github.com/pytorch/tensordict/pull/PR)).

## Installation

```bash
pip install tensordict==X.Y.Z
```

## Contributors

Thanks to @login1 (first contribution), @login2, and @login3 for their contributions to this release.

**Full Changelog:** https://github.com/pytorch/tensordict/compare/vPREV...vX.Y.Z
````

- Write one bullet per fix. Group the bullets by area, for example Indexing;
  Memory-mapped tensordicts and archives; TensorDictStore; UnbatchedTensor,
  non-tensor data and TypedTensorDict; tensorclasses; tensordict.nn; Other
  fixes.
- Put a follow-up fix into the bullet of the fix that it completes (#1792 and
  #1875 in 0.14.3).
- If a fix was adapted for the branch, describe what the branch has.
- Leave test and CI commits out of the notes.
- Leave out the "Behavior changes" section if there are none, and drop the
  last sentence of the Highlights.
- List the authors of the PRs in the notes in alphabetical order, and mark
  first contributions.
- Put any announcement that the maintainer asks for into its own section
  after the Highlights. For example, 0.14.3 announced that an upcoming release
  will require `torch>=2.13`.
- For a minor release, add the sections "Breaking Changes", "Deprecations",
  "Features" and "Performance" when they have entries, and give the target
  version of each deprecation. The changelog link compares with
  `vX.(Y-1).0`.
- No emojis.

## 6. Ask the maintainer

First fetch `main` again. If fixes were merged after you chose the commits,
add them to the branch, test them, and do the dry run again.

Then send one message with:

- the release branch and its head commit, for example `release/0.14.3` at
  `6fdce72af` (`[BugFix] Fix TensorClass fields named after TensorClass
  methods (#1851)`);
- the newest `main` commit whose fix is included;
- the PRs that are included, and the commits that you left out, each with its
  reason;
- the test results: which tests failed, and whether they also fail on
  `$PREV`;
- the link to the dry run, and its number of wheels;
- the release notes, in full;
- "Reply ok to publish."

Don't push the tag or start the real run before the reply. If the maintainer
asks for changes, make them on the branch with new commits, run again what the
changes affect, and ask again.

## 7. Publish after "ok"

Release exactly the commit that the maintainer confirmed. If the branch has
moved since then, stop and ask.

```bash
SHA=$(git rev-parse <the confirmed commit>)
git fetch upstream
[ "$(git rev-parse "upstream/release/$VERSION")" = "$SHA" ] &&
  git tag -a "v$VERSION" -m "TensorDict v$VERSION" "$SHA" &&
  git push upstream "v$VERSION"
gh release create "v$VERSION" -R pytorch/tensordict --draft --target "release/$VERSION" \
  --title "TensorDict v$VERSION" --notes-file "../tensordict-v$VERSION-release-notes.md"
gh workflow run release.yml -R pytorch/tensordict --ref "release/$VERSION" \
  -f tag="v$VERSION" -f dry_run=false \
  -f python_versions="$PYTHON_VERSIONS" -f pytorch_release="$PYTORCH_RELEASE"
```

The URL of the draft release contains `untagged-...` until the release is
published. When the run succeeds, it has attached the wheels to the draft
release and published them to PyPI. Check that PyPI has the same wheels as
the dry run, then install the release in a new environment:

```bash
curl -fsS "https://pypi.org/pypi/tensordict/$VERSION/json" | python3 -c 'import json, sys; print("\n".join(sorted(u["filename"] for u in json.load(sys.stdin)["urls"])))'
uv venv --python 3.12 ~/.cache/tensordict-release/check
uv pip install --python ~/.cache/tensordict-release/check/bin/python "torch==$TORCH_STABLE" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python ~/.cache/tensordict-release/check/bin/python "tensordict==$VERSION"
~/.cache/tensordict-release/check/bin/python -c "import tensordict; print(tensordict.__version__)"
```

Then publish the GitHub release. `--latest` makes it the latest release of
the repository; leave it out for a release of an older line.

```bash
gh release edit "v$VERSION" -R pytorch/tensordict --draft=false --latest
```

Report the PyPI and GitHub links. conda-forge needs nothing: its bot opens the
feedstock update and merges it.

## If something fails

- **Before the tag is pushed:** fix it on the branch with new commits, test
  again, and do the dry run again.
- **After the tag is pushed:** if other inputs fix the problem, for example
  another `pytorch_release`, start the real run again. If the code must
  change, stop and ask the maintainer: the tag is already public, so the fix
  may need a new version.
- **PyPI has some of the files:** re-run the failed "Publish to PyPI" job from
  the page of the run. It skips the files that PyPI already has. A new run for
  a version that PyPI has stops in the sanity checks.
- **PyPI has the release:** never upload it again. If it is broken, the
  maintainer can yank it on PyPI, and the fix goes into the next patch
  release.
- **Stopping a run:** `gh run cancel <run id> -R pytorch/tensordict`. A second
  run on the same branch waits for the first one instead of cancelling it.
- **The docs build of the tag fails:** re-run it from its run page. For a new
  minor version, the "Update Stable Docs" job of the real run needs the `X.Y`
  folder that this build creates on `gh-pages`.

## One-time setup (done)

PyPI trusts `pytorch/tensordict`, workflow `release.yml`, environment `pypi`,
as a trusted publisher, so the workflow needs no PyPI token. To check or
change this, go to https://pypi.org/manage/project/tensordict/settings/publishing/.
The `pypi` environment of the repository has no protection rules.
