# TensorDict Release Agent Prompt

This document provides instructions for an AI assistant to help automate TensorDict releases.

## Prerequisites

Before starting a release, ensure you have:
- Write access to the repository
- Ability to create branches, tags, and pull requests
- Access to view GitHub Actions workflow runs

## Input Parameters

Collect the following information from the user:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `version_tag` | The version to release | `v0.11.0` |
| `release_type` | Major (0.x.0) or minor (0.x.y) release | `major` or `minor` |
| `pytorch_release` | PyTorch release branch to build against | `release/2.8` |
| `previous_version` | Previous release tag (for release notes) | `v0.10.0` |

---

## Step 1: Analyze Commits for Release Notes

### For Major Releases (e.g., 0.10.x → 0.11.0)

Get commits from the last major release:

```bash
# Find the last major release tag
git fetch --tags
git log v0.10.0..HEAD --oneline --no-merges
```

### For Minor Releases (e.g., 0.11.0 → 0.11.1)

Get commits from the last release:

```bash
git log v0.11.0..HEAD --oneline --no-merges
```

### Categorize Commits

Review each commit and categorize into:

1. **Features** - New functionality
   - Look for: "add", "implement", "support", "new"

2. **Bug Fixes** - Issue corrections
   - Look for: "fix", "bug", "issue", "patch", "correct"

3. **Breaking Changes** - API changes that may break existing code
   - Look for: "breaking", "remove", "deprecate", "rename" (of public APIs)
   - Check for removed or renamed public functions/classes
   - Check for changed function signatures

4. **Deprecations** - Features marked for future removal
   - Look for: "deprecate", "warn"

5. **Performance** - Speed or memory improvements
   - Look for: "perf", "optimize", "speed", "memory", "faster"

6. **Documentation** - Doc improvements (usually not included in notes)

7. **CI/Infrastructure** - Build/test changes (usually not included in notes)

### Generate Human-Readable Summaries

Transform commit messages into user-friendly descriptions:

**Bad:** `fix: handle edge case in TensorDict.to_dict() when nested (#1234)`
**Good:** `Fixed an issue where TensorDict.to_dict() would fail with deeply nested structures`

**Bad:** `feat: add support for bf16 in memmap (#1235)`
**Good:** `Added bfloat16 support for memory-mapped tensors`

---

## Step 2: Draft Release Notes

Present the following template to the user for review:

```markdown
## TensorDict {version_tag}

### Highlights

<!-- 2-3 sentence summary of the most important changes -->

### Breaking Changes

<!-- List any breaking changes. If none, write "No breaking changes in this release." -->

### Deprecations

<!-- List any new deprecations. If none, remove this section. -->

### Features

- Feature description ([#PR_NUMBER](link))
- ...

### Bug Fixes

- Fix description ([#PR_NUMBER](link))
- ...

### Performance

- Performance improvement description ([#PR_NUMBER](link))
- ...

### Contributors

Thanks to all contributors:
<!-- List first-time contributors especially -->
```

**Important:** Wait for user approval before proceeding.

---

## Step 3: Update Version Files

Update version in all required locations:

### 1. Root version.txt

```bash
echo "{version_without_v}" > version.txt
```

### 2. GitHub Scripts version.txt

```bash
echo "{version_without_v}" > .github/scripts/version.txt
```

### 3. version_script.sh

Update `BASE_VERSION` on line 6:

```bash
sed -i 's/^BASE_VERSION=.*/BASE_VERSION={version_without_v}/' .github/scripts/version_script.sh
```

### 4. version_script_windows.sh (if it has a version)

Check and update if necessary.

---

## Step 4: Create Release Branch

```bash
# Create release branch from main
git checkout main
git pull origin main
git checkout -b release/{major.minor}

# Example: for v0.11.0
git checkout -b release/0.11
```

---

## Step 5: Commit Version Changes

```bash
git add version.txt .github/scripts/version.txt .github/scripts/version_script.sh
git commit -m "Bump version to {version_without_v}"
```

---

## Step 6: Create and Push Tag

```bash
# Create annotated tag
git tag -a {version_tag} -m "TensorDict {version_tag}"

# Push branch and tag
git push origin release/{major.minor}
git push origin {version_tag}
```

---

## Step 7: Trigger Release Workflow

The push of the tag will automatically trigger `.github/workflows/release.yml`.

To manually trigger with specific options:

1. Go to Actions → Release workflow
2. Click "Run workflow"
3. Fill in:
   - `tag`: The version tag (e.g., `v0.11.0`)
   - `pytorch_release`: The PyTorch branch (e.g., `release/2.8`)
   - `dry_run`: Check for testing without publishing

---

## Step 8: Create Draft GitHub Release

If not using the automated workflow, create manually:

```bash
gh release create {version_tag} \
  --draft \
  --title "TensorDict {version_tag}" \
  --notes-file RELEASE_NOTES.md
```

---

## Step 9: Monitor Workflow

Watch the release workflow for:

1. **Sanity checks** - Verify all version files match
2. **Wheel builds** - All platforms should succeed
3. **Docs update** - Stable symlink updated
4. **Release creation** - Draft created with wheels

---

## Post-Release Manual Steps

### 1. Review Draft Release Notes

- Go to GitHub Releases
- Find the draft release for `{version_tag}`
- Edit the release notes with the content prepared in Step 2
- Add any last-minute changes or acknowledgments

### 2. Publish the Release

- Once satisfied with release notes, click "Publish release"
- This makes the release public and notifies watchers

### 3. Approve PyPI Publishing

- Go to the release workflow run
- Find the "Publish to PyPI" job waiting for approval
- Review the wheels to be published
- Approve the environment to proceed

### 4. Verify PyPI Upload

```bash
# Check that the package is available
pip index versions tensordict

# Or install and verify
pip install tensordict=={version_without_v}
python -c "import tensordict; print(tensordict.__version__)"
```

### 5. Verify Documentation

- Check https://pytorch.org/tensordict/stable/ points to the new version
- Verify the version selector includes the new version

### 6. Announce the Release

Consider announcing on:
- PyTorch forums
- Twitter/X
- Discord/Slack channels
- Project mailing lists

---

## Troubleshooting

### Version Mismatch Errors

If sanity checks fail due to version mismatch:
```bash
# Check all version files
cat version.txt
cat .github/scripts/version.txt
grep BASE_VERSION .github/scripts/version_script.sh
```

### Wheel Build Failures

Check the individual build workflow logs. Common issues:
- PyTorch version compatibility
- Missing dependencies
- Platform-specific compilation errors

### Docs Update Failure

If the version folder doesn't exist on gh-pages:
1. Trigger the docs workflow manually for the release tag
2. Wait for docs to build and upload
3. Re-run the release workflow or manually update stable symlink

### PyPI Upload Failure

If OIDC authentication fails:
1. Verify the repository is configured as a trusted publisher on PyPI
2. Check the workflow permissions include `id-token: write`
3. Ensure the environment name matches PyPI configuration

---

## Environment Setup for PyPI Trusted Publishing

Before the first release, configure PyPI trusted publishing:

1. Go to https://pypi.org/manage/project/tensordict/settings/publishing/
2. Add a new trusted publisher:
   - Owner: `pytorch`
   - Repository: `tensordict`
   - Workflow name: `release.yml`
   - Environment name: `pypi`

---

## Quick Reference Commands

```bash
# Check current version
cat version.txt

# List recent tags
git tag --sort=-creatordate | head -10

# View commits since last release
git log $(git describe --tags --abbrev=0)..HEAD --oneline

# Verify tag exists and is annotated
git show {version_tag}

# Check workflow status
gh run list --workflow=release.yml

# Download release artifacts
gh release download {version_tag} --dir ./release-artifacts
```
