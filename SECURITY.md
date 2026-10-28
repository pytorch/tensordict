# Security

## Report a vulnerability

Report suspected TensorDict vulnerabilities through
[GitHub's private vulnerability reporting channel](https://github.com/pytorch/tensordict/security/advisories/new).
Do not include vulnerability details in a public issue or pull request.

Include, when available:

- The affected TensorDict version and whether the latest stable release is affected.
- Python, PyTorch and relevant optional dependency versions, plus the operating system.
- A description of the affected API, input source and impact.
- Minimal reproduction steps or a small example, without credentials or private data.

## Versions and dependencies

Use the latest stable TensorDict release and patched versions of its dependencies.
Report the versions you tested, even if you cannot test the latest release.
This recommendation does not promise security backports for older releases.

Optional integrations and native libraries bundled with dependencies need separate
updates. Updating TensorDict alone does not update every installed dependency.
See [Load data that you did not write](https://docs.pytorch.org/tensordict/main/saving.html#safe-loading)
for pickle controls, trusted writers and peers, and dependency-specific guidance.
