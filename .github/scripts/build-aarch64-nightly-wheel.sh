#!/bin/bash

set -euo pipefail

sed -i.bak 's/name = "tensordict"/name = "tensordict-nightly"/' pyproject.toml
trap 'mv pyproject.toml.bak pyproject.toml' EXIT

# Dependencies, including nightly Torch, are installed by test-infra.
SETUPTOOLS_SCM_PRETEND_VERSION="$(date +%Y.%m.%d)" python -m pip wheel --no-deps .
mkdir -p dist
mv tensordict_nightly-*.whl dist/
