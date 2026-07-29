#!/usr/bin/env bash

set -euo pipefail

sed -i.bak 's/name = "tensordict"/name = "tensordict-nightly"/' pyproject.toml
trap 'rm -f pyproject.toml.bak' EXIT

export SETUPTOOLS_SCM_PRETEND_VERSION
SETUPTOOLS_SCM_PRETEND_VERSION=$(date +%Y.%m.%d)

pip wheel .
mkdir -p dist
mv tensordict_nightly-*.whl dist/
