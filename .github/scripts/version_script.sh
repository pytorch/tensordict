#!/bin/bash

# Only set static version for release branches and release candidate tags
if [[ "$GITHUB_REF_TYPE" == "branch" && "$GITHUB_REF_NAME" == release/* ]] || [[ "$GITHUB_REF_TYPE" == "tag" && "$GITHUB_REF_NAME" =~ ^v[0-9]+\.[0-9]+\.[0-9]+-rc[0-9]+$ ]]; then
    echo "Setting static version for release: $GITHUB_REF_NAME"
    export TENSORDICT_BUILD_VERSION=0.10.0
    export SETUPTOOLS_SCM_PRETEND_VERSION=$TENSORDICT_BUILD_VERSION
else
    echo "Using dynamic versioning for development build: $GITHUB_REF_NAME"
    # Ensure the variable is unset for dynamic versioning
    unset SETUPTOOLS_SCM_PRETEND_VERSION
fi

# TODO: consider lower this
export MACOSX_DEPLOYMENT_TARGET=14.0

${CONDA_RUN} pip install --upgrade pip

# for orjson
export UNSAFE_PYO3_BUILD_FREE_THREADED=1

${CONDA_RUN} conda install -c conda-forge pybind11 -y
