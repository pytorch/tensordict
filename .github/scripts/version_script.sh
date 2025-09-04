#!/bin/bash

# Read base version from version.txt
BASE_VERSION=$(cat version.txt | head -1 | tr -d '\n\r')
echo "Base version from version.txt: $BASE_VERSION"

# Only set static version for release branches and release candidate tags
if [[ "$GITHUB_REF_TYPE" == "branch" && "$GITHUB_REF_NAME" == release/* ]] || [[ "$GITHUB_REF_TYPE" == "tag" && "$GITHUB_REF_NAME" =~ ^v[0-9]+\.[0-9]+\.[0-9]+-rc[0-9]+$ ]]; then
    echo "Setting static version for release: $GITHUB_REF_NAME"
    export TENSORDICT_BUILD_VERSION=$BASE_VERSION
    export SETUPTOOLS_SCM_PRETEND_VERSION=$TENSORDICT_BUILD_VERSION
else
    echo "Setting development version for build: $GITHUB_REF_NAME"
    # Get git info for development version
    GIT_COMMIT=$(git rev-parse --short HEAD)
    GIT_COMMIT_COUNT=$(git rev-list --count HEAD)
    DATE_STR=$(date +%Y%m%d)
    
    # Format: <base_version>.dev<commits>+g<hash>.d<date>
    DEV_VERSION="${BASE_VERSION}.dev${GIT_COMMIT_COUNT}+g${GIT_COMMIT}.d${DATE_STR}"
    echo "Using development version: $DEV_VERSION"
    export SETUPTOOLS_SCM_PRETEND_VERSION=$DEV_VERSION
fi

# TODO: consider lower this
export MACOSX_DEPLOYMENT_TARGET=14.0

${CONDA_RUN} pip install --upgrade pip

# for orjson
export UNSAFE_PYO3_BUILD_FREE_THREADED=1

${CONDA_RUN} conda install -c conda-forge pybind11 -y
