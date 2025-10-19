#!/bin/bash

# Read base version from version.txt (explicit absolute path)
# This script is located at: .github/scripts/version_script.sh
# The version.txt file is at: .github/scripts/version.txt
BASE_VERSION=0.10.0
echo "Base version: $BASE_VERSION"

# Only set static version for release branches and release candidate tags
if [[ "$GITHUB_REF_TYPE" == "branch" && "$GITHUB_REF_NAME" == release/* ]] || [[ "$GITHUB_REF_TYPE" == "tag" && "$GITHUB_REF_NAME" =~ ^v[0-9]+\.[0-9]+\.[0-9]+-rc[0-9]+$ ]]; then
    echo "Setting static version for release: $GITHUB_REF_NAME"
    export TENSORDICT_BUILD_VERSION=$BASE_VERSION
    export SETUPTOOLS_SCM_PRETEND_VERSION=$TENSORDICT_BUILD_VERSION
else
    echo "Setting development version for build: $GITHUB_REF_NAME"

    # Debug: Print available environment variables
    echo "Debug environment variables:"
    echo "  GITHUB_SHA: ${GITHUB_SHA:-not set}"
    echo "  GITHUB_REF_TYPE: ${GITHUB_REF_TYPE:-not set}"
    echo "  GITHUB_REF_NAME: ${GITHUB_REF_NAME:-not set}"
    echo "  GITHUB_RUN_NUMBER: ${GITHUB_RUN_NUMBER:-not set}"
    echo "  GITHUB_RUN_ATTEMPT: ${GITHUB_RUN_ATTEMPT:-not set}"

    # Use environment variables instead of git commands
    GIT_COMMIT="${GITHUB_SHA:0:9}"  # First 9 chars of commit hash
    if [[ -z "$GIT_COMMIT" ]]; then
        GIT_COMMIT="unknown"
    fi

    # Use GitHub run number as substitute for commit count
    GIT_COMMIT_COUNT="${GITHUB_RUN_NUMBER:-0}"

    DATE_STR=$(date +%Y%m%d)

    # Format: <base_version>.dev<commits>+g<hash>.d<date>
    DEV_VERSION="${BASE_VERSION}.dev${GIT_COMMIT_COUNT}+g${GIT_COMMIT}.d${DATE_STR}"
    echo "Using development version: $DEV_VERSION"
    export SETUPTOOLS_SCM_PRETEND_VERSION=$DEV_VERSION
fi

# TODO: consider lower this
export MACOSX_DEPLOYMENT_TARGET=14.0

# Set CONDA_RUN if not set
if [[ -z "${CONDA_RUN:-}" ]]; then

    pip install --upgrade pip

    # for orjson
    export UNSAFE_PYO3_BUILD_FREE_THREADED=1

    pip install "pybind11[global]"

    # Install setuptools_scm which is required for building with --no-isolation
    # This is done here (not in pre-script) to avoid cache issues
    pip install setuptools_scm

else

    ${CONDA_RUN} pip install --upgrade pip

    # for orjson
    export UNSAFE_PYO3_BUILD_FREE_THREADED=1

    ${CONDA_RUN} conda install -c conda-forge pybind11 -y

    # Install setuptools_scm which is required for building with --no-isolation
    # This is done here (not in pre-script) to avoid cache issues
    ${CONDA_RUN} pip install setuptools_scm
fi
