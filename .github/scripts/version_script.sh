#!/bin/bash

export TENSORDICT_BUILD_VERSION=0.8.2
export SETUPTOOLS_SCM_PRETEND_VERSION=$TENSORDICT_BUILD_VERSION
# TODO: consider lower this
export MACOSX_DEPLOYMENT_TARGET=15.0

${CONDA_RUN} pip install --upgrade pip

# for orjson
export UNSAFE_PYO3_BUILD_FREE_THREADED=1

${CONDA_RUN} conda install -c conda-forge pybind11 -y
