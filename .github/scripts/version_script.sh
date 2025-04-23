#!/bin/bash

export TENSORDICT_BUILD_VERSION=0.8.0
${CONDA_RUN} pip install --upgrade pip

${CONDA_RUN} conda install conda-forge::rust -y
# for orjson
export UNSAFE_PYO3_BUILD_FREE_THREADED=1
