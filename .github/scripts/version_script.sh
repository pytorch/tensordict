#!/bin/bash

export TENSORDICT_BUILD_VERSION=0.7.0
export export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH

${CONDA_RUN} python -m pip install cmake pybind11 -U
