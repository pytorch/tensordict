#!/bin/bash

export TENSORDICT_BUILD_VERSION=0.7.0

${CONDA_RUN} python -m pip install cmake pybind11 -U
