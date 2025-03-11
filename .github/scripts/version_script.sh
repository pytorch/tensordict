#!/bin/bash

export TENSORDICT_BUILD_VERSION=0.7.0

${CONDA_RUN} install conda-forge::pybind11 -y
