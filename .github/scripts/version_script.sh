#!/bin/bash

export TENSORDICT_BUILD_VERSION=0.8.0

${CONDA_RUN} install conda-forge::pybind11 -y
