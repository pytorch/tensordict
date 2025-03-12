#!/bin/bash

export TENSORDICT_BUILD_VERSION=0.8.0

if [ "$(uname)" == "Darwin" ]; then
  # For OSX
  export CXXFLAGS="-march=armv8-a+fp16+sha3"
  export CMAKE_OSX_ARCHITECTURES=arm64
fi

${CONDA_RUN} conda install -c conda-forge pybind11 -y
