#!/bin/bash

export TENSORDICT_BUILD_VERSION=0.8.0
${CONDA_RUN} pip install --upgrade pip

${CONDA_RUN} conda install conda-forge::rust -y
# for orjson
export UNSAFE_PYO3_BUILD_FREE_THREADED=1

#if [ "$(uname)" == "Darwin" ]; then
#  # For OSX
#  echo $(gcc --version)
#  echo $(clang --version)
#  brew update
#  brew install gcc
#  brew install clang-build-analyzer
#  brew install --cask clay
#  brew install llvm
##  brew upgrade gcc
##  brew upgrade clang
##  export CXXFLAGS="-march=armv8-a+fp16+sha3"
#  export CMAKE_OSX_ARCHITECTURES=arm64
#fi

${CONDA_RUN} conda install -c conda-forge pybind11 -y
