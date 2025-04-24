#!/bin/bash

#if [ "$(uname)" != "Darwin" ]; then
#  yum update gcc
#  yum update libstdc++
#else
#  echo $(gcc --version)
#  echo $(clang --version)
#  brew update
#  brew upgrade gcc
#  brew upgrade clang
#
#  # For OSX
##  export CXXFLAGS="-march=armv8-a+fp16+sha3"
#  export CMAKE_OSX_ARCHITECTURES=arm64
#fi

${CONDA_RUN} conda install -c conda-forge pybind11 -y
