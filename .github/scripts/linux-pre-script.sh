#!/bin/bash

if [ "$(uname)" != "Darwin" ]; then
  yum update gcc
  yum update libstdc++
else
  brew update
  brew upgrade gcc

  # For OSX
  export CXXFLAGS="-march=armv8-a+fp16+sha3"
  export CMAKE_OSX_ARCHITECTURES=arm64
fi

${CONDA_RUN} conda install -c conda-forge pybind11 -y
