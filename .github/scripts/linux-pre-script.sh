#!/bin/bash

if [ "$(uname)" != "Darwin" ]; then
  yum update gcc
  yum update libstdc++
else
  # For OSX
  export CXXFLAGS="-march=armv8-a+fp16+sha3"
fi

${CONDA_RUN} conda install -c conda-forge pybind11 -y
