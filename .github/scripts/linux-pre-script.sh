#!/bin/bash

if [ "$(uname)" != "Darwin" ]; then
  yum update gcc
  yum update libstdc++
fi

${CONDA_RUN} conda install -c conda-forge pybind11 -y
