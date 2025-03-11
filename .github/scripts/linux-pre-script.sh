#!/bin/bash

yum update gcc
yum update libstdc++

export export CMAKE_PREFIX_PATH=$CONDA_PREFIX:$CMAKE_PREFIX_PATH
${CONDA_RUN} python -m pip install cmake pybind11 -U
