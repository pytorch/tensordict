#!/bin/bash

yum update gcc
yum update libstdc++

${CONDA_RUN} python -m pip install cmake pybind11 -U
