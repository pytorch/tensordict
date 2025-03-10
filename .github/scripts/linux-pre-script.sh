#!/bin/bash

yum update gcc
yum update libstdc++

${CONDA_RUN} pip install cmake pybind11
