#!/bin/bash

yum update gcc
yum update libstdc++

${CONDA_RUN} conda install -c conda-forge pybind11 -y
