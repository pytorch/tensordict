#!/bin/bash

yum update gcc
yum update libstdc++

${CONDA_RUN} install conda-forge::pybind11 -y
