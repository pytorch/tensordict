#!/bin/bash

${CONDA_RUN} conda install -c conda-forge pybind11 -y
# Install setuptools_scm which is required for building with --no-isolation
${CONDA_RUN} pip install setuptools_scm
