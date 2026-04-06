#!/bin/bash

${CONDA_RUN} conda install -c conda-forge pybind11 -y
# setuptools>=82 removed pkg_resources and no longer vendors `packaging`;
# PyTorch imports `from packaging.version import Version` at init time,
# so the standalone package must be present.
${CONDA_RUN} pip install setuptools_scm packaging
