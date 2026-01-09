#!/usr/bin/env bash

set -e

# Activate the environment
if [ "${PYTHON_VERSION}" == "3.14t" ]; then
    source ./env/bin/activate
else
    eval "$(./conda/bin/conda shell.bash hook)"
    conda activate ./env
fi
