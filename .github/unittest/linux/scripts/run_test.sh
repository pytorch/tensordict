#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
export TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1
export TD_GET_DEFAULTS_TO_NONE=1
export LIST_TO_STACK=1

coverage run -m pytest test/smoke_test.py -v --durations 20
coverage run -m pytest --runslow --instafail -v --durations 20 --timeout 120
coverage run -m pytest ./benchmarks --instafail -v --durations 20
coverage xml -i
