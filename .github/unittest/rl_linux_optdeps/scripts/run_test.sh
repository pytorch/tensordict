#!/usr/bin/env bash

set -e

eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env
apt-get update && apt-get install -y git wget freeglut3 freeglut3-dev

# find libstdc
STDC_LOC=$(find conda/ -name "libstdc++.so.6" | head -1)

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1
# RL should work with the new API
export TD_GET_DEFAULTS_TO_NONE='1'

#MUJOCO_GL=glfw pytest --cov=torchrl --junitxml=test-results/junit.xml -v --durations 20
MUJOCO_GL=egl python -m pytest rl/test --instafail -v --durations 20 --ignore rl/test/test_distributed.py
