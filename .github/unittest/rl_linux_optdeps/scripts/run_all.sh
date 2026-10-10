#!/usr/bin/env bash

set -euxo pipefail
set -v

# =============================================================================== #
# ================================ Init ========================================= #

# Prevent interactive prompts (notably tzdata) in CI.
export DEBIAN_FRONTEND=noninteractive
export TZ="${TZ:-Etc/UTC}"
ln -snf "/usr/share/zoneinfo/${TZ}" /etc/localtime || true
echo "${TZ}" > /etc/timezone || true

apt-get update
apt-get install -y --no-install-recommends tzdata
dpkg-reconfigure -f noninteractive tzdata || true

apt-get upgrade -y
apt-get install -y vim git wget cmake curl python3-dev gcc g++ freeglut3 freeglut3-dev

if [ "${CU_VERSION:-}" == cpu ] ; then
  apt-get upgrade -y libstdc++6
  apt-get dist-upgrade -y
fi

# ==================================================================================== #
# ================================ Setup env ========================================= #

# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/venv"

cd "${root_dir}"

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Create venv with uv
printf "* Creating venv with Python ${PYTHON_VERSION}\n"
# Ensure a clean environment
rm -rf "${env_dir}"
uv venv --python "${PYTHON_VERSION}" "${env_dir}"
source "${env_dir}/bin/activate"
uv_pip_install() {
  uv pip install --no-progress --python "${env_dir}/bin/python" "$@"
}

# Verify CPython
python -c "import sys; assert sys.implementation.name == 'cpython', f'Expected CPython, got {sys.implementation.name}'"

# Set environment variables
if [ "${CU_VERSION:-}" == cpu ] ; then
  export MUJOCO_GL=glfw
else
  export MUJOCO_GL=egl
fi

export PYTORCH_TEST_WITH_SLOW='1'
export MKL_THREADING_LAYER=GNU
export CKPT_BACKEND=torch
export TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1
# RL should work with the new API
export TD_GET_DEFAULTS_TO_NONE='1'

# ==================================================================================== #
# ================================ Install dependencies ============================== #

printf "* Installing dependencies\n"

# Install base dependencies
uv_pip_install \
  hypothesis \
  future \
  cloudpickle \
  pytest \
  pytest-mock \
  pytest-instafail \
  pytest-rerunfailures \
  pytest-timeout \
  expecttest \
  "pybind11[global]>=2.13" \
  pyyaml \
  scipy \
  orjson \
  ninja \
  pyvers \
  packaging

# ============================================================================================ #
# ================================ PyTorch & TensorDict & TorchRL ============================ #

unset PYTORCH_VERSION

if [ "${CU_VERSION:-}" == cpu ] ; then
    echo "Using cpu build"
else
    if [[ ${#CU_VERSION} -eq 4 ]]; then
        CUDA_VERSION="${CU_VERSION:2:1}.${CU_VERSION:3:1}"
    elif [[ ${#CU_VERSION} -eq 5 ]]; then
        CUDA_VERSION="${CU_VERSION:2:2}.${CU_VERSION:4:1}"
    fi
    echo "Using CUDA $CUDA_VERSION as determined by CU_VERSION ($CU_VERSION)"
fi

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [ "${CU_VERSION:-}" == cpu ] ; then
    uv_pip_install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
else
    uv_pip_install --upgrade --pre torch --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION
fi

# smoke test
python -c "import functorch"

# Help CMake find pybind11 when building TorchRL's C++ extension from source.
pybind11_DIR="$(python -m pybind11 --cmakedir)"
export pybind11_DIR

# Install build dependencies for --no-build-isolation
uv_pip_install setuptools wheel

# install tensordict
printf "* Installing tensordict\n"
uv_pip_install --no-build-isolation --no-deps -e .

# smoke test
python -c "import tensordict"

printf "* Installing hoptorch\n"
uv_pip_install "hoptorch>=0.1.4"

printf "* Installing torchrl\n"
git clone https://github.com/pytorch/rl
git -C rl checkout --detach "${TORCHRL_REF:-565e826ef7589006fbde5c4c45c0ec5e2329538b}"
cd rl
uv_pip_install --no-build-isolation --no-deps -e .

# smoke test
python -c "import torchrl"

# ==================================================================================== #
# ================================ Run tests ========================================= #

python -m torch.utils.collect_env

# TorchRL validates its standalone Triton GRU numerics in its own CI. Keep this
# reverse-dependency job focused on TensorDict interoperability.
#
# The other deselected tests fail now and then at the pinned TorchRL revision,
# for reasons inside TorchRL. Rerunning them would not help: TorchRL's
# prevent_leaking_rng fixture restores the RNG state after each test, so a
# rerun draws the same numbers.
# - test_multiagent_reset_mlp is unseeded and fails when any parameter that
#   reset_parameters() draws lands within rtol=1e-5 of its old value or of
#   another agent's value: once in about 500 runs for [True-False-3]. Not
#   fixed in TorchRL yet.
# - test_ddpg_prioritized_weights is unseeded. TorchRL seeds it in 5ef3c124.
# - test_rssm_rollout_higher_order_scan_matches_loop compares float32 CUDA
#   gradients with rtol=5e-5. TorchRL raises rtol to 1e-3 in d7c14b23.
#
# The rerun is for a race in the multiprocess collectors' shutdown, which
# test_env_that_errors hits: when one worker dies first,
# _check_for_faulty_process closes the others, and shutdown then calls
# is_alive() on a closed process. TorchRL fixes the shutdown in 0e3f69bc.
#
# Drop the last two deselects and the rerun once TORCHRL_REF includes their fixes.
MUJOCO_GL=egl python -m pytest test --instafail -v --durations 20 \
  --ignore test/test_distributed.py \
  --ignore test/llm \
  --deselect test/modules/test_dreamer_components.py::test_public_block_gru_triton_gradient_parity \
  --deselect test/modules/test_dreamer_components.py::test_public_block_gru_triton_compile_recurrent_loss \
  --deselect test/modules/test_multiagent_models.py::TestMultiAgent::test_multiagent_reset_mlp \
  --deselect test/objectives/test_ddpg.py::TestDDPG::test_ddpg_prioritized_weights \
  --deselect test/modules/test_dreamer_components.py::TestDreamerV3Components::test_rssm_rollout_higher_order_scan_matches_loop \
  --reruns 1 --only-rerun "ValueError: process object is closed" \
  --timeout=120
