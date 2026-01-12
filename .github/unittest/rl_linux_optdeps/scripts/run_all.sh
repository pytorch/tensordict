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

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

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
  pytest-cov \
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
  packaging \
  importlib_metadata

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

# Help CMake find pybind11 when building tensordict from source.
pybind11_DIR="$(python -m pybind11 --cmakedir)"
export pybind11_DIR

# Install build dependencies for --no-build-isolation
uv_pip_install setuptools wheel

# install tensordict
printf "* Installing tensordict\n"
uv_pip_install --no-build-isolation --no-deps -e .

# smoke test
python -c "import tensordict"

printf "* Installing torchrl\n"
git clone https://github.com/pytorch/rl
cd rl
uv_pip_install --no-build-isolation --no-deps -e .
cd ..

# smoke test
python -c "import torchrl"

# ==================================================================================== #
# ================================ Run tests ========================================= #

python -m torch.utils.collect_env

MUJOCO_GL=egl python -m pytest rl/test --instafail -v --durations 20 \
  --ignore rl/test/test_distributed.py \
  --ignore rl/test/llm \
  --timeout=120

# ==================================================================================== #
# ================================ Post-proc ========================================= #

bash ${this_dir}/post_process.sh
