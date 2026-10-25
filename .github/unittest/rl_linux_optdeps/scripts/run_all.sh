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
curl -LsSf --retry 3 https://astral.sh/uv/install.sh | sh
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
  psutil \
  "hydra-core<1.4" \
  tqdm \
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
git -C rl checkout --detach "${TORCHRL_REF:-f0af32d31045f235b30210fa2ce01d581c08b986}"
cd rl
uv_pip_install --no-build-isolation --no-deps -e .

# smoke test
python -c "import torchrl"

# ==================================================================================== #
# ================================ Run tests ========================================= #

python -m torch.utils.collect_env

# RL_TEST_SHARD picks one of three parallel jobs, split as TorchRL splits its
# own suite: test_collectors.py, the other process-spawning files, and the
# rest. Together they run every test once.
collector_paths=(test/test_collectors.py)
mp_paths=(
  test/envs/test_parallel.py
  test/envs/test_special.py
  test/envs/test_auto_reset.py
  test/envs/test_env_base.py
  test/envs/test_nested.py
  test/envs/test_step_mdp.py
  test/services
  test/test_inference_server.py
  test/test_loggers.py
)
case "${RL_TEST_SHARD:-}" in
  collectors) shard_args=("${collector_paths[@]}") ;;
  mp) shard_args=("${mp_paths[@]}") ;;
  rest)
    shard_args=(test)
    for path in "${collector_paths[@]}" "${mp_paths[@]}"; do
      shard_args+=(--ignore "${path}")
    done
    ;;
  *)
    echo "Unknown RL_TEST_SHARD='${RL_TEST_SHARD:-}'. Expected: collectors|mp|rest."
    exit 1
    ;;
esac

# TorchRL validates its standalone Triton GRU numerics, and the Triton kernels
# behind LSTMModule and GRUModule, in its own CI. Keep this reverse-dependency
# job focused on TensorDict interoperability: the other backends in test_rnn.py
# go through the same TensorDict code. The Triton tests of test_rnn.py took 6.4
# of its 9.8 minutes. --deselect drops every test whose ID starts with the
# given text, so test_lstm_module_triton_ covers five tests.
# torchvision is not installed here (it would have to match the torch nightly),
# and this test fails instead of skipping without it.
# These tests fail at random, whatever the TensorDict change, and turn
# unrelated PRs red. Run them again once TorchRL fixes them:
# - test_discrete_actor[cpu-autocast] and [cuda-autocast]: under autocast two
#   logits can be equal, and the deterministic action then has two ones
#   (pytorch/rl#4508).
# - test_worker_affinity_error_shuts_down_workers: AsyncEnvPool reports
#   "worker process exited unexpectedly" when another worker exits before
#   worker 0 reports its CPU affinity error (pytorch/rl#4563).
# - test_multiprocess_shutdown_is_bounded_and_idempotent: a worker can still be
#   alive just after shutdown(timeout=0.5) returns. It failed on four unrelated
#   commits and passed on rerun (runs 38340954621, 38827005518, and attempt 1
#   of 38843598244 and 38973273656).
# - test_replay_buffer_prefetch_without_replacement_roundtrip: the restored
#   buffer yields a different number of batches than the source ("assert 3 ==
#   7"), depending on what the prefetch thread had queued when state_dict ran.
#   It failed on a docs-only commit (runs 38583899324 and 38820956479).
# test_setup.py checks how TorchRL itself builds and installs, which TorchRL
# tests in its own CI; TensorDict plays no part in it.
# test_dreamer_v3_reinforce_return_normalization builds its expected loss from
# the log-probability that ProbabilisticTensorDictModule stored for the sampled
# action, while DreamerV3ActorLoss scores the action again with
# dist.log_prob(action). Since #2563 the module stores the log-probability of
# the draw itself (rsample_and_log_prob), which differs from the re-scored one
# when tanh saturates. The loss is unchanged. Run the test again once TorchRL
# computes its expected value the way the loss does.
MUJOCO_GL=egl python -m pytest "${shard_args[@]}" --instafail -v --durations 20 \
  --ignore test/test_distributed.py \
  --ignore test/llm \
  --ignore test/test_setup.py \
  --deselect test/modules/test_dreamer_components.py::test_public_block_gru_triton_gradient_parity \
  --deselect test/modules/test_dreamer_components.py::test_public_block_gru_triton_compile_recurrent_loss \
  --deselect test/modules/test_rnn.py::TestLSTMModule::test_lstm_module_triton_ \
  --deselect test/modules/test_rnn.py::TestLSTMModule::test_lstm_triton_recompute_parity \
  --deselect "test/modules/test_rnn.py::TestLSTMModule::test_lstm_module_backends_non_contiguous_recurrent_inputs[triton]" \
  --deselect test/modules/test_rnn.py::TestGRUModule::test_gru_module_triton_ \
  --deselect test/modules/test_rnn.py::TestGRUModule::test_gru_triton_recompute_parity \
  --deselect "test/modules/test_rnn.py::TestGRUModule::test_gru_module_backends_non_contiguous_recurrent_inputs[triton]" \
  --deselect test/modules/test_rnn.py::TestRecurrentMatmulPrecision::test_lstm_triton_precision_grad_parity \
  --deselect test/modules/test_rnn.py::TestRecurrentMatmulPrecision::test_gru_triton_precision_grad_parity \
  --deselect test/modules/test_rnn.py::test_triton_compile_forward_backward \
  --deselect "test/test_helpers.py::test_transformed_env_constructor_with_state_dict[from_pixels1]" \
  --deselect "test/modules/test_dreamer_components.py::TestDreamerV3Components::test_discrete_actor[cpu-autocast]" \
  --deselect "test/modules/test_dreamer_components.py::TestDreamerV3Components::test_discrete_actor[cuda-autocast]" \
  --deselect test/envs/test_special.py::TestAsyncEnvPool::test_worker_affinity_error_shuts_down_workers \
  --deselect test/test_collectors.py::TestCollectorGeneric::test_multiprocess_shutdown_is_bounded_and_idempotent \
  --deselect test/rb/test_rb_core.py::test_replay_buffer_prefetch_without_replacement_roundtrip \
  --deselect test/objectives/test_dreamer_v3.py::TestDreamerV3::test_dreamer_v3_reinforce_return_normalization \
  --timeout=120
