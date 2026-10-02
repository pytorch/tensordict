#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e
set -v

# Activate the environment
if [ "${PYTHON_VERSION}" == "3.14t" ]; then
    source ./env/bin/activate
else
    eval "$(./conda/bin/conda shell.bash hook)"
    conda activate ./env
fi

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

# No submodules: this repo has no .gitmodules, so the sync/update pair was
# always a no-op. It is only removed rather than guarded because it also
# needs a .git, which a runner whose image lacks git does not get --
# actions/checkout falls back to a REST API tarball there. If submodules
# are ever added, linux_job_v3's `submodules:` input is where to ask for
# them, so the checkout fetches them properly.

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
  else
      python -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  else
      python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/$CU_VERSION
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi

printf "* Installing tensordict\n"
# Install runtime deps explicitly (except torch/torchvision which are handled above),
# then install tensordict without resolving dependencies to avoid any solver changing
# the PyTorch build (stable vs nightly).
python -m pip install -U packaging pyvers importlib_metadata
python -m pip install redis pandas pyarrow
python -m pip install -e . --no-deps

# smoke test
python -c "import functorch"
