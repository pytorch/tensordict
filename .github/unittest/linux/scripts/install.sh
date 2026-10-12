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

# submodules
git submodule sync && git submodule update --init --recursive

printf "Installing PyTorch with %s\n" "${CU_VERSION}"
# The test environment already holds a torch from PyPI (mosaicml-streaming
# depends on it), so pip must replace it: without --upgrade it keeps it.
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  if [ "${CU_VERSION:-}" == cpu ] ; then
      python -m pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
  else
      python -m pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/$CU_VERSION
  fi
elif [[ "$TORCH_VERSION" == "stable" ]]; then
    if [ "${CU_VERSION:-}" == cpu ] ; then
      python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
  else
      python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/$CU_VERSION
  fi
elif [[ "$TORCH_VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
  # a release series, such as 2.13, with the torchvision release built for it
  if [ "${CU_VERSION:-}" == cpu ] ; then
      python -m pip install "torch==${TORCH_VERSION}.*" torchvision --index-url https://download.pytorch.org/whl/cpu
  else
      python -m pip install "torch==${TORCH_VERSION}.*" torchvision --index-url https://download.pytorch.org/whl/$CU_VERSION
  fi
else
  printf "Failed to install pytorch"
  exit 1
fi
python -c "import torch; print('Installed torch', torch.__version__)"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  # an index that no longer gets nightlies (such as cu128) leaves an older build
  python -c "import torch, sys; sys.exit('.dev' not in torch.__version__)" \
    || { printf "Expected a nightly build of torch\n"; exit 1; }
fi

printf "* Installing tensordict\n"
# Install runtime deps explicitly (except torch/torchvision which are handled above),
# then install tensordict without resolving dependencies to avoid any solver changing
# the PyTorch build (stable vs nightly).
python -m pip install -U packaging pyvers
python -m pip install redis pandas pyarrow
python -m pip install -e . --no-deps

# smoke test
python -c "import functorch"
