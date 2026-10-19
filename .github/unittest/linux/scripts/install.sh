#!/usr/bin/env bash

unset PYTORCH_VERSION
# For unittest, nightly PyTorch is used as the following section,
# so no need to set PYTORCH_VERSION.
# In fact, keeping PYTORCH_VERSION forces us to hardcode PyTorch version in config.

set -e
set -v

# Activate the environment
if [[ "${PYTHON_VERSION}" == *t ]]; then
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

# setup_env.sh installed PyTorch before the test dependencies, so that
# none of them pulls torch from PyPI.
python -c "import torch; print('Installed torch', torch.__version__)"
if [[ "$TORCH_VERSION" == "nightly" ]]; then
  # an index that no longer gets nightlies (such as cu128) leaves an older build,
  # and a test dependency could pull a release from PyPI
  python -c "import torch, sys; sys.exit('.dev' not in torch.__version__)" \
    || { printf "Expected a nightly build of torch\n"; exit 1; }
fi

printf "* Installing tensordict\n"
# Install runtime deps explicitly (except torch/torchvision, which setup_env.sh installs),
# then install tensordict without resolving dependencies to avoid any solver changing
# the PyTorch build (stable vs nightly).
python -m pip install -U packaging pyvers
python -m pip install redis pandas pyarrow
# Test with the newest NumPy for this Python, or with NUMPY_SPEC (such as "<2")
# when the job sets it. mosaicml-streaming caps NumPy below 2.2, but the to_mds
# tests pass with newer releases, so this overrides the cap.
python -m pip install --upgrade "numpy${NUMPY_SPEC:-}"
python -c "import numpy; print('Installed numpy', numpy.__version__)"
python -m pip install -e . --no-deps

# smoke test
python -c "import functorch"
