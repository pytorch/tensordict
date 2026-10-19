#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.

set -e
set -v

export DEBIAN_FRONTEND=noninteractive
export TZ=Etc/UTC

apt update -y && apt install git wget gcc -y

this_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'
root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

case "$(uname -s)" in
    Darwin*) os=MacOSX;;
    *) os=Linux
esac

# 1. Install conda at ./conda
if [ "${os}" == "MacOSX" ]; then
    # test-infra's macos_job.yml has installed Miniconda and exported its
    # CONDA_EXE: link to it rather than download a second one. Keep the package
    # cache out of that install, which actions/cache saves after the job.
    ln -sfn "$(dirname "$(dirname "${CONDA_EXE:?macos_job.yml exports CONDA_EXE}")")" "${conda_dir}"
    export CONDA_PKGS_DIRS="${root_dir}/conda-pkgs"
    printf "* Using conda at %s\n" "$(readlink "${conda_dir}")"
fi
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    wget -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-${ARCH}.sh"
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
eval "$(${conda_dir}/bin/conda shell.bash hook)"

# Free-threaded Pythons (3.14t, 3.15t) are built with pyenv. They and Python
# 3.15 install the test dependencies with pip instead of environment.yml, and
# skip the optional ones that do not install yet: h5py, for instance, has no
# wheels for 3.15.
free_threaded=false
pip_only=false
if [[ "${PYTHON_VERSION}" == *t ]]; then
    free_threaded=true
    pip_only=true
elif [[ "${PYTHON_VERSION}" == "3.15" ]]; then
    pip_only=true
fi

# 2. Create test environment at ./env
printf "python: ${PYTHON_VERSION}\n"
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment\n"
    if ${free_threaded}; then
        # Install free-threaded Python using pyenv with --disable-gil
        # Install build dependencies
        apt install -y build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev curl git \
            libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

        # Install pyenv
        export PYENV_ROOT="${root_dir}/.pyenv"
        curl https://pyenv.run | bash
        export PATH="${PYENV_ROOT}/bin:${PATH}"
        eval "$(pyenv init -)"

        # Build Python with free-threading (--disable-gil): 3.15t builds 3.15
        PYTHON_CONFIGURE_OPTS="--disable-gil" pyenv install "${PYTHON_VERSION%t}"

        # Set pyenv to use this version
        pyenv global "${PYTHON_VERSION%t}"

        # Create a virtual environment using free-threaded Python
        python3 -m venv "${env_dir}"
    else
        conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
    fi
fi

# Activate the environment
if ${free_threaded}; then
    # Ensure pyenv is initialized (needed for activation even if env already exists)
    export PYENV_ROOT="${root_dir}/.pyenv"
    export PATH="${PYENV_ROOT}/bin:${PATH}"
    if command -v pyenv &> /dev/null; then
        eval "$(pyenv init -)"
    fi
    source "${env_dir}/bin/activate"
else
    conda activate "${env_dir}"
fi

# 3. Install dependencies
pip install pip --upgrade

# PyTorch first: mosaicml-streaming depends on torch, and without a torch
# installed pip would download torch from PyPI with its CUDA libraries and
# triton, which install.sh would then replace.
printf "Installing PyTorch with %s\n" "${CU_VERSION}"
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

printf "* Installing the test dependencies\n"

if ${pip_only}; then
    # Install test dependencies (mirrors environment.yml)
    pip install numpy cloudpickle pytest pytest-benchmark pytest-instafail \
        pytest-rerunfailures pytest-timeout pytest-xdist redis mypy
    # Optional test dependencies, which may have no wheels for this Python yet
    for pkg in h5py orjson mosaicml-streaming "zarr>=3.0"; do
        pip install "${pkg}" || echo "${pkg} not available for Python ${PYTHON_VERSION}, skipping"
    done
else
    # For regular Python, use conda
    echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
    cat "${this_dir}/environment.yml"
    conda env update --file "${this_dir}/environment.yml" --prune
    # zarr>=3 requires Python>=3.11; skip on older interpreters
    pip install "zarr>=3.0" || echo "zarr>=3.0 not available for Python ${PYTHON_VERSION}, skipping"
fi

#if [[ $OSTYPE == 'darwin'* ]]; then
#  printf "* Installing C++ for OSX\n"
#  conda install -c conda-forge cxx-compiler -y
#fi
