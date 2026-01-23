#!/usr/bin/env bash

# This script is for setting up environment in which unit test is ran.
# To speed up the CI time, the resulting environment is cached.
#
# Do not install PyTorch and torchvision here, otherwise they also get cached.

set -e
set -v

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
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    if [ "${os}" == "MacOSX" ]; then
      curl -L -o miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-${ARCH}.sh"
    else
      wget -O miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-${os}-${ARCH}.sh"
    fi
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
eval "$(${conda_dir}/bin/conda shell.bash hook)"

# 2. Create test environment at ./env
printf "python: ${PYTHON_VERSION}\n"
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment\n"
    if [ "${PYTHON_VERSION}" == "3.14t" ]; then
        # Install free-threaded Python 3.14 using pyenv with --disable-gil
        # Install build dependencies
        apt install -y build-essential libssl-dev zlib1g-dev \
            libbz2-dev libreadline-dev libsqlite3-dev curl git \
            libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

        # Install pyenv
        export PYENV_ROOT="${root_dir}/.pyenv"
        curl https://pyenv.run | bash
        export PATH="${PYENV_ROOT}/bin:${PATH}"
        eval "$(pyenv init -)"

        # Build Python 3.14 with free-threading (--disable-gil)
        PYTHON_CONFIGURE_OPTS="--disable-gil" pyenv install 3.14

        # Set pyenv to use this version
        pyenv global 3.14

        # Create a virtual environment using free-threaded Python
        python3 -m venv "${env_dir}"
    else
        conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
    fi
fi

# Activate the environment
if [ "${PYTHON_VERSION}" == "3.14t" ]; then
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
printf "* Installing dependencies (except PyTorch)\n"

pip install pip --upgrade

if [ "${PYTHON_VERSION}" == "3.14t" ]; then
    # For free-threaded Python, install dependencies via pip
    # Install build tools from apt
    apt install -y cmake
    # Install test dependencies (mirrors environment.yml)
    pip install pybind11 numpy expecttest pyyaml hypothesis future cloudpickle \
        pytest pytest-benchmark pytest-cov pytest-mock pytest-instafail \
        pytest-rerunfailures pytest-timeout coverage h5py orjson ninja protobuf
    # Note: mosaicml-streaming may not be available for 3.14t yet, skip if fails
    pip install mosaicml-streaming || echo "mosaicml-streaming not available for Python 3.14t, skipping"
else
    # For regular Python, use conda
    echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
    cat "${this_dir}/environment.yml"
    conda env update --file "${this_dir}/environment.yml" --prune
    conda install anaconda::cmake -y
    conda install -c conda-forge pybind11 -y
fi

#if [[ $OSTYPE == 'darwin'* ]]; then
#  printf "* Installing C++ for OSX\n"
#  conda install -c conda-forge cxx-compiler -y
#fi
