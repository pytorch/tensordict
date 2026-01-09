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
        # Install free-threaded Python 3.14 from conda-forge
        # Need both python=3.14 AND python-freethreading to get the nogil build
        # Also include pip explicitly as it's not included by default in free-threaded builds
        # Configure conda-forge channel first to avoid 403 errors
        conda config --add channels conda-forge
        conda config --set channel_priority strict
        conda create --prefix "${env_dir}" -y python=3.14 python-freethreading pip
    else
        conda create --prefix "${env_dir}" -y python="$PYTHON_VERSION"
    fi
fi

conda activate "${env_dir}"

# 3. Install Conda dependencies
printf "* Installing dependencies (except PyTorch)\n"
# Don't add python version constraint for free-threaded builds
if [ "${PYTHON_VERSION}" != "3.14t" ]; then
    echo "  - python=${PYTHON_VERSION}" >> "${this_dir}/environment.yml"
fi
cat "${this_dir}/environment.yml"

pip install pip --upgrade

conda env update --file "${this_dir}/environment.yml" --prune

conda install anaconda::cmake -y
conda install -c conda-forge pybind11 -y

#if [[ $OSTYPE == 'darwin'* ]]; then
#  printf "* Installing C++ for OSX\n"
#  conda install -c conda-forge cxx-compiler -y
#fi
