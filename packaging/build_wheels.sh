#!/bin/bash
set -ex

script_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$script_dir/pkg_helpers.bash"

export BUILD_TYPE=wheel
setup_env
setup_wheel_python
pip_install numpy pyyaml future ninja
pip_install --upgrade setuptools
setup_pip_pytorch_version
python setup.py clean

# Copy binaries to be included in the wheel distribution
if [[ "$(uname)" == Darwin || "$OSTYPE" == "msys" ]]; then
    python_exec="$(which python)"
    bin_path=$(dirname $python_exec)
else
    # Point to custom libraries
    export LD_LIBRARY_PATH=$(pwd)/ext_libraries/lib:$LD_LIBRARY_PATH
fi

if [[ "$OSTYPE" == "msys" ]]; then
  echo "ERROR: Windows installation is not supported yet." && exit 100
else
    python setup.py bdist_wheel
    if [[ "$(uname)" != Darwin ]]; then
      rename "linux_x86_64" "manylinux1_x86_64" dist/*.whl
    fi
fi
