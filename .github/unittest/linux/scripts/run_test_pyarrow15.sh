#!/usr/bin/env bash

# Run the tabular tests with pyarrow 15, the oldest release that from_arrow and
# to_arrow support. run_test_min_deps.sh tests pyarrow 10, which the parquet and
# tabular extras accept, and so deselects the tests of from_arrow and to_arrow.
# The test-pyarrow15-cpu job of test-linux.yml runs it after setup_env.sh and
# install.sh, which install the newest releases.

set -e
set -v

# Activate the environment
eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# pyarrow 15 requires numpy<2, and pandas 2.2 came out at the same time as
# pyarrow 15.
pyarrow15_deps=(
    "pyarrow==15.0.*"
    "pandas==2.2.*"
    "numpy==1.26.*"
)
python -m pip install "${pyarrow15_deps[@]}"
python -c "
import numpy, pandas, pyarrow, torch
for module in (torch, numpy, pyarrow, pandas):
    print(module.__name__, module.__version__)
"

root_dir="$(git rev-parse --show-toplevel)"
lib_dir="${root_dir}/env/lib"

# As in run_test.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
export TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1
export TD_GET_DEFAULTS_TO_NONE=1
export LIST_TO_STACK=1
export TD_CHECK_INVARIANTS=1

JUNIT_DIR="${RUNNER_ARTIFACT_DIR:-.}"
mkdir -p "$JUNIT_DIR"

test_status=0
python -m pytest --runslow -n "${TD_TEST_WORKERS:-4}" --dist worksteal \
    test/tensordict/test_tabular.py \
    --instafail -v --durations 20 --timeout 120 \
    --junitxml="$JUNIT_DIR/junit-tests-pyarrow15.xml" || test_status=$?

if [ -n "$RUNNER_TEST_RESULTS_DIR" ]; then
    cp "$JUNIT_DIR"/junit-*.xml "$RUNNER_TEST_RESULTS_DIR/" 2>/dev/null || true
fi
exit "$test_status"
