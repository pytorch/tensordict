#!/usr/bin/env bash

# Run the tests of the optional features with the oldest releases of their
# packages that pyproject.toml's extras accept. The test-min-deps-cpu job of
# test-linux.yml runs it after setup_env.sh and install.sh, which install the
# newest releases.

set -e
set -v

# Activate the environment
eval "$(./conda/bin/conda shell.bash hook)"
conda activate ./env

# The minimum of each extra in pyproject.toml, in its oldest minor series
# (zarr at its exact floor, as 3.0.0 to 3.0.7 are yanked): update them
# together. numpy is the oldest series that all of these accept: zarr 3.0.8
# needs numpy>=1.25. safetensors 0.4.5 is the oldest release that handles
# float8 tensors and empty tensordicts.
min_deps=(
    "pyarrow==10.0.*"
    "pandas==1.5.*"
    "h5py==3.9.*"
    "zarr==3.0.8"
    "safetensors==0.4.5"
    "numpy==1.25.*"
)
python -m pip install "${min_deps[@]}"
python -c "
import h5py, numpy, pandas, pyarrow, safetensors, torch, zarr
for module in (torch, numpy, pyarrow, pandas, h5py, zarr, safetensors):
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

# The tests of the tabular readers and writers, of the h5 and zarr backends and
# of memory-mapping, and the tests elsewhere whose names mention h5, zarr or
# safetensors, such as the h5 cases of the shared tensordict tests.
# from_arrow and to_arrow need pyarrow 15 or later, and raise a clear
# ImportError below that, which the old_pyarrow and MinimumVersion tests check.
# The parquet and tabular extras admit pyarrow 10, which this job tests through
# from_parquet, to_parquet and pandas, so the -k deselects the other tests of
# from_arrow and to_arrow.
test_status=0
python -m pytest --runslow -n "${TD_TEST_WORKERS:-4}" --dist worksteal \
    test/tensordict/test_tabular.py test/memmap \
    -k "not (TestArrow or from_arrow or to_arrow) or old_pyarrow or MinimumVersion" \
    --instafail -v --durations 20 --timeout 120 \
    --junitxml="$JUNIT_DIR/junit-tests-min-deps.xml" || test_status=$?
python -m pytest --runslow -n "${TD_TEST_WORKERS:-4}" --dist worksteal \
    test/tensordict test/tensorclass -k "h5 or zarr or safetensors" \
    --instafail -v --durations 20 --timeout 120 \
    --junitxml="$JUNIT_DIR/junit-tests-min-deps-named.xml" || test_status=$?

if [ -n "$RUNNER_TEST_RESULTS_DIR" ]; then
    cp "$JUNIT_DIR"/junit-*.xml "$RUNNER_TEST_RESULTS_DIR/" 2>/dev/null || true
fi
exit "$test_status"
