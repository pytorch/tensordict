#!/usr/bin/env bash

set -e

# Activate the environment
if [[ "${PYTHON_VERSION}" == *t ]]; then
    source ./env/bin/activate
else
    eval "$(./conda/bin/conda shell.bash hook)"
    conda activate ./env
fi

export PYTORCH_TEST_WITH_SLOW='1'
python -m torch.utils.collect_env
# Avoid error: "fatal: unsafe repository"
git config --global --add safe.directory '*'

root_dir="$(git rev-parse --show-toplevel)"
env_dir="${root_dir}/env"
lib_dir="${env_dir}/lib"

# solves ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$lib_dir
export MKL_THREADING_LAYER=GNU
export TORCHDYNAMO_INLINE_INBUILT_NN_MODULES=1
export TD_GET_DEFAULTS_TO_NONE=1
export LIST_TO_STACK=1
# check the tensordicts that are built without validation (see _check_invariants)
export TD_CHECK_INVARIANTS=1

# Ping a Redis-compatible server on port $2 up to 30 times, a second apart, and
# fail the job, showing log file $3, if it does not answer.
wait_for_server() {
    local ping="import redis; redis.Redis(port=$2, socket_connect_timeout=1).ping()"
    for _ in $(seq 30); do
        python -c "$ping" 2> /dev/null && return 0
        sleep 1
    done
    if ! python -c "$ping"; then
        echo "$1 did not start on port $2:"
        cat "$3"
        exit 1
    fi
}

# Start Redis on port 6379. The store tests skip when it is not reachable, so a
# Linux job fails if Redis cannot be started, as it does for Dragonfly below.
if [ "$(uname -s)" == "Linux" ]; then
    if ! command -v redis-server &> /dev/null; then
        apt update -y
        apt install -y redis-server
    fi
    redis-server --daemonize yes --port 6379 --save "" --appendonly no --logfile /tmp/redis.log
    wait_for_server Redis 6379 /tmp/redis.log
elif command -v redis-server &> /dev/null; then
    redis-server --daemonize yes --port 6379 --save "" --appendonly no || true
else
    case "$(uname -s)" in
        Darwin*)
            brew install redis 2>/dev/null && redis-server --daemonize yes --port 6379 --save "" --appendonly no || echo "Redis server not available, redis tests will be skipped"
            ;;
        *)
            echo "Redis server not available on this platform, redis tests will be skipped"
            ;;
    esac
fi

# Start Dragonfly on port 6380. The store tests run against both Redis and
# Dragonfly, so a Linux x86_64 job fails if Dragonfly cannot be started.
if command -v dragonfly &> /dev/null; then
    dragonfly_bin="dragonfly"
elif [ "$(uname -s)-$(uname -m)" == "Linux-x86_64" ]; then
    # Update the checksum when changing the version.
    DRAGONFLY_VERSION="v1.27.1"
    DRAGONFLY_SHA256="b61e8580076392ced641f2ed3d3d6edf7e10e6e5329437ef8fa2ad832b7f9faa"
    DRAGONFLY_URL="https://github.com/dragonflydb/dragonfly/releases/download/${DRAGONFLY_VERSION}/dragonfly-x86_64.tar.gz"
    # wget is installed by setup_env.sh; the image has no curl.
    wget --no-verbose -O /tmp/dragonfly.tar.gz "$DRAGONFLY_URL"
    echo "${DRAGONFLY_SHA256}  /tmp/dragonfly.tar.gz" | sha256sum --check -
    tar -xzf /tmp/dragonfly.tar.gz -C /tmp
    dragonfly_bin="/tmp/dragonfly-x86_64"
else
    echo "Dragonfly is not set up on $(uname -s)-$(uname -m), dragonfly tests will be skipped"
fi
if [ -n "${dragonfly_bin:-}" ]; then
    # Dragonfly has no --daemonize flag, so run it in the background.
    "$dragonfly_bin" --port 6380 --dbfilename "" --logtostderr > /tmp/dragonfly.log 2>&1 &
    wait_for_server Dragonfly 6380 /tmp/dragonfly.log
fi

JUNIT_DIR="${RUNNER_ARTIFACT_DIR:-.}"
mkdir -p "$JUNIT_DIR"

python -m pytest test/smoke_test.py -v --durations 20 --junitxml="$JUNIT_DIR/junit-smoke.xml"
test_status=0
# Run the examples of the docstrings and of the docs. The jobs with a pinned
# torch set TD_RUN_DOCTESTS: the printed reprs can change with nightly torch.
if [ -n "${TD_RUN_DOCTESTS:-}" ]; then
    # Without the cache, so that the --last-failed rerun below skips them.
    python -m pytest --doctest-modules tensordict -n 4 -p no:cacheprovider \
        --junitxml="$JUNIT_DIR/junit-doctests.xml" || test_status=$?
    python -m pytest --doctest-glob='*.rst' docs/source -n 4 -p no:cacheprovider \
        --junitxml="$JUNIT_DIR/junit-doctests-docs.xml" || test_status=$?
fi
# Run the tests in TD_TEST_WORKERS pytest-xdist workers (4 unless the job sets
# it), except two files that run on their own afterwards. With worksteal, an
# idle worker takes tests from the queue of a busy one: the default scheduler
# gives each worker a block of consecutive tests at the start, so a run of slow
# tests, such as those of TestContainersCompile, would stay on one worker.
python -m pytest --runslow -n "${TD_TEST_WORKERS:-4}" --dist worksteal \
    --ignore test/distributed/test_distributed.py \
    --ignore test/utils/test_setup.py \
    --instafail -v --durations 20 --timeout 120 \
    --junitxml="$JUNIT_DIR/junit-tests.xml" || test_status=$?
# test_distributed.py starts its process groups on fixed ports, and its
# worker processes can hang at exit when the machine is busy. test_setup.py
# reinstalls tensordict from this checkout. So these two run alone, one test
# at a time. The single-GPU jobs set TD_SKIP_DISTRIBUTED_TESTS:
# the CUDA tests of test_distributed.py need 2 GPUs, and the CPU jobs run the
# rest on the same Python and torch.
serial_files=(test/utils/test_setup.py)
if [ -z "${TD_SKIP_DISTRIBUTED_TESTS:-}" ]; then
    serial_files=(test/distributed/test_distributed.py "${serial_files[@]}")
fi
python -m pytest --runslow "${serial_files[@]}" \
    --instafail -v --durations 20 --timeout 120 \
    --junitxml="$JUNIT_DIR/junit-tests-serial.xml" || test_status=$?

if [ "$test_status" -ne 0 ]; then
    # Record same-commit evidence without hiding the original CI failure.
    # The rerun is serial, so a failure that only shows under xdist passes here.
    python -m pytest --runslow --last-failed --last-failed-no-failures none \
        --instafail -v --durations 20 --timeout 120 \
        --junitxml="$JUNIT_DIR/junit-tests-rerun.xml" || true
    if [ -n "$RUNNER_TEST_RESULTS_DIR" ]; then
        cp "$JUNIT_DIR"/junit-*.xml "$RUNNER_TEST_RESULTS_DIR/" 2>/dev/null || true
    fi
    exit "$test_status"
fi

# The benchmark workflows time the benchmarks. Here, run each one once, to
# check that it still works.
# pytest-benchmark 5.3.0 reads pytest's private node._nodeid, which pytest 9.2
# removed (ionelmc/pytest-benchmark#321): the tests above run on the latest
# pytest, the benchmarks below on pytest<9.2. Drop this once a fixed release is out.
python -m pip install --quiet "pytest<9.2"
python -m pytest ./benchmarks --benchmark-disable --instafail -v --durations 20 --junitxml="$JUNIT_DIR/junit-benchmarks.xml"

if [ -n "$RUNNER_TEST_RESULTS_DIR" ]; then
    cp "$JUNIT_DIR"/junit-*.xml "$RUNNER_TEST_RESULTS_DIR/" 2>/dev/null || true
fi
