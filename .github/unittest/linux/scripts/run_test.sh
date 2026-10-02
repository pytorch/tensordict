#!/usr/bin/env bash

set -e

# Activate the environment
if [ "${PYTHON_VERSION}" == "3.14t" ]; then
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

# Start Redis server on port 6379 (non-fatal if unavailable)
if command -v redis-server &> /dev/null; then
    redis-server --daemonize yes --port 6379 --save "" --appendonly no || true
else
    case "$(uname -s)" in
        Linux*)
            { { command -v apt > /dev/null 2>&1 && apt update -y && apt install -y redis-server; } ||
              { command -v yum > /dev/null 2>&1 && yum install -y redis; } ; } &&
              redis-server --daemonize yes --port 6379 --save "" --appendonly no ||
              echo "Redis server not available, redis tests will be skipped"
            ;;
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
    ping_dragonfly="import redis; redis.Redis(port=6380, socket_connect_timeout=1).ping()"
    for _ in $(seq 30); do
        python -c "$ping_dragonfly" 2> /dev/null && break
        sleep 1
    done
    if ! python -c "$ping_dragonfly"; then
        echo "Dragonfly did not start on port 6380:"
        cat /tmp/dragonfly.log
        exit 1
    fi
fi

JUNIT_DIR="${RUNNER_ARTIFACT_DIR:-.}"
mkdir -p "$JUNIT_DIR"

coverage run -m pytest test/smoke_test.py -v --durations 20 --junitxml="$JUNIT_DIR/junit-smoke.xml"
test_status=0
coverage run -m pytest --runslow --instafail -v --durations 20 --timeout 120 --junitxml="$JUNIT_DIR/junit-tests.xml" || test_status=$?

if [ "$test_status" -ne 0 ]; then
    # Record same-commit evidence without hiding the original CI failure.
    python -m pytest --runslow --last-failed --last-failed-no-failures none \
        --instafail -v --durations 20 --timeout 120 \
        --junitxml="$JUNIT_DIR/junit-tests-rerun.xml" || true
    if [ -n "$RUNNER_TEST_RESULTS_DIR" ]; then
        cp "$JUNIT_DIR"/junit-*.xml "$RUNNER_TEST_RESULTS_DIR/" 2>/dev/null || true
    fi
    exit "$test_status"
fi

coverage run -m pytest ./benchmarks --instafail -v --durations 20 --junitxml="$JUNIT_DIR/junit-benchmarks.xml"
coverage xml -i

if [ -n "$RUNNER_TEST_RESULTS_DIR" ]; then
    cp "$JUNIT_DIR"/junit-*.xml "$RUNNER_TEST_RESULTS_DIR/" 2>/dev/null || true
fi
