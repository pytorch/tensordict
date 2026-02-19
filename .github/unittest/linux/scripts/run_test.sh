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
            apt update -y && apt install -y redis-server && redis-server --daemonize yes --port 6379 --save "" --appendonly no || echo "Redis server not available, redis tests will be skipped"
            ;;
        Darwin*)
            brew install redis 2>/dev/null && redis-server --daemonize yes --port 6379 --save "" --appendonly no || echo "Redis server not available, redis tests will be skipped"
            ;;
        *)
            echo "Redis server not available on this platform, redis tests will be skipped"
            ;;
    esac
fi

# Start Dragonfly server on port 6380 (non-fatal if unavailable)
if command -v dragonfly &> /dev/null; then
    dragonfly --daemonize --port 6380 --dbfilename "" || true
else
    case "$(uname -s)" in
        Linux*)
            DRAGONFLY_VERSION="v1.27.1"
            DRAGONFLY_URL="https://github.com/dragonflydb/dragonfly/releases/download/${DRAGONFLY_VERSION}/dragonfly-x86_64.tar.gz"
            curl -fsSL "$DRAGONFLY_URL" -o /tmp/dragonfly.tar.gz && \
              tar -xzf /tmp/dragonfly.tar.gz -C /tmp && \
              /tmp/dragonfly-x86_64 --daemonize --port 6380 --dbfilename "" || \
              echo "Dragonfly server not available, dragonfly tests will be skipped"
            ;;
        *)
            echo "Dragonfly server not available on this platform, dragonfly tests will be skipped"
            ;;
    esac
fi

JUNIT_DIR="${RUNNER_ARTIFACT_DIR:-.}"
mkdir -p "$JUNIT_DIR"

coverage run -m pytest test/smoke_test.py -v --durations 20 --junitxml="$JUNIT_DIR/junit-smoke.xml"
coverage run -m pytest --runslow --instafail -v --durations 20 --timeout 120 --junitxml="$JUNIT_DIR/junit-tests.xml"
coverage run -m pytest ./benchmarks --instafail -v --durations 20 --junitxml="$JUNIT_DIR/junit-benchmarks.xml"
coverage xml -i

if [ -n "$RUNNER_TEST_RESULTS_DIR" ]; then
    cp "$JUNIT_DIR"/junit-*.xml "$RUNNER_TEST_RESULTS_DIR/" 2>/dev/null || true
fi
