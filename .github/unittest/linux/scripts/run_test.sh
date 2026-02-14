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

# Start Redis server for test_redis.py (non-fatal if unavailable)
if command -v redis-server &> /dev/null; then
    redis-server --daemonize yes --save "" --appendonly no || true
else
    case "$(uname -s)" in
        Linux*)
            apt update -y && apt install -y redis-server && redis-server --daemonize yes --save "" --appendonly no || echo "Redis server not available, redis tests will be skipped"
            ;;
        Darwin*)
            brew install redis 2>/dev/null && redis-server --daemonize yes --save "" --appendonly no || echo "Redis server not available, redis tests will be skipped"
            ;;
        *)
            echo "Redis server not available on this platform, redis tests will be skipped"
            ;;
    esac
fi

coverage run -m pytest test/smoke_test.py -v --durations 20
coverage run -m pytest --runslow --instafail -v --durations 20 --timeout 120
coverage run -m pytest ./benchmarks --instafail -v --durations 20
coverage xml -i
