# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark RedisTensorDict vs local TensorDict for common operations.

Requires a running Redis server on localhost:6379.  Skipped automatically
when the ``redis`` package is missing or the server is unreachable.
"""

import importlib

import pytest
import torch

from tensordict import TensorDict

_has_redis_pkg = importlib.util.find_spec("redis", None) is not None


def _redis_available():
    if not _has_redis_pkg:
        return False
    import redis

    try:
        r = redis.Redis(host="localhost", port=6379, db=0, socket_connect_timeout=2)
        r.ping()
        r.close()
        return True
    except (redis.ConnectionError, redis.exceptions.ConnectionError, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(),
    reason="Redis server not reachable on localhost:6379 or redis package missing",
)

N = 10_000
N_KEYS = 5
FEAT = 64

IDX_INT = 42
IDX_SLICE = slice(100, 356)
IDX_STEP = slice(0, N, 3)
IDX_FANCY = torch.randint(0, N, (256,))
IDX_BOOL = torch.zeros(N, dtype=torch.bool)
IDX_BOOL[torch.randint(0, N, (256,))] = True


@pytest.fixture()
def local_td():
    d = {f"key_{i}": torch.randn(N, FEAT) for i in range(N_KEYS)}
    return TensorDict(d, batch_size=[N])


@pytest.fixture()
def redis_td():
    from tensordict.store import TensorDictStore as RedisTensorDict

    td = RedisTensorDict(batch_size=[N], db=14)
    for i in range(N_KEYS):
        td[f"key_{i}"] = torch.randn(N, FEAT)
    yield td
    td.clear_redis()
    td.close()


@pytest.fixture(params=["local", "redis"])
def td(request, local_td, redis_td):
    if request.param == "local":
        return local_td
    return redis_td


class TestKeyOps:
    def test_get_single_key(self, td, benchmark):
        benchmark(lambda: td["key_0"])

    def test_set_single_key(self, td, benchmark):
        v = torch.randn(N, FEAT)
        benchmark(td.__setitem__, "key_0", v)

    def test_keys_iter(self, td, benchmark):
        benchmark(lambda: list(td.keys()))

    def test_values_iter(self, td, benchmark):
        benchmark(lambda: list(td.values()))

    def test_items_iter(self, td, benchmark):
        benchmark(lambda: list(td.items()))


class TestIndexedRead:
    def test_read_int(self, td, benchmark):
        benchmark(lambda: td[IDX_INT])

    def test_read_slice(self, td, benchmark):
        benchmark(lambda: td[IDX_SLICE])

    def test_read_step(self, td, benchmark):
        benchmark(lambda: td[IDX_STEP])

    def test_read_fancy(self, td, benchmark):
        benchmark(lambda: td[IDX_FANCY])

    def test_read_bool(self, td, benchmark):
        benchmark(lambda: td[IDX_BOOL])


class TestIndexedWrite:
    def test_write_int(self, td, benchmark):
        sub = TensorDict({f"key_{i}": torch.randn(FEAT) for i in range(N_KEYS)}, [])
        benchmark(lambda: td.__setitem__(IDX_INT, sub))

    def test_write_slice(self, td, benchmark):
        n = len(range(*IDX_SLICE.indices(N)))
        sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])
        benchmark(lambda: td.__setitem__(IDX_SLICE, sub))

    def test_write_step(self, td, benchmark):
        n = len(range(*IDX_STEP.indices(N)))
        sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])
        benchmark(lambda: td.__setitem__(IDX_STEP, sub))

    def test_write_fancy(self, td, benchmark):
        n = IDX_FANCY.numel()
        sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])
        benchmark(lambda: td.__setitem__(IDX_FANCY, sub))

    def test_write_bool(self, td, benchmark):
        n = int(IDX_BOOL.sum().item())
        sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])
        benchmark(lambda: td.__setitem__(IDX_BOOL, sub))


class TestConversion:
    def test_to_tensordict(self, td, benchmark):
        benchmark(lambda: td.to_tensordict())

    def test_index_to_td_int(self, td, benchmark):
        benchmark(lambda: td[IDX_INT].to_tensordict())

    def test_index_to_td_slice(self, td, benchmark):
        benchmark(lambda: td[IDX_SLICE].to_tensordict())

    def test_index_to_td_fancy(self, td, benchmark):
        benchmark(lambda: td[IDX_FANCY].to_tensordict())
