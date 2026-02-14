# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark RedisTensorDict vs local TensorDict for common operations.

Measures key-based get/set, key/value iteration, and indexed read/write
(int, slice, fancy) for both backends.

Run with:
    python benchmarks/storage/bench_redis.py
"""

import importlib
import time

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N = 10_000  # batch size
N_KEYS = 5  # number of leaf keys
FEAT = 64  # feature dim per key
WARMUP = 2
ROUNDS = 10

IDX_INT = 42
IDX_SLICE = slice(100, 356)  # 256 rows, step=1
IDX_STEP = slice(0, N, 3)  # every 3rd row
IDX_FANCY = torch.randint(0, N, (256,))
IDX_BOOL = torch.zeros(N, dtype=torch.bool)
IDX_BOOL[torch.randint(0, N, (256,))] = True


def _make_local_td():
    d = {f"key_{i}": torch.randn(N, FEAT) for i in range(N_KEYS)}
    return TensorDict(d, batch_size=[N])


def _make_redis_td():
    from tensordict.redis import RedisTensorDict

    td = RedisTensorDict(batch_size=[N], db=14)
    for i in range(N_KEYS):
        td[f"key_{i}"] = torch.randn(N, FEAT)
    return td


def _timeit(fn, warmup=WARMUP, rounds=ROUNDS):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    return mean, std


# ---------------------------------------------------------------------------
# Benchmark definitions -- each returns (mean_s, std_s)
# ---------------------------------------------------------------------------


def bench_get_single_key(td):
    return _timeit(lambda: td["key_0"])


def bench_set_single_key(td):
    v = torch.randn(N, FEAT)
    return _timeit(lambda: td.__setitem__("key_0", v))


def bench_keys_iter(td):
    return _timeit(lambda: list(td.keys()))


def bench_values_iter(td):
    return _timeit(lambda: list(td.values()))


def bench_items_iter(td):
    return _timeit(lambda: list(td.items()))


def bench_read_int(td):
    return _timeit(lambda: td[IDX_INT])


def bench_read_slice(td):
    return _timeit(lambda: td[IDX_SLICE])


def bench_read_step(td):
    return _timeit(lambda: td[IDX_STEP])


def bench_read_fancy(td):
    return _timeit(lambda: td[IDX_FANCY])


def bench_read_bool(td):
    return _timeit(lambda: td[IDX_BOOL])


def bench_write_int(td):
    sub = TensorDict({f"key_{i}": torch.randn(FEAT) for i in range(N_KEYS)}, [])

    def _write():
        td[IDX_INT] = sub

    return _timeit(_write)


def bench_write_slice(td):
    n = len(range(*IDX_SLICE.indices(N)))
    sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])

    def _write():
        td[IDX_SLICE] = sub

    return _timeit(_write)


def bench_write_step(td):
    n = len(range(*IDX_STEP.indices(N)))
    sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])

    def _write():
        td[IDX_STEP] = sub

    return _timeit(_write)


def bench_write_fancy(td):
    n = IDX_FANCY.numel()
    sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])

    def _write():
        td[IDX_FANCY] = sub

    return _timeit(_write)


def bench_write_bool(td):
    n = int(IDX_BOOL.sum().item())
    sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])

    def _write():
        td[IDX_BOOL] = sub

    return _timeit(_write)


def bench_to_tensordict(td):
    return _timeit(lambda: td.to_tensordict())


def bench_index_to_td_int(td):
    return _timeit(lambda: td[IDX_INT].to_tensordict())


def bench_index_to_td_slice(td):
    return _timeit(lambda: td[IDX_SLICE].to_tensordict())


def bench_index_to_td_fancy(td):
    return _timeit(lambda: td[IDX_FANCY].to_tensordict())


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

BENCHMARKS = [
    ("get single key", bench_get_single_key),
    ("set single key", bench_set_single_key),
    ("keys() iteration", bench_keys_iter),
    ("values() iteration", bench_values_iter),
    ("items() iteration", bench_items_iter),
    ("read td[int]", bench_read_int),
    ("read td[slice]", bench_read_slice),
    ("read td[::3]", bench_read_step),
    ("read td[fancy]", bench_read_fancy),
    ("read td[bool]", bench_read_bool),
    ("write td[int]=v", bench_write_int),
    ("write td[slice]=v", bench_write_slice),
    ("write td[::3]=v", bench_write_step),
    ("write td[fancy]=v", bench_write_fancy),
    ("write td[bool]=v", bench_write_bool),
    ("to_tensordict()", bench_to_tensordict),
    ("td[int].to_tensordict()", bench_index_to_td_int),
    ("td[slice].to_tensordict()", bench_index_to_td_slice),
    ("td[fancy].to_tensordict()", bench_index_to_td_fancy),
]


def _fmt(mean, std):
    if mean < 1e-3:
        return f"{mean * 1e6:8.1f} us +/- {std * 1e6:6.1f} us"
    if mean < 1:
        return f"{mean * 1e3:8.2f} ms +/- {std * 1e3:6.2f} ms"
    return f"{mean:8.3f}  s +/- {std:6.3f}  s"


def main():
    if not _redis_available():
        print(  # noqa: T201
            "ERROR: Redis server not reachable on localhost:6379. Start it first."
        )
        return

    print("=" * 90)  # noqa: T201
    print(  # noqa: T201
        f"  RedisTensorDict vs TensorDict benchmark"
        f"  (N={N}, keys={N_KEYS}, feat={FEAT})"
    )
    print(f"  warmup={WARMUP}, rounds={ROUNDS}")  # noqa: T201
    print("=" * 90)  # noqa: T201

    local_td = _make_local_td()
    redis_td = _make_redis_td()

    header = f"{'Operation':<25s} | {'TensorDict':>28s} | {'RedisTensorDict':>28s} | {'Ratio':>8s}"
    print(header)  # noqa: T201
    print("-" * len(header))  # noqa: T201

    for name, bench_fn in BENCHMARKS:
        m_local, s_local = bench_fn(local_td)
        m_redis, s_redis = bench_fn(redis_td)
        ratio = m_redis / m_local if m_local > 0 else float("inf")
        print(  # noqa: T201
            f"{name:<25s} | {_fmt(m_local, s_local):>28s} | "
            f"{_fmt(m_redis, s_redis):>28s} | {ratio:>7.1f}x"
        )

    print("-" * len(header))  # noqa: T201
    print()  # noqa: T201

    redis_td.clear_redis()
    redis_td.close()


if __name__ == "__main__":
    main()
