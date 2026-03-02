# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark TensorDictStore across backends (Redis, Dragonfly) vs local TensorDict.

Run with:
    python benchmarks/storage/bench_backends.py
"""

import importlib
import time

import torch
from tensordict import TensorDict

_has_redis_pkg = importlib.util.find_spec("redis", None) is not None

# Backend configs: (name, port)
BACKENDS = [
    ("redis", 6379),
    ("dragonfly", 6380),
]


def _server_available(host, port):
    if not _has_redis_pkg:
        return False
    import redis

    try:
        r = redis.Redis(host=host, port=port, db=0, socket_connect_timeout=2)
        r.ping()
        r.close()
        return True
    except (redis.ConnectionError, redis.exceptions.ConnectionError, OSError):
        return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N = 10_000
N_KEYS = 5
FEAT = 64
WARMUP = 2
ROUNDS = 10

IDX_INT = 42
IDX_SLICE = slice(100, 356)
IDX_STEP = slice(0, N, 3)
IDX_FANCY = torch.randint(0, N, (256,))
IDX_BOOL = torch.zeros(N, dtype=torch.bool)
IDX_BOOL[torch.randint(0, N, (256,))] = True


def _make_local_td():
    d = {f"key_{i}": torch.randn(N, FEAT) for i in range(N_KEYS)}
    return TensorDict(d, batch_size=[N])


def _make_store_td(backend, port):
    from tensordict.store import TensorDictStore

    td = TensorDictStore(batch_size=[N], db=14, backend=backend, port=port)
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
# Benchmark definitions
# ---------------------------------------------------------------------------


def test_get_single_key(td):
    return _timeit(lambda: td["key_0"])


def test_set_single_key(td):
    v = torch.randn(N, FEAT)
    return _timeit(lambda: td.__setitem__("key_0", v))


def test_keys_iter(td):
    return _timeit(lambda: list(td.keys()))


def test_values_iter(td):
    return _timeit(lambda: list(td.values()))


def test_items_iter(td):
    return _timeit(lambda: list(td.items()))


def test_read_int(td):
    return _timeit(lambda: td[IDX_INT])


def test_read_slice(td):
    return _timeit(lambda: td[IDX_SLICE])


def test_read_step(td):
    return _timeit(lambda: td[IDX_STEP])


def test_read_fancy(td):
    return _timeit(lambda: td[IDX_FANCY])


def test_read_bool(td):
    return _timeit(lambda: td[IDX_BOOL])


def test_write_int(td):
    sub = TensorDict({f"key_{i}": torch.randn(FEAT) for i in range(N_KEYS)}, [])

    def _write():
        td[IDX_INT] = sub

    return _timeit(_write)


def test_write_slice(td):
    n = len(range(*IDX_SLICE.indices(N)))
    sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])

    def _write():
        td[IDX_SLICE] = sub

    return _timeit(_write)


def test_write_step(td):
    n = len(range(*IDX_STEP.indices(N)))
    sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])

    def _write():
        td[IDX_STEP] = sub

    return _timeit(_write)


def test_write_fancy(td):
    n = IDX_FANCY.numel()
    sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])

    def _write():
        td[IDX_FANCY] = sub

    return _timeit(_write)


def test_write_bool(td):
    n = int(IDX_BOOL.sum().item())
    sub = TensorDict({f"key_{i}": torch.randn(n, FEAT) for i in range(N_KEYS)}, [n])

    def _write():
        td[IDX_BOOL] = sub

    return _timeit(_write)


def test_to_tensordict(td):
    return _timeit(lambda: td.to_tensordict())


def test_index_to_td_int(td):
    return _timeit(lambda: td[IDX_INT].to_tensordict())


def test_index_to_td_slice(td):
    return _timeit(lambda: td[IDX_SLICE].to_tensordict())


def test_index_to_td_fancy(td):
    return _timeit(lambda: td[IDX_FANCY].to_tensordict())


BENCHMARKS = [
    ("get single key", test_get_single_key),
    ("set single key", test_set_single_key),
    ("keys() iteration", test_keys_iter),
    ("values() iteration", test_values_iter),
    ("items() iteration", test_items_iter),
    ("read td[int]", test_read_int),
    ("read td[slice]", test_read_slice),
    ("read td[::3]", test_read_step),
    ("read td[fancy]", test_read_fancy),
    ("read td[bool]", test_read_bool),
    ("write td[int]=v", test_write_int),
    ("write td[slice]=v", test_write_slice),
    ("write td[::3]=v", test_write_step),
    ("write td[fancy]=v", test_write_fancy),
    ("write td[bool]=v", test_write_bool),
    ("to_tensordict()", test_to_tensordict),
    ("td[int].to_td()", test_index_to_td_int),
    ("td[slice].to_td()", test_index_to_td_slice),
    ("td[fancy].to_td()", test_index_to_td_fancy),
]


def _fmt(mean, std):
    if mean < 1e-3:
        return f"{mean * 1e6:8.1f} us +/- {std * 1e6:6.1f} us"
    if mean < 1:
        return f"{mean * 1e3:8.2f} ms +/- {std * 1e3:6.2f} ms"
    return f"{mean:8.3f}  s +/- {std:6.3f}  s"


def main():
    # Discover available backends
    available = []
    for name, port in BACKENDS:
        if _server_available("localhost", port):
            available.append((name, port))
            print(f"  {name:12s} (port {port}): available")  # noqa: T201
        else:
            print(f"  {name:12s} (port {port}): NOT available, skipping")  # noqa: T201

    if not available:
        print("ERROR: No store backends available.")  # noqa: T201
        return

    print()  # noqa: T201
    print("=" * (30 + 30 * (1 + len(available))))  # noqa: T201
    print(  # noqa: T201
        f"  TensorDictStore benchmark  (N={N}, keys={N_KEYS}, feat={FEAT})"
    )
    print(f"  warmup={WARMUP}, rounds={ROUNDS}")  # noqa: T201
    print("=" * (30 + 30 * (1 + len(available))))  # noqa: T201
    print()  # noqa: T201

    # Build TDs
    local_td = _make_local_td()
    store_tds = {}
    for name, port in available:
        print(f"  Populating {name} store...", flush=True)  # noqa: T201
        store_tds[name] = _make_store_td(name, port)

    # Header
    cols = ["TensorDict"] + [n for n, _ in available]
    if len(available) >= 2:
        cols.append(f"{available[0][0]}/{available[1][0]}")
    header = f"{'Operation':<25s}"
    for c in cols:
        header += f" | {c:>28s}"
    print(header)  # noqa: T201
    print("-" * len(header))  # noqa: T201

    # Run benchmarks
    for bname, bench_fn in BENCHMARKS:
        m_local, s_local = bench_fn(local_td)
        row = f"{bname:<25s} | {_fmt(m_local, s_local):>28s}"

        means = {}
        for name, _ in available:
            m, s = bench_fn(store_tds[name])
            means[name] = m
            # ratio = m / m_local if m_local > 0 else float("inf")
            row += f" | {_fmt(m, s):>28s}"

        if len(available) >= 2:
            n0, n1 = available[0][0], available[1][0]
            ratio_stores = means[n0] / means[n1] if means[n1] > 0 else float("inf")
            row += f" | {ratio_stores:>7.2f}x"

        print(row)  # noqa: T201

    print("-" * len(header))  # noqa: T201
    print()  # noqa: T201

    # Cleanup
    for td in store_tds.values():
        td.clear_redis()
        td.close()


if __name__ == "__main__":
    main()
