# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Benchmark comparing strategies for writing stacked tensordicts into storage.

This benchmark compares three strategies:
1. Stack first, write once: torch.stack(tds) -> storage[slice] = stacked
2. Lazy stack, write each: iterate and write each element individually
3. Lazy stack, stack onto view: torch.stack(tds, out=storage[slice])

Run with:
    pytest benchmarks/storage/storage_write_test.py -v --benchmark-sort=mean
"""

import torch

from tensordict import TensorDict

try:
    import pytest
except ImportError:
    pytest = None


def get_available_devices():
    """Return list of available devices for testing."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda:0"))
    return devices


def create_storage(N: int, device: torch.device) -> TensorDict:
    """Create a storage tensordict with N elements and 10 fields.

    Fields:
        - image1, image2: shape (N, 3, 64, 64) - image data
        - reward: shape (N,) - scalar reward
        - done: shape (N,) - boolean done flag
        - action: shape (N, 4) - action vector
        - obs1, obs2, obs3: shape (N, 16) - observation vectors
        - info1, info2: shape (N, 8) - info vectors
    """
    return TensorDict(
        image1=torch.zeros(N, 3, 64, 64, device=device),
        image2=torch.zeros(N, 3, 64, 64, device=device),
        reward=torch.zeros(N, device=device),
        done=torch.zeros(N, dtype=torch.bool, device=device),
        action=torch.zeros(N, 4, device=device),
        obs1=torch.zeros(N, 16, device=device),
        obs2=torch.zeros(N, 16, device=device),
        obs3=torch.zeros(N, 16, device=device),
        info1=torch.zeros(N, 8, device=device),
        info2=torch.zeros(N, 8, device=device),
        batch_size=[N],
    )


def create_data_list(M: int, device: torch.device) -> list[TensorDict]:
    """Create a list of M single-element tensordicts to be stacked.

    Each tensordict has the same structure as the storage but with batch_size=[].
    """
    return [
        TensorDict(
            image1=torch.randn(3, 64, 64, device=device),
            image2=torch.randn(3, 64, 64, device=device),
            reward=torch.randn((), device=device),
            done=torch.zeros((), dtype=torch.bool, device=device),
            action=torch.randn(4, device=device),
            obs1=torch.randn(16, device=device),
            obs2=torch.randn(16, device=device),
            obs3=torch.randn(16, device=device),
            info1=torch.randn(8, device=device),
            info2=torch.randn(8, device=device),
            batch_size=[],
        )
        for _ in range(M)
    ]


# Strategy implementations


def stack_then_write(storage: TensorDict, start: int, data_list: list[TensorDict]):
    """Strategy 1: Stack all tensordicts first, then write once to storage."""
    stacked = torch.stack(data_list, dim=0)
    storage[start : start + len(data_list)] = stacked


def lazy_stack_write_each(
    storage: TensorDict, start: int, data_list: list[TensorDict]
):
    """Strategy 2: Write each tensordict individually to storage."""
    for i, td in enumerate(data_list):
        storage[start + i] = td


def lazy_stack_onto_view(
    storage: TensorDict, start: int, data_list: list[TensorDict]
):
    """Strategy 3: Stack directly onto the storage view using out= parameter."""
    view = storage[start : start + len(data_list)]
    torch.stack(data_list, dim=0, out=view)


def lazy_stack_assign_slice(
    storage: TensorDict, start: int, data_list: list[TensorDict]
):
    """Strategy 4: Create lazy stack, assign to slice (uses optimized path)."""
    from tensordict import lazy_stack

    lazy_stacked = lazy_stack(data_list, dim=0)
    storage[start : start + len(data_list)] = lazy_stacked


STRATEGIES = {
    "stack_then_write": stack_then_write,
    "lazy_stack_write_each": lazy_stack_write_each,
    "lazy_stack_onto_view": lazy_stack_onto_view,
    "lazy_stack_assign_slice": lazy_stack_assign_slice,
}


# Benchmark tests (only available when pytest is installed)


if pytest is not None:

    class TestStorageWriteBenchmark:
        """Benchmark tests for storage write strategies."""

        @pytest.fixture(params=[(50000, 1000), (5000, 100)], ids=["N50K_M1K", "N5K_M100"])
        def config(self, request):
            """Parametrize N (storage size) and M (data size)."""
            return request.param

        @pytest.fixture(params=get_available_devices(), ids=lambda d: str(d))
        def device(self, request):
            """Parametrize device (CPU, CUDA if available)."""
            return request.param

        @pytest.mark.parametrize(
            "strategy",
            ["stack_then_write", "lazy_stack_write_each", "lazy_stack_onto_view", "lazy_stack_assign_slice"],
        )
        def test_storage_write(self, benchmark, config, device, strategy):
            """Benchmark storage write strategies."""
            N, M = config
            storage = create_storage(N, device)
            start_idx = 1000  # Write starting at index 1000

            def setup():
                data_list = create_data_list(M, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                return (storage, start_idx, data_list), {}

            def run(storage, start, data_list):
                STRATEGIES[strategy](storage, start, data_list)
                if device.type == "cuda":
                    torch.cuda.synchronize()

            benchmark.pedantic(
                run,
                setup=setup,
                warmup_rounds=2,
                rounds=10,
                iterations=1,
            )


# Standalone benchmark runner for quick testing without pytest-benchmark


def run_standalone_benchmark():
    """Run benchmarks without pytest for quick testing."""
    import time

    print("=" * 80)
    print("TensorDict Storage Write Benchmark")
    print("=" * 80)

    configs = [(50000, 1000), (5000, 100)]
    devices = get_available_devices()

    for device in devices:
        print(f"\nDevice: {device}")
        print("-" * 40)

        for N, M in configs:
            print(f"\n  Config: N={N}, M={M}")
            storage = create_storage(N, device)
            start_idx = 1000

            for strategy_name, strategy_fn in STRATEGIES.items():
                # Warmup
                data_list = create_data_list(M, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                strategy_fn(storage, start_idx, data_list)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                # Benchmark
                times = []
                for _ in range(5):
                    data_list = create_data_list(M, device)
                    if device.type == "cuda":
                        torch.cuda.synchronize()

                    t0 = time.perf_counter()
                    strategy_fn(storage, start_idx, data_list)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    times.append(t1 - t0)

                mean_time = sum(times) / len(times)
                std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
                print(f"    {strategy_name:25s}: {mean_time*1000:8.2f} ms ± {std_time*1000:.2f} ms")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    run_standalone_benchmark()
